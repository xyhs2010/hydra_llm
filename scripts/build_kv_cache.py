"""Build dense-prefill KV caches for repeated sparse-attention sweeps.

Prefill is always dense (no sparsification), so one cache per sample is shared
by every algorithm / retention ratio. This script produces that cache and
writes it to disk in a resumable layout.

Memory strategy (important on a 32 GB card at 256K):

  A standard HF `DynamicCache` keeps all 36 layers' fp16 K/V on the GPU —
  36 GB at 256K, which does not fit. `QuantizedCache` instead stores every
  layer quantized on the GPU (q8 ≈ 18 GB, q4 ≈ 9 GB) and dequantizes only the
  current layer to fp16 for the attention call (≈ 1 GB at 256K).

Quantization scheme (per KVQuant / KIVI):
  - K: per-channel, scales over tokens       -> [heads, 1, head_dim]
  - V: per-token,   scales over head_dim     -> [heads, seq, 1]
  - q8: int8 symmetric; q4: packed signed 4-bit, two values per byte.

Layout:
  <out>/<model_tag>/<quant>/<benchmark>/L<length>/<task>/<sample_id>/
      meta.json
      tokens.safetensors
      layer_00.safetensors ... layer_35.safetensors

Each layer file holds either {k, v} (fp16) or {k_q, k_scale, v_q, v_scale}.

Usage (RULER, 256K only, q8):
  python scripts/build_kv_cache.py \
      --model data/models/Qwen3-8B-Base --weight-bits 4 \
      --benchmark ruler --lengths 131072 262144 \
      --tasks niah_single_2 niah_multikey_1 niah_multivalue niah_multiquery cwe fwe \
      --quant q8 --out data/kv_cache --chunk-size 8192

Smoke test:
  ... --limit 1 --verify
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import Cache

from hydra_llm.sparse.benchmark.quality.loaders import load_pg19, load_ruler


# --------------------------------------------------------------------------- quantization


def _scale_for(x: torch.Tensor, axis: int, qmax: int, eps: float = 1e-8) -> torch.Tensor:
    return (x.abs().amax(dim=axis, keepdim=True).clamp_min(eps) / qmax).to(torch.float32)


def _quant_signed(x: torch.Tensor, scale: torch.Tensor, qmin: int, qmax: int) -> torch.Tensor:
    return torch.round(x.float() / scale.float()).clamp(qmin, qmax).to(torch.int8)


def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """Pack signed 4-bit values two per byte along the last dimension."""
    if q.shape[-1] % 2:
        raise ValueError("head_dim must be even for int4 packing")
    lo = (q[..., 0::2] & 0xF).to(torch.uint8)
    hi = (q[..., 1::2] & 0xF).to(torch.uint8)
    return lo | (hi << 4)


def unpack_int4(p: torch.Tensor) -> torch.Tensor:
    lo = (p & 0xF).to(torch.int8)
    hi = ((p >> 4) & 0xF).to(torch.int8)
    lo = torch.where(lo > 7, lo - 16, lo)
    hi = torch.where(hi > 7, hi - 16, hi)
    out = torch.empty(p.shape[:-1] + (p.shape[-1] * 2,), dtype=torch.int8, device=p.device)
    out[..., 0::2] = lo
    out[..., 1::2] = hi
    return out


def quantize_tensor(x: torch.Tensor, scheme: str, axis: int) -> dict[str, torch.Tensor]:
    if scheme == "fp16":
        return {"x": x.contiguous()}
    if scheme == "q8":
        scale = _scale_for(x, axis, 127)
        return {"x_q": _quant_signed(x, scale, -127, 127), "x_scale": scale}
    if scheme == "q4":
        scale = _scale_for(x, axis, 7)
        q = _quant_signed(x, scale, -8, 7)
        return {"x_q": pack_int4(q), "x_scale": scale}
    raise ValueError(f"unknown quant scheme {scheme!r}")


def dequantize_tensor(d: dict[str, torch.Tensor], scheme: str, dtype: torch.dtype) -> torch.Tensor:
    if scheme == "fp16":
        return d["x"].to(dtype)
    q = d["x_q"]
    if scheme == "q4":
        q = unpack_int4(q)
    return (q.float() * d["x_scale"].float()).to(dtype)


# --------------------------------------------------------------------------- cache


class QuantizedCache(Cache):
    """Stores every layer quantized; returns fp16 K/V for attention.

    K is quantized per-channel (axis=-2, over tokens) and V per-token
    (axis=-1, over head_dim). Each chunk re-quantizes the whole prefix, which
    costs O(T) extra traffic per chunk but keeps the code simple and the scales
    global rather than chunk-local.

    `store_device="cpu"` keeps the quantized cache in host RAM and streams only
    the current layer to the GPU (the project's host-KV layout). On a 32 GB
    card this is what makes 256K + q8 fit: GPU holds weights + one layer's fp16
    K/V (~1 GB) instead of all 36 layers' q8 KV (~18 GB).
    """

    def __init__(self, scheme: str, dtype: torch.dtype, store_device: str = "cuda"):
        self.scheme = scheme
        self.dtype = dtype
        self.store_device = store_device
        self._k: dict[int, dict[str, torch.Tensor]] = {}
        self._v: dict[int, dict[str, torch.Tensor]] = {}
        self._seq_len = 0

    def _to_store(self, d: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {n: t.detach().to(self.store_device) for n, t in d.items()}

    def _from_store(self, d: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {n: t.to("cuda") for n, t in d.items()}

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if layer_idx in self._k:
            k_past = dequantize_tensor(self._from_store(self._k[layer_idx]), self.scheme, self.dtype)
            v_past = dequantize_tensor(self._from_store(self._v[layer_idx]), self.scheme, self.dtype)
            k_full = torch.cat([k_past, key_states], dim=-2)
            v_full = torch.cat([v_past, value_states], dim=-2)
            del k_past, v_past
        else:
            k_full, v_full = key_states, value_states

        self._k[layer_idx] = self._to_store(quantize_tensor(k_full, self.scheme, axis=-2))
        self._v[layer_idx] = self._to_store(quantize_tensor(v_full, self.scheme, axis=-1))
        self._seq_len = k_full.shape[-2]
        return k_full, v_full

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._seq_len

    def get_max_cache_shape(self) -> int:
        return -1

    def get_max_length(self) -> int:
        return -1

    def get_mask_sizes(self, q_length: int, layer_idx: int = 0):
        # (kv_length, kv_offset): past + current query chunk
        return self._seq_len + int(q_length), 0

    @property
    def is_initialized(self) -> bool:
        return bool(self._k)

    @property
    def is_sliding(self) -> list[bool]:
        # modeling code does `False in past_key_values.is_sliding`
        return [False] * max(1, len(self._k))

    @property
    def is_compileable(self) -> bool:
        return False

    def __len__(self) -> int:
        return len(self._k)

    def reset(self) -> None:
        self._k.clear()
        self._v.clear()
        self._seq_len = 0

    # Used by save_cache
    def layers(self):
        return sorted(self._k)


# --------------------------------------------------------------------------- model / prefill


def load_model(path: str, weight_bits: int, dtype: torch.dtype, attn: str = "flex_attention"):
    if attn == "hydra":
        from hydra_flash_attention import register_attention

        register_attention()
    kwargs = {"attn_implementation": attn}
    try:  # transformers >= 5 renamed torch_dtype -> dtype
        model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, **kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, **kwargs)
    model.eval()
    if weight_bits < 16:
        try:
            # torchao >= 0.18 uses config objects
            from torchao.quantization import (
                Int4WeightOnlyConfig,
                Int8WeightOnlyConfig,
                quantize_,
            )

            if weight_bits == 8:
                quant_fn = Int8WeightOnlyConfig()
            else:
                if dtype != torch.bfloat16:
                    raise SystemExit("int4 TILE_PACKED_TO_4D requires --dtype bfloat16")
                # TILE_PACKED_TO_4D is the only int4 packing format that works
                # on this stack (PLAIN/PRESHUFFLED need mslk, PLAIN_INT32 is
                # CPU-only). It requires CUDA + bfloat16.
                try:
                    from torchao.quantization.quantize_.workflows.int4.int4_packing_format import (
                        Int4PackingFormat,
                    )

                    quant_fn = Int4WeightOnlyConfig(
                        group_size=128,
                        int4_packing_format=Int4PackingFormat.TILE_PACKED_TO_4D,
                    )
                except ImportError:
                    quant_fn = Int4WeightOnlyConfig(group_size=128)
        except ImportError:
            try:
                # older torchao API
                from torchao.quantization import (
                    int4_weight_only,
                    int8_weight_only,
                    quantize_,
                )

                quant_fn = (
                    int8_weight_only()
                    if weight_bits == 8
                    else int4_weight_only(group_size=128)
                )
            except ImportError as exc:
                raise SystemExit(
                    f"torchao weight quantization unavailable ({exc}); "
                    "install torchao or pass --weight-bits 16"
                ) from exc
        # int4 tile-packed tensors are CUDA-only, so quantize after moving.
        model.to("cuda", dtype)
        quantize_(model, quant_fn)
    model.to("cuda")
    return model


@torch.no_grad()
def dense_prefill(model, ids: torch.Tensor, cache, chunk_size: int):
    total = ids.shape[1]
    if chunk_size <= 0:
        chunk_size = total  # one pass; flash attention handles the full context
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        model(
            input_ids=ids[:, start:end],
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
    return cache


# --------------------------------------------------------------------------- io


def sample_dir(root: Path, model_tag: str, quant: str, benchmark: str, length: int, task: str, sid: str) -> Path:
    return root / model_tag / quant / benchmark / f"L{length}" / task / sid


def cache_is_complete(path: Path, num_layers: int) -> bool:
    meta = path / "meta.json"
    if not meta.exists():
        return False
    return all((path / f"layer_{i:02d}.safetensors").exists() for i in range(num_layers))


class CacheSnapshot:
    """CPU copy of a QuantizedCache, safe to hand to a writer thread."""

    def __init__(self, cache: QuantizedCache):
        self._k = {l: {n: t.detach().cpu() for n, t in d.items()} for l, d in cache._k.items()}
        self._v = {l: {n: t.detach().cpu() for n, t in d.items()} for l, d in cache._v.items()}

    def layers(self):
        return sorted(self._k)


def save_cache(cache: CacheSnapshot, path: Path, meta: dict, tokens: list[int]) -> None:
    tmp = path.parent / (path.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    save_file({"tokens": torch.tensor(tokens, dtype=torch.int32)}, str(tmp / "tokens.safetensors"))
    for layer in cache.layers():
        tensors: dict[str, torch.Tensor] = {}
        for key, store in (("k", cache._k[layer]), ("v", cache._v[layer])):
            for name, tensor in store.items():
                # fp16:  {"x": t}          -> "k" / "v"
                # q8/q4: {"x_q": t, ...}   -> "k_q", "k_scale", ...
                out_name = key if name == "x" else f"{key}_{name[2:]}"
                tensors[out_name] = tensor.detach().contiguous().cpu()
        save_file(tensors, str(tmp / f"layer_{layer:02d}.safetensors"))
        del tensors

    meta = dict(meta)
    meta["num_layers"] = len(cache.layers())
    (tmp / "meta.json").write_text(json.dumps(meta, indent=2))
    tmp.rename(path)


# --------------------------------------------------------------------------- verify


@torch.no_grad()
def verify_quantization(model, tokenizer, tokens: list[int], scheme: str, dtype: torch.dtype, prefix: int, store_device: str = "cuda") -> None:
    """Compare the quantized cache's last-token logits against an fp16 cache.

    The last token is held out of the prefill and fed as the decode query, so
    the caches contain exactly the prefix and the check mirrors real decode.
    """
    if prefix < 2:
        return
    prefix_ids = tokens[: prefix - 1]
    last = tokens[prefix - 1]
    ids = torch.tensor([prefix_ids], dtype=torch.long, device="cuda")

    ref_cache = QuantizedCache("fp16", dtype, store_device)
    q_cache = QuantizedCache(scheme, dtype, store_device)
    dense_prefill(model, ids, ref_cache, chunk_size=1024)
    dense_prefill(model, ids, q_cache, chunk_size=1024)

    def last_logits(cache):
        out = model(
            input_ids=torch.tensor([[last]], dtype=torch.long, device="cuda"),
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
        return out.logits[:, -1, :].float()

    ref = last_logits(ref_cache)
    got = last_logits(q_cache)
    diff = (ref - got).abs()
    print(
        f"verify ({scheme}, prefix={prefix}): "
        f"max|dlogit|={diff.max().item():.4f} mean|dlogit|={diff.mean().item():.4f} "
        f"top1_match={bool(ref.argmax(-1) == got.argmax(-1))}"
    )
    del ref_cache, q_cache, ids, ref, got
    torch.cuda.empty_cache()


# --------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="data/models/Qwen3-8B-Base")
    ap.add_argument("--weight-bits", type=int, default=4, choices=(16, 8, 4))
    ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16"))
    ap.add_argument("--benchmark", default="ruler", choices=("ruler", "pg19"))
    ap.add_argument("--lengths", type=int, nargs="+", default=[262144])
    ap.add_argument("--tasks", nargs="+", default=["niah_single_2", "niah_multikey_1", "niah_multivalue", "niah_multiquery", "cwe", "fwe"])
    ap.add_argument("--ruler-root", default="data/raw/ruler")
    ap.add_argument("--pg19-root", default="data/raw/pg19/test")
    ap.add_argument("--n-windows", type=int, default=10, help="PG19 windows per length")
    ap.add_argument("--out", default="data/kv_cache")
    ap.add_argument("--quant", default="q8", choices=("fp16", "q8", "q4"))
    ap.add_argument("--kv-store", default="gpu", choices=("gpu", "cpu"), help="where quantized KV lives; cpu = host-KV")
    ap.add_argument("--attn", default="hydra", choices=("hydra", "flex_attention", "sdpa"),
                    help="hydra = mask-free chunked flash attention (recommended)")
    ap.add_argument("--chunk-size", type=int, default=0, help="tokens per prefill chunk; 0 = whole sequence")
    ap.add_argument("--limit", type=int, default=0, help="max samples per (length, task); 0 = all")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--sync-write", action="store_true", help="write in the main thread (debug)")
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    if not Path(args.model).exists():
        raise SystemExit(
            f"model path not found: {args.model}\n"
            "Download it first, e.g.:\n"
            "  export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1\n"
            "  unset ALL_PROXY all_proxy HTTPS_PROXY HTTP_PROXY https_proxy http_proxy\n"
            "  python -c \"from huggingface_hub import snapshot_download; "
            "snapshot_download('Qwen/Qwen3-8B-Base', local_dir='data/models/Qwen3-8B-Base')\"\n"
            "or pass --model /path/to/local/model"
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_model(args.model, args.weight_bits, dtype, args.attn)
    model_tag = f"{Path(args.model).name}-w{args.weight_bits}"
    out_root = Path(args.out)

    tasks = args.tasks if args.benchmark == "ruler" else ["pg19"]
    built = skipped = 0
    executor = ThreadPoolExecutor(max_workers=1)
    pending: list = []

    def check_pending() -> None:
        for future in list(pending):
            if future.done():
                pending.remove(future)
                exc = future.exception()
                if exc is not None:
                    raise exc

    for length in args.lengths:
        for task in tasks:
            if args.benchmark == "ruler":
                samples = load_ruler(args.ruler_root, tokenizer, length=length, task=task)
            else:
                samples = load_pg19(args.pg19_root, tokenizer, length=length, n_windows=args.n_windows)

            if args.limit:
                samples = samples[: args.limit]

            for sample in samples:
                path = sample_dir(out_root, model_tag, args.quant, args.benchmark, length, task, sample.sample_id)
                num_layers = model.config.num_hidden_layers
                if cache_is_complete(path, num_layers) and not args.overwrite:
                    skipped += 1
                    continue

                print(f"[build] {sample.sample_id} tokens={sample.length} -> {path}")
                ids = torch.tensor([sample.context_tokens], dtype=torch.long, device="cuda")
                store_device = "cuda" if args.kv_store == "gpu" else "cpu"
                cache = QuantizedCache(args.quant, dtype, store_device)
                try:
                    dense_prefill(model, ids, cache, args.chunk_size)
                except torch.cuda.OutOfMemoryError:
                    print("  OOM: lower --chunk-size, use a smaller --quant, or quantize weights")
                    raise

                meta = {
                    "sample_id": sample.sample_id,
                    "benchmark": args.benchmark,
                    "task": task,
                    "length": length,
                    "seq_len": int(ids.shape[1]),
                    "model": args.model,
                    "weight_bits": args.weight_bits,
                    "quant": args.quant,
                    "dtype": args.dtype,
                    "chunk_size": args.chunk_size,
                    "num_kv_heads": model.config.num_key_value_heads,
                    "head_dim": model.config.head_dim,
                    "scheme": {
                        "k": "per-channel over tokens",
                        "v": "per-token over head_dim",
                    },
                }
                save_cache(cache, path, meta, sample.context_tokens)
                built += 1
                print(f"  peak GPU: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
                torch.cuda.reset_peak_memory_stats()

                if args.verify and built == 1:
                    verify_quantization(
                        model, tokenizer, sample.context_tokens, args.quant, dtype,
                        prefix=min(sample.length - 1, 4096), store_device=store_device,
                    )

                del cache, ids
                torch.cuda.empty_cache()

    executor.shutdown(wait=True)
    check_pending()
    print(f"done: built={built} skipped={skipped}")


if __name__ == "__main__":
    main()
