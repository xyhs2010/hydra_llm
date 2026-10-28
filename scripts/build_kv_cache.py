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
    """Stores every layer quantized on GPU; returns fp16 K/V for attention.

    K is quantized per-channel (axis=-2, over tokens) and V per-token
    (axis=-1, over head_dim). Each chunk re-quantizes the whole prefix, which
    costs O(T) extra traffic per chunk but keeps the code simple and the scales
    global rather than chunk-local.
    """

    def __init__(self, scheme: str, dtype: torch.dtype):
        self.scheme = scheme
        self.dtype = dtype
        self._k: dict[int, dict[str, torch.Tensor]] = {}
        self._v: dict[int, dict[str, torch.Tensor]] = {}
        self._seq_len = 0

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if layer_idx in self._k:
            k_past = dequantize_tensor(self._k[layer_idx], self.scheme, self.dtype)
            v_past = dequantize_tensor(self._v[layer_idx], self.scheme, self.dtype)
            k_full = torch.cat([k_past, key_states], dim=-2)
            v_full = torch.cat([v_past, value_states], dim=-2)
            del k_past, v_past
        else:
            k_full, v_full = key_states, value_states

        self._k[layer_idx] = quantize_tensor(k_full, self.scheme, axis=-2)
        self._v[layer_idx] = quantize_tensor(v_full, self.scheme, axis=-1)
        self._seq_len = k_full.shape[-2]
        return k_full, v_full

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._seq_len

    def get_max_cache_shape(self) -> int:
        return -1

    def get_max_length(self) -> int:
        return -1

    def get_mask_sizes(self, cache_position):
        # (kv_length, kv_offset): past + current query chunk
        query_length = cache_position.shape[0] if hasattr(cache_position, "shape") else 1
        return self._seq_len + query_length, 0

    @property
    def is_initialized(self) -> bool:
        return bool(self._k)

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


def load_model(path: str, weight_bits: int, dtype: torch.dtype):
    try:  # transformers >= 5 renamed torch_dtype -> dtype
        model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype)
    model.eval()
    if weight_bits < 16:
        try:
            from torchao.quantization import int4_weight_only, int8_weight_only, quantize_
        except ImportError as exc:
            raise SystemExit(
                "torchao is required for weight quantization; install it or pass --weight-bits 16"
            ) from exc
        quant_fn = int8_weight_only() if weight_bits == 8 else int4_weight_only(group_size=128)
        try:
            quantize_(model, quant_fn)  # CPU-first keeps the 16 GB fp16 copy in host RAM
        except Exception:  # noqa: BLE001 - some backends require CUDA
            model.to("cuda")
            quantize_(model, quant_fn)
    model.to("cuda")
    return model


@torch.no_grad()
def dense_prefill(model, ids: torch.Tensor, cache, chunk_size: int):
    total = ids.shape[1]
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


def save_cache(cache: QuantizedCache, path: Path, meta: dict, tokens: list[int]) -> None:
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
def verify_quantization(model, tokenizer, tokens: list[int], scheme: str, dtype: torch.dtype, prefix: int) -> None:
    """Compare the quantized cache's last-token logits against an fp16 cache.

    The last token is held out of the prefill and fed as the decode query, so
    the caches contain exactly the prefix and the check mirrors real decode.
    """
    if prefix < 2:
        return
    prefix_ids = tokens[: prefix - 1]
    last = tokens[prefix - 1]
    ids = torch.tensor([prefix_ids], dtype=torch.long, device="cuda")

    ref_cache = QuantizedCache("fp16", dtype)
    q_cache = QuantizedCache(scheme, dtype)
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
    ap.add_argument("--chunk-size", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=0, help="max samples per (length, task); 0 = all")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_model(args.model, args.weight_bits, dtype)
    model_tag = f"{Path(args.model).name}-w{args.weight_bits}"
    out_root = Path(args.out)

    tasks = args.tasks if args.benchmark == "ruler" else ["pg19"]
    built = skipped = 0

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
                cache = QuantizedCache(args.quant, dtype)
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

                if args.verify and built == 1:
                    verify_quantization(
                        model, tokenizer, sample.context_tokens, args.quant, dtype,
                        prefix=min(sample.length - 1, 4096),
                    )

                del cache, ids
                torch.cuda.empty_cache()

    print(f"done: built={built} skipped={skipped}")


if __name__ == "__main__":
    main()
