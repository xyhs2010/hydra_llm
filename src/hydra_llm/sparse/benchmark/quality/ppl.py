"""PPL（维度 A：语言建模 / dense 逼近）。

两条路径：

1. `evaluate_ppl_full` —— 经典 PPL：整条序列逐位置算 NLL，稀疏 attention 在
   **每个位置**都生效。必须分块（256K × vocab 的 logits 约 80 GB），且每个
   稀疏配置都要完整前向一遍。

2. `evaluate_ppl_decode` —— 本项目基准流程：把被测位置之前的全部 token 当作
   **dense prefill**（不稀疏化），只在 decode 位置上施加稀疏。一次 dense
   prefill 的 KV 可被所有算法复用，成本从 O(T) 降到 O(算法数 × K)。

因果性保证了复用是精确的：位置 j 的 K/V 只依赖 x_0..x_j，因此 prefill 得到
的 0..t-1 的 K/V 与「只为预测 t+1 而做的 prefill」完全一致。algo=None 时，
两条路径应给出逐位相同的 NLL —— 这是实现的正确性基准。
"""

from __future__ import annotations

import hashlib
import random

import torch
import torch.nn.functional as F

from .data import Sample
from .metrics import DecodeResult


@torch.no_grad()
def dense_prefill(
    model,
    ids: torch.Tensor,
    *,
    chunk_size: int = 1024,
):
    """Dense, chunked prefill. Returns the cache covering every position."""
    past = None
    total = ids.shape[1]
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        out = model(input_ids=ids[:, start:end], past_key_values=past, use_cache=True)
        past = out.past_key_values
    return past


@torch.no_grad()
def evaluate_ppl_full(
    model,
    sample: Sample,
    *,
    chunk_size: int = 1024,
    device: torch.device | str | None = None,
) -> DecodeResult:
    """Classic teacher-forced NLL over every position."""
    if device is None:
        device = next(model.parameters()).device
    ids = torch.tensor([sample.context_tokens], dtype=torch.long, device=device)
    total = ids.shape[1]
    if total < 2:
        raise ValueError(f"sample {sample.sample_id!r} is too short for PPL")

    past = None
    nll_chunks: list[torch.Tensor] = []
    start = 0
    while start < total - 1:
        end = min(start + chunk_size, total)
        out = model(input_ids=ids[:, start:end], past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits  # [1, end-start, vocab]

        # targets are already clipped at the sequence end, so in the last chunk
        # the target slice is one shorter; drop only the logit that has no target.
        targets = ids[:, start + 1 : end + 1]
        if end == total:
            logits = logits[:, :-1]

        if targets.shape[1] > 0:
            nll = F.cross_entropy(
                logits.transpose(1, 2).float(), targets, reduction="none"
            )[0]
            nll_chunks.append(nll.cpu())
        start = end

    return DecodeResult(sample_id=sample.sample_id, nll=torch.cat(nll_chunks))


def _sample_positions(total: int, n: int, sample_id: str) -> list[int]:
    """Positions t whose target is x[t+1]; stratified (evenly spaced) sampling.

    Even spacing is still unbiased for the mean NLL and has lower variance than
    uniform random sampling. Deterministic per sample, seeded by sample_id.
    """
    last = total - 2  # t+1 must exist
    if last < 0:
        raise ValueError("sequence too short for PPL")
    if n >= last + 1:
        return list(range(last + 1))
    rng = random.Random(int(hashlib.md5(sample_id.encode()).hexdigest()[:8], 16))
    width = (last + 1) / n
    positions = sorted({min(last, int(rng.uniform(i * width, (i + 1) * width))) for i in range(n)})
    return positions


@torch.no_grad()
def evaluate_ppl_decode(
    model,
    sample: Sample,
    *,
    n_positions: int = 128,
    positions: list[int] | None = None,
    chunk_size: int = 1024,
    device: torch.device | str | None = None,
) -> DecodeResult:
    """Dense prefill + sparse decode NLL at sampled positions.

    Positions are processed from the end backwards so the cache can be cropped
    monotonically instead of copied per position.
    """
    if device is None:
        device = next(model.parameters()).device
    ids = torch.tensor([sample.context_tokens], dtype=torch.long, device=device)
    total = ids.shape[1]
    if total < 2:
        raise ValueError(f"sample {sample.sample_id!r} is too short for PPL")
    if positions is None:
        positions = _sample_positions(total, n_positions, sample.sample_id)

    past = dense_prefill(model, ids, chunk_size=chunk_size)

    nll_by_pos: dict[int, float] = {}
    for t in sorted(positions, reverse=True):
        # keep K/V for 0..t-1, then feed x_t at position t (causal, and the
        # cache length supplies position_ids automatically). transformers'
        # crop() takes a negative count of tokens to remove.
        current = past.get_seq_length()
        if current > t:
            past.crop(-(current - t))
        out = model(
            input_ids=ids[:, t : t + 1],
            past_key_values=past,
            use_cache=True,
            logits_to_keep=1,
        )
        target = ids[:, t + 1]
        nll = F.cross_entropy(out.logits[:, -1, :].float(), target, reduction="none")
        nll_by_pos[t] = float(nll[0])

    ordered = [nll_by_pos[t] for t in sorted(nll_by_pos)]
    return DecodeResult(
        sample_id=sample.sample_id,
        nll=torch.tensor(ordered),
    )
