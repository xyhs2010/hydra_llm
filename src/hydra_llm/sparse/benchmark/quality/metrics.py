"""decode 结果 → 分数。

decode 的产物是算法无关的中间值（逐位置 NLL 或生成的 token ids），评分函数把它和
sample 的 target 一起变成分数。注意 score 需要 sample、不只是 decode 结果 ——
因为要拿 target 对比（NIAH 比 needle、QA 比参考答案）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from .data import Sample


@dataclass
class DecodeResult:
    """一次 decode 的中间产物。PPL 与生成任务二选一填字段。"""

    sample_id: str
    nll: Tensor | None = None  # PPL：逐位置负对数似然 [seq_len]
    generated_ids: list[int] | None = None  # 生成：解码出的 token ids
    generated_text: str | None = None  # 生成：解码并后处理后的文本


class Metric(Protocol):
    """评分函数：sample（提供 target）+ decode 结果 → 一个或多个分数。"""

    def __call__(self, sample: Sample, result: DecodeResult) -> dict[str, float]: ...


# ---------------------------------------------------------------- PPL


def ppl_metric(sample: Sample, result: DecodeResult) -> dict[str, float]:
    """perplexity = exp(mean NLL)。"""
    if result.nll is None or result.nll.numel() == 0:
        raise ValueError(f"no NLL for sample {sample.sample_id!r}")
    return {"ppl": float(torch.exp(result.nll.mean()))}


# ---------------------------------------------------------------- RULER


_NON_PRINTABLE = re.compile(r"[\x00-\x1f]")


def postprocess_pred(text: str) -> str:
    """Same normalisation as RULER's evaluate.py."""
    return _NON_PRINTABLE.sub("\n", text.strip()).strip()


def _references(sample: Sample) -> list[str]:
    refs = sample.target
    if refs is None:
        raise ValueError(f"sample {sample.sample_id!r} has no target")
    if isinstance(refs, str):
        refs = [refs]
    return [str(r) for r in refs]


def answer_hits(sample: Sample, result: DecodeResult) -> list[bool]:
    """Per-answer hit flags, aligned with `sample.target`.

    Exposed separately from `ruler_metric` so the runner can aggregate by
    needle depth without re-parsing the prediction.
    """
    text = (result.generated_text or "").lower()
    return [ref.lower() in text for ref in _references(sample)]


def ruler_metric(sample: Sample, result: DecodeResult) -> dict[str, float]:
    """RULER's official `string_match_all`: fraction of references found in the
    prediction, averaged over samples, in percent.

    Note this returns the single-sample contribution; the runner averages it.
    """
    hits = answer_hits(sample, result)
    return {"score": 100.0 * sum(hits) / len(hits)}


# LongBench QA (to be added):
#   qa_metric(sample, result) -> {"f1": ...}
