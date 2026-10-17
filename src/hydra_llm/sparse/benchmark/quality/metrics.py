"""decode 结果 → 分数。

decode 的产物是算法无关的中间值（逐位置 NLL 或生成的 token ids），评分函数把它和
sample 的 target 一起变成分数。注意 score 需要 sample、不只是 decode 结果 ——
因为要拿 target 对比（NIAH 比 needle、QA 比参考答案）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from torch import Tensor


@dataclass
class DecodeResult:
    """一次 decode 的中间产物。PPL 与生成任务二选一填字段。"""

    sample_id: str
    nll: Tensor | None = None  # PPL：逐位置负对数似然 [seq_len]
    generated_ids: list[int] | None = None  # 生成：解码出的 token ids


class Metric(Protocol):
    """评分函数：sample（提供 target）+ decode 结果 → 一个或多个分数。"""

    def __call__(self, sample: "Sample", result: DecodeResult) -> dict[str, float]: ...


# 具体 metric（待实现）：
#   ppl_metric(sample, result)  -> {"ppl": exp(mean(result.nll))}
#   niah_metric(sample, result) -> {"acc": 解码文本是否命中 sample.target}
#   qa_metric(sample, result)   -> {"f1": ...}   # LongBench QA
