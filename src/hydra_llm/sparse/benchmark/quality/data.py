"""原始数据源 → 统一的 Sample。

不同 benchmark 的原始文件格式各异（PG19 纯文本、RULER 是 JSON、LongBench 是
JSONL），但都归一到同一个 Sample：一段需要 dense prefill 的长上下文，外加一个
叠加在上面的任务。loader 负责把原始数据 + tokenizer 转成 token id。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol


@dataclass
class Sample:
    """一次评测的最小单元。

    - PPL：context_tokens 就是被评分的文档，query_tokens 空、target 为 None，
      分数来自对 context 自身的逐位置 NLL。
    - 生成类（RULER NIAH / LongBench）：context_tokens 是长上下文，query_tokens
      是追加在上下文后的提问/指令，target 是用于评分的 ground truth。
    """

    sample_id: str  # 全局唯一；KV 池的 key、报告的标识
    context_tokens: list[int] = field(default_factory=list)  # 长上下文（已 tokenize），dense prefill 一次
    query_tokens: list[int] = field(default_factory=list)  # 生成任务追加的提问（PPL 为空）
    target: Any = None  # ground truth（PPL 为 None）
    metadata: dict[str, Any] = field(default_factory=dict)  # benchmark/长度/needle 位置，画图筛选用

    @property
    def length(self) -> int:
        return len(self.context_tokens)


class SampleSource(Protocol):
    """loader 的统一接口：每个 benchmark 提供一个可迭代的 Sample 流。"""

    def __iter__(self) -> Iterator[Sample]: ...

    def __len__(self) -> int: ...


# 各 benchmark loader（待实现，签名如下）：
#   load_pg19(path, tokenizer, *, n_samples, max_len) -> SampleSource
#   load_ruler_niah(path, tokenizer, *, length, needle_pos) -> SampleSource
#   load_ruler_common_words(...) -> SampleSource
#   load_longbench(task, tokenizer, ...) -> SampleSource
