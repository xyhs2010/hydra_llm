"""质量评测（阶段一）：dense prefill 一次 → 复用 KV → decode 换稀疏算法对比。

三个共享的数据结构都在这里导出，benchmark 各自实现评测逻辑：
- Sample / SampleSource（原始数据源）
- KVCache / KVCachePool（dense prefill KV cache）
- DecodeResult / Metric（decode 结果 → 分数）
"""

from .data import Sample, SampleSource
from .kv_cache import KVCache, KVCachePool
from .loaders import load_pg19, load_ruler
from .metrics import DecodeResult, Metric, answer_hits, ppl_metric, ruler_metric
from .runner import run_ppl, run_ruler, summarize, summarize_by_depth

__all__ = [
    "Sample",
    "SampleSource",
    "KVCache",
    "KVCachePool",
    "DecodeResult",
    "Metric",
    "load_pg19",
    "load_ruler",
    "answer_hits",
    "ppl_metric",
    "ruler_metric",
    "run_ppl",
    "run_ruler",
    "summarize",
    "summarize_by_depth",
]
