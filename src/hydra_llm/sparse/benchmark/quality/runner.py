"""评测循环与汇总。

输入：模型 + 样本流（一种 benchmark × 长度）。
输出：逐样本记录 + 按 (benchmark, task, length) 聚合的 mean±std，
NIAH 另外按 needle 深度分桶。

当前是 dense 路径（algo=None）。稀疏算法接入时，替换 evaluate_ppl /
evaluate_ruler 内部的 attention 调用即可，本文件的聚合逻辑不变。
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Iterable, Sequence

from .data import Sample
from .metrics import answer_hits, ppl_metric, ruler_metric
from .ppl import evaluate_ppl_decode, evaluate_ppl_full
from .ruler import evaluate_ruler

Row = dict


def run_ppl(
    model,
    samples: Iterable[Sample],
    *,
    n_positions: int | None = 128,
    chunk_size: int = 1024,
    device=None,
) -> list[Row]:
    """`n_positions=None` runs the classic full-sequence PPL; otherwise dense
    prefill + sparse decode at `n_positions` sampled positions."""
    rows: list[Row] = []
    for sample in samples:
        if n_positions is None:
            result = evaluate_ppl_full(model, sample, chunk_size=chunk_size, device=device)
        else:
            result = evaluate_ppl_decode(
                model, sample, n_positions=n_positions, chunk_size=chunk_size, device=device
            )
        score = ppl_metric(sample, result)
        nll = result.nll
        rows.append(
            {
                "sample_id": sample.sample_id,
                "benchmark": sample.metadata.get("benchmark", "ppl"),
                "task": sample.metadata.get("task", "ppl"),
                "length": sample.metadata.get("length", sample.length),
                **score,
                "nll_sum": float(nll.sum()),
                "n_tokens": int(nll.numel()),
            }
        )
    return rows


def run_ruler(
    model,
    tokenizer,
    samples: Iterable[Sample],
    *,
    device=None,
) -> list[Row]:
    rows: list[Row] = []
    for sample in samples:
        result = evaluate_ruler(model, tokenizer, sample, device=device)
        score = ruler_metric(sample, result)
        hits = answer_hits(sample, result)
        depths = sample.metadata.get("needle_depths")
        rows.append(
            {
                "sample_id": sample.sample_id,
                "benchmark": sample.metadata.get("benchmark", "ruler"),
                "task": sample.metadata.get("task", "ruler"),
                "length": sample.metadata.get("length", sample.length),
                **score,
                "hits": hits,
                "depths": depths,
                "prediction": result.generated_text,
            }
        )
    return rows


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return mean, std


def summarize(
    rows: Iterable[Row],
    *,
    keys: tuple[str, ...] = ("benchmark", "task", "length"),
) -> list[Row]:
    """Aggregate rows into one entry per group.

    PPL groups report perplexity as exp(total NLL / total tokens) — the standard
    definition — plus the std of per-sample PPL. RULER groups report mean±std of
    the per-sample score.
    """
    groups: dict[tuple, list[Row]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k) for k in keys)].append(row)

    out: list[Row] = []
    for group_key, items in sorted(groups.items(), key=lambda kv: str(kv[0])):
        entry = dict(zip(keys, group_key))
        if "ppl" in items[0]:
            total_nll = sum(r["nll_sum"] for r in items)
            total_tokens = sum(r["n_tokens"] for r in items)
            mean, std = _mean_std([r["ppl"] for r in items])
            entry.update(
                {
                    "ppl": math.exp(total_nll / total_tokens),
                    "ppl_mean": mean,
                    "ppl_std": std,
                    "n": len(items),
                }
            )
        else:
            mean, std = _mean_std([r["score"] for r in items])
            entry.update({"score": mean, "score_std": std, "n": len(items)})
        out.append(entry)
    return out


# Depth buckets for NIAH: beginning / middle / end. The middle is where
# sink+recent sparse algorithms are expected to fail.
DEPTH_BUCKETS = (("start", 0.0, 0.1), ("middle", 0.1, 0.9), ("end", 0.9, 1.01))


def summarize_by_depth(
    rows: Iterable[Row],
    *,
    keys: tuple[str, ...] = ("benchmark", "task", "length"),
) -> list[Row]:
    """Per-answer hit rate bucketed by needle depth (NIAH tasks only)."""
    groups: dict[tuple, list[Row]] = defaultdict(list)
    for row in rows:
        if row.get("depths") and row.get("hits"):
            groups[tuple(row.get(k) for k in keys)].append(row)

    out: list[Row] = []
    for group_key, items in sorted(groups.items(), key=lambda kv: str(kv[0])):
        entry = dict(zip(keys, group_key))
        for name, lo, hi in DEPTH_BUCKETS:
            hits = [
                hit
                for r in items
                for hit, depth in zip(r["hits"], r["depths"])
                if depth is not None and lo <= depth < hi
            ]
            entry[name] = 100.0 * sum(hits) / len(hits) if hits else None
            entry[f"{name}_n"] = len(hits)
        out.append(entry)
    return out
