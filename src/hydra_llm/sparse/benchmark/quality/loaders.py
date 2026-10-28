"""原始数据 → Sample 流的 loader。

当前实现 PG19（PPL 用）。约定：
- 原始文件在仓库外/`data/` 下，路径由调用方传入；
- tokenizer 由调用方传入（harness 负责模型/tokenizer 的生命周期）；
- 长文按 token 切窗，窗口之间不重叠。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from .data import Sample

# RULER task -> number of tokens to generate (from the official synthetic/constants.py)
RULER_TOKENS_TO_GENERATE = {
    "niah_single_2": 128,
    "niah_multikey_1": 128,
    "niah_multivalue": 128,
    "niah_multiquery": 128,
    "cwe": 120,
    "fwe": 50,
}


class Tokenizer(Protocol):
    """只用到 tokenizer 的这两个能力，避免依赖 transformers 类型。"""

    def __call__(self, text: str, *, add_special_tokens: bool = ...) -> object: ...

    def decode(self, ids: list[int]) -> str: ...


def _encode(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False).input_ids


def load_pg19(
    root: str | Path,
    tokenizer,
    *,
    length: int,
    n_windows: int | None = None,
    prefix: str = "pg19",
) -> list[Sample]:
    """把 PG19 的纯文本文件拼成 token 流，切成固定长度的 PPL 窗口。

    PG19 test 的 100 本书每本约 6~10 万 token，128K/256K 窗口需要跨书拼接，
    所以这里统一做法是：按文件名排序后顺序拼接，再切非重叠窗口。
    """
    root = Path(root)
    files = sorted(root.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"no .txt files under {root}")

    stream: list[int] = []
    sources: list[str] = []
    for path in files:
        ids = _encode(tokenizer, path.read_text(encoding="utf-8", errors="replace"))
        if ids:
            stream.extend(ids)
            sources.extend([path.name] * len(ids))

    samples: list[Sample] = []
    for i in range(0, len(stream) - length + 1, length):
        if n_windows is not None and len(samples) >= n_windows:
            break
        start = i
        end = i + length
        books = sorted(set(sources[start:end]))
        samples.append(
            Sample(
                sample_id=f"{prefix}_{length}_{len(samples):04d}",
                context_tokens=stream[start:end],
                metadata={
                    "benchmark": "pg19",
                    "length": length,
                    "books": books,
                },
            )
        )
    return samples


def load_ruler(
    root: str | Path,
    tokenizer,
    *,
    length: int,
    task: str,
    split: str = "test",
) -> list[Sample]:
    """Load one RULER (length, task) cell into Samples.

    RULER's `input` already contains the full prompt (template + haystack +
    question + answer prefix), so it fills `context_tokens` and
    `query_tokens` stays empty. `outputs` is the reference list.
    """
    path = Path(root) / f"L{length}" / task / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing RULER file: {path}")

    samples: list[Sample] = []
    for line in path.open():
        row = json.loads(line)
        samples.append(
            Sample(
                sample_id=f"ruler_{task}_{length}_{row['index']}",
                context_tokens=_encode(tokenizer, row["input"]),
                target=row["outputs"],
                metadata={
                    "benchmark": "ruler",
                    "task": task,
                    "length": length,
                    "index": row["index"],
                    "tokens_to_generate": RULER_TOKENS_TO_GENERATE[task],
                    # NIAH only; None for aggregation tasks (cwe/fwe).
                    "needle_positions": row.get("needle_positions"),
                    "needle_depths": row.get("needle_depths"),
                },
            )
        )
    return samples
