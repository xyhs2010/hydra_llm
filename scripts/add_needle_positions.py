"""Add `needle_positions` to the generated RULER NIAH jsonl files.

RULER's own `token_position_answer` is a single int locating only `outputs[0]`.
Multi-needle tasks need every needle's position for depth bucketing, so this
script locates each answer's token sequence in the input and writes the
positions (aligned with `outputs`) into the row.

Only NIAH tasks get the field: cwe/fwe are aggregation tasks whose "answers"
occur many times throughout the haystack, so they have no single needle depth.

Runs in place but writes atomically (temp file + rename).
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

TOKENIZER_PATH = "data/models/Qwen3-8B"
NIAH_TASKS = ("niah_single_2", "niah_multikey_1", "niah_multivalue", "niah_multiquery")


def _tokenize_all(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False).input_ids


def _find_all(ids: list[int], needle: list[int]) -> list[int]:
    n = len(needle)
    if n == 0:
        return []
    return [i for i in range(len(ids) - n + 1) if ids[i : i + n] == needle]


def process_file(path_str: str) -> tuple[str, int, int]:
    from transformers import AutoTokenizer

    path = Path(path_str)
    tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    rows = [json.loads(line) for line in path.open()]
    ambiguous = 0
    for row in rows:
        ids = _tokenize_all(tok, row["input"])
        positions = []
        for answer in row["outputs"]:
            needle = tok(str(answer), add_special_tokens=False).input_ids
            hits = _find_all(ids, needle)
            if len(hits) != 1:
                ambiguous += 1
                positions.append(None)
            else:
                positions.append(hits[0])
        row["needle_positions"] = positions
        row["needle_depths"] = (
            [round(p / len(ids), 6) if p is not None else None for p in positions]
        )

    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    return str(path), len(rows), ambiguous


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/raw/ruler"))
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    files = [
        str(p)
        for p in sorted(args.root.glob("L*/**/test.jsonl"))
        if p.parent.name in NIAH_TASKS
    ]
    print(f"processing {len(files)} NIAH files with {args.workers} workers")
    total_rows = total_amb = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for path, n_rows, amb in pool.map(process_file, files):
            total_rows += n_rows
            total_amb += amb
            print(f"  {path}: {n_rows} rows, ambiguous={amb}")
    print(f"done: {total_rows} rows, ambiguous answers={total_amb}")


if __name__ == "__main__":
    main()
