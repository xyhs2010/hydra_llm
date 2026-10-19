"""dense prefill 的 KV cache —— 每个样本 prefill 只做一次，所有稀疏算法复用同一份 KV。

阶段一（质量）这里只存 dense fp16 KV，稀疏靠 decode 时的 mask 表达，不在这里裁剪；
阶段二（性能）换成 host + q4/q8 后端，接口不变，算法再从这份 dense KV 里物理选块。
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass
class KVCache:
    """一个样本的 dense prefill KV，按 layer 索引。"""

    layers: dict[int, tuple[Tensor, Tensor]]  # layer -> (k, v)，形状 [1, n_kv_heads, seq_len, head_dim]


class KVCachePool:
    """管理多份 KVCache 的容器（内存版）。

    可选磁盘层：dense prefill 昂贵且确定，可按 (model_id, sample_id, dtype) 落盘，
    下次跑（换算法 / 加 sweep 轴）直接加载跳过 prefill。
    """

    def __init__(self) -> None:
        self._items: dict[str, KVCache] = {}

    def put(self, sample_id: str, kv: KVCache) -> None:
        self._items[sample_id] = kv

    def get(self, sample_id: str) -> KVCache:
        try:
            return self._items[sample_id]
        except KeyError:
            raise KeyError(f"no KV cached for sample {sample_id!r}") from None

    def has(self, sample_id: str) -> bool:
        return sample_id in self._items

    def drop(self, sample_id: str) -> None:
        """释放该样本占用的内存，腾预算给下一个样本。"""
        self._items.pop(sample_id, None)

    def __len__(self) -> int:
        return len(self._items)
