"""Chunked prefill attention without materialising a causal mask.

HF's default SDPA path builds an explicit [q_len, kv_len] causal mask whenever
queries have a KV cache prefix (q_offset > 0). At long context this is O(q*kv)
memory and forces SDPA off the flash kernel.

This implementation splits the chunk's attention into two mask-free flash calls:

    past keys   : full attention over the cache prefix   (is_causal=False)
    chunk keys  : causal within the chunk                (is_causal=True)

and merges them with the standard log-sum-exp trick. With no cache prefix the
whole sequence is a single causal flash call.

Registered as attention implementation "hydra". Because it is *not*
registered in ALL_MASK_ATTENTION_FUNCTIONS, transformers skips mask building
entirely (`_preprocess_mask_arguments` early-exits).

Interface matches `sdpa_attention_forward`: query/key/value are [B, H, T, D],
the return value is ([B, T, H, D], None).
"""

from __future__ import annotations

import torch

_FALLBACKS = []


def _flash(q, k, v, causal: bool, scale: float | None):
    """aten flash op; returns (out, lse) with lse in natural log."""
    r = torch.ops.aten._scaled_dot_product_flash_attention.default(
        q, k, v, 0.0, causal, False, scale=scale
    )
    return r[0], r[1]


def _merge(o1, l1, o2, l2):
    m = torch.maximum(l1, l2)
    w1 = torch.exp(l1 - m).unsqueeze(-1)
    w2 = torch.exp(l2 - m).unsqueeze(-1)
    return (o1.float() * w1 + o2.float() * w2) / (w1 + w2)


def flash_attention_forward(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    position_bias=None,
    **kwargs,
):
    if attention_mask is not None or position_bias is not None:
        raise RuntimeError("hydra_flash only supports mask-free causal attention")
    if dropout:
        raise RuntimeError("hydra_flash does not support dropout")
    if kwargs.get("softcap") is not None:
        raise RuntimeError("hydra_flash does not support softcap")

    q_len = query.shape[2]
    kv_len = key.shape[2]
    past = kv_len - q_len
    if past < 0:
        raise RuntimeError(f"kv_len ({kv_len}) < q_len ({q_len})")

    with torch.no_grad():
        if past == 0:
            out, _ = _flash(query, key, value, causal=q_len > 1, scale=scaling)
        else:
            o_past, l_past = _flash(
                query, key[:, :, :past], value[:, :, :past], causal=False, scale=scaling
            )
            o_cur, l_cur = _flash(
                query, key[:, :, past:], value[:, :, past:], causal=q_len > 1, scale=scaling
            )
            out = _merge(o_past, l_past, o_cur, l_cur).to(query.dtype)

    return out.transpose(1, 2).contiguous(), None


def register_attention() -> None:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    if hasattr(ALL_ATTENTION_FUNCTIONS, "register"):
        ALL_ATTENTION_FUNCTIONS.register("hydra", flash_attention_forward)
    else:  # older containers expose the mapping directly
        ALL_ATTENTION_FUNCTIONS._global_mapping["hydra"] = flash_attention_forward
