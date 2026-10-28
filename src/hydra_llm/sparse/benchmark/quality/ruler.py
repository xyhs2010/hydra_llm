"""RULER（维度 B/C/D：检索 / 聚合 / 多跳）。

流程与 PPL 不同：这里是生成式评测，prompt 只 prefill 一次，得到 dense KV；
之后贪心解码（每步只看最后一个 token 的 logits）。这个 prefill KV 可以在
不同稀疏算法之间复用 —— 正是项目基准流程要测的东西。

`logits_to_keep=1` 让 prefill 只返回最后一个位置的 logits，避免
seq_len × vocab 的显存爆炸。
"""

from __future__ import annotations

import torch

from .data import Sample
from .metrics import DecodeResult, postprocess_pred


@torch.no_grad()
def evaluate_ruler(
    model,
    tokenizer,
    sample: Sample,
    *,
    max_new_tokens: int | None = None,
    device: torch.device | str | None = None,
) -> DecodeResult:
    """Dense prefill + greedy decode; returns the decoded continuation."""
    if device is None:
        device = next(model.parameters()).device
    if max_new_tokens is None:
        max_new_tokens = int(sample.metadata.get("tokens_to_generate", 128))

    ids = torch.tensor([sample.context_tokens], dtype=torch.long, device=device)
    out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
    past = out.past_key_values
    logits = out.logits[:, -1, :]

    eos = tokenizer.eos_token_id
    generated: list[int] = []
    for _ in range(max_new_tokens):
        next_id = int(logits.argmax(dim=-1).item())
        generated.append(next_id)
        if eos is not None and next_id == eos:
            break
        out = model(
            input_ids=torch.tensor([[next_id]], dtype=torch.long, device=device),
            past_key_values=past,
            use_cache=True,
            logits_to_keep=1,
        )
        past = out.past_key_values
        logits = out.logits[:, -1, :]

    text = postprocess_pred(tokenizer.decode(generated, skip_special_tokens=True))
    return DecodeResult(
        sample_id=sample.sample_id,
        generated_ids=generated,
        generated_text=text,
    )
