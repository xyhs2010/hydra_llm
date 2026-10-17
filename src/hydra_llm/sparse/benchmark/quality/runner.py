"""评测循环：输入 = 模型 + 稀疏算法 + 长度 + 稀疏率 → 输出 = 质量分数。

1. 遍历 Sample：KV 池缺该样本则 dense prefill 并 put。
2. 从池取 dense KV。
3. 在 algo 下 decode（PPL 因果重放 / 生成自回归）。
4. metric(sample, result) → 分数。
5. 聚合 mean±std，按 benchmark / 长度 / 稀疏率 / needle 位置 出曲线。

dense 基线 = algo=None，先跑通它。
"""
