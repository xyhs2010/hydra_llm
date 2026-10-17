"""PPL（维度 A：语言建模 / dense 逼近）。

流程：dense prefill 出整份 KV 进池 → 在稀疏策略下因果重放、收集逐位置 NLL →
ppl_metric 算 exp(mean NLL)。基准：PG19 / LDC。
"""
