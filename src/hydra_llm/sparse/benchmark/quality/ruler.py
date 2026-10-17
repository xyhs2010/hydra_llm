"""RULER（维度 B/C/D：检索 / 聚合 / 多跳）。

NIAH single/multi、common/frequent words、multi-hop。生成式：prefill 长上下文 →
自回归解码答案 → 与 target 比对。聚合题优先，最能拉开算法差距。
"""
