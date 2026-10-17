# hydra_llm

面向长上下文（目标 256K）的自研稀疏注意力推理项目。核心思路：KV cache 放 host 内存，用稀疏化降低 host↔GPU 的 PCIe 带宽需求，并对比不同稀疏算法对推理质量与性能的影响。

## 项目目标

- 实现一套**自研的稀疏注意力算法**（思路与 StreamingLLM / H2O / Quest / InfLLM / MInference 均不同，不照搬）。
- 基准流程：**dense prefill 只做一次 → 复用同一份 KV → decode 阶段换不同稀疏算法对比**。
- 评测两个独立维度：**质量**（丢多少精度）和**性能**（tok/s、KV 显存），最终产出「质量 × 稀疏率 × 上下文长度」的 Pareto 前沿。

## 环境与硬件

- 本机：RTX 4050 Laptop，6 GB 显存（6141 MiB），驱动 595.99.02。
- 开发机：AutoDL 租卡，256K dense 基线需 A800/A100 80G（稀疏算法开发可用 24G/48G 的便宜卡）。
- Python 3.12，包管理用 `uv`，torch 走 `pytorch-cu130` index。
- 已安装：`torch==2.13.0+cu130`（仅此一个，transformers/accelerate 等尚未装）。
- 依赖方向：`uv add transformers accelerate torchao`（量化用 torchao 而非 bitsandbytes —— cu130 太新，第三方量化库 kernel 常跟不上）。

## 技术约束（关键数字）

- 目标模型 **Qwen3-8B**：36 层，GQA 8 个 KV head，head_dim 128，KV ≈ 144 KB/token（fp16）。原生 32K，YaRN 外推到 256K。
- 8B 权重：fp16 = 16 GB（放不下），int4 ≈ 4.5 GB（必须量化才能进 6G）。
- 256K 的 KV：fp16 ≈ 36 GB，q8 ≈ 18 GB，q4 ≈ 9 GB。
- RTX 4050 是 PCIe 4.0 x8，实测带宽 ≈ 13 GB/s —— 这是 host-KV 方案的真正瓶颈。
- **256K prefill 时 flash attention 是刚需**：朴素 attention 的 score 矩阵 O(n²) ≈ 270 GB 直接 OOM。用内置 `F.scaled_dot_product_attention`（自动 dispatch 到 flash 内核）或直接装 flash-attn。256K prefill 在 A800 上约几分钟量级。

## 核心设计要点

1. **mask ≠ 真稀疏**：在 SDPA/flash 内核里加 attention mask，算力和显存和 dense 完全一样，只测「质量」够用、测「性能」无用。测速度/显存必须物理丢弃或跳过 KV。
2. 能立刻用 tensor 切片拿到真实收益的算法族（StreamingLLM 的 sink+滑窗、H2O 剪枝、滑窗/top-k、Quest 式 block 选择）优先做；需要自定义 kernel 的（MInference 这类 pattern）后置。
3. host-KV 的 PyTorch 实现两个必踩点：KV 用 **`pin_memory=True`**（否则 CPU→GPU 拷贝慢 2~3 倍）、**`non_blocking=True` + CUDA stream** 让预取和计算重叠。
4. 稀疏化优先选 **query-aware**（Quest 式），静态的（StreamingLLM/H2O）在长文随机检索会崩。
5. 按 **block 粒度**取 KV（16~64 token/page，coalesced 读），保留 attention sink + 最近窗口。
6. KV 量化（q8/q4）与稀疏化是乘法关系，可叠加。

## 推理质量评测方法

两个根本问题：(a) 还像不像 dense 一样预测下一个 token？(b) 被丢掉的 KV 有没有破坏长上下文利用能力？

### 六个维度

| 维度 | 测什么 | 基准 |
|---|---|---|
| A. 语言建模 / dense 逼近 | PPL | PG19、LDC |
| B. 检索 / 定位 | 找到远处事实 | RULER single/multi NIAH |
| C. 聚合 / 全局依赖 | 同时 attend 大量位置 | RULER common/frequent words、计数 |
| D. 多跳 / 综合推理 | 链式关联多个事实 | RULER multi-hop、LongBench multi-doc QA |
| E. 真实任务泛化 | 非合成任务 | LongBench、∞Bench |
| F. 生成质量 | 端到端输出 | ROUGE / LLM-as-judge（下游产品时才加）|

### 三个 sweep 轴（「多场景」的关键）

1. **上下文长度**：16K → 32K → 64K → 128K → 256K，出曲线不是单点。
2. **稀疏率**：保留 10% / 20% / 50% / 100% KV。
3. **关键信息位置**：needle 放开头/中间/结尾 —— 稀疏算法普遍保 sink+recent，**中间是重灾区**，是最尖锐的测试。

### 方法论要点

- 永远报 **dense 基线的 gap**，不只报绝对值（Qwen3-8B 在 256K 是 YaRN 外推，dense 本身可能已衰减）。
- 样本量要够、报 mean±std，NIAH 单点噪声大，别拿一次命中说事。
- **RULER 的聚合题优先做**——比单 needle 更能拉开不同稀疏算法的差距，是最有说服力的证据。
- 起步组合：PPL（PG19）+ RULER（全自动、可到 256K），跑通后再补 LongBench 做真实任务背书。

### 两阶段推进

1. 阶段一（质量，快）：transformers 里 monkeypatch attention 加 mask / KV 裁剪，先画质量曲线，淘汰掉明显掉精度的算法。
2. 阶段二（速度）：只对幸存的 2~3 个算法实现真切片 + flash 解码，测 tok/s 和显存。**不要给最终伤质量的算法白写 kernel。**

## 当前状态

- 仓库基本是空脚手架（仅 `src/hydra_llm/__init__.py`），尚未有推理或评测代码。
- 下一步：搭 PPL（PG19）+ RULER 的评测 harness（输入 = 模型 + 稀疏算法 + 长度 + 稀疏率，输出 = 质量曲线），先把 dense 基线跑通。

## 待定

- AutoDL 具体租哪档卡（基线倾向 80G：A800/A100）。
- 评测 harness 的框架选型（transformers 手写循环 vs 参考 Quest/H2O 等现成代码）。
