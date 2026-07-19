# LLM 推理入门学习计划

生成日期：2026-07-20

这份计划面向“尽快入门 LLM 推理与端侧部署”的目标。重点不是先把所有数学公式推完，而是先建立能复述、能画图、能跑通最小实验的系统理解：模型结构如何产生 KV cache，推理框架如何调度和管理显存，量化为什么能把模型放到端侧，缓存命中和投机推理为什么能降低延迟。

师哥原建议中的几个写法这里做了规范：

- `lama.cpp` 统一记为 `llama.cpp`。
- `llm` 按推理框架语境理解为 `vLLM`。
- `dspark` 记为 `DSpark`，即 DeepSeek/SGLang 近期的投机推理方案。
- `DSA` 按 DeepSeek Sparse Attention 理解。

## 0. 入门目标

完成本计划后，应能做到：

1. 用自己的话讲清 Transformer 从 token 到 logits 的主流程。
2. 解释 MHA、MQA、GQA、MLA、DSA 的差异，以及它们对 KV cache、显存和长上下文的影响。
3. 跑通 `llama.cpp` 的 GGUF 模型加载、量化模型推理和 OpenAI-compatible server。
4. 说清 `llama.cpp`、`vLLM`、`SGLang` 分别适合什么场景。
5. 解释 GPTQ、AWQ 的核心思路，并知道 GGUF/INT4/FP8/KV cache quantization 在部署中的位置。
6. 说清 prefix caching、RadixAttention、PagedAttention、cache-aware scheduling 的核心区别。
7. 用 draft-verify 的视角讲清投机推理，并能比较 vanilla speculative decoding、EAGLE-3、DSpark。
8. 做出一份 10 页以内的入门汇报或 3000 字以内的技术总结。

## 1. 总体节奏

推荐周期：6 周。

如果每周只有 8-10 小时，就按 6 周走；如果每周能投入 18-25 小时，可以压缩到 4 周。每天不要只看文章，至少保留 30% 时间做输出：画流程图、写小结、跑命令、记录现象。

每个主题都按这个闭环执行：

```text
读一手资料 -> 画一张流程图 -> 写一页自己的解释 -> 跑一个最小实验 -> 复盘“慢在哪里/省在哪里”
```

## 2. 第 1 周：模型结构主线

目标：看懂 LLM 推理为什么会被 KV cache、显存带宽和注意力结构支配。

必学内容：

- Transformer decoder-only 主流程：tokenization、embedding、positional encoding/RoPE、self-attention、MLP、residual、layer norm、logits。
- Prefill 与 decode：prefill 一次处理整段 prompt，decode 一次通常生成一个 token。
- KV cache：为什么能避免重复计算历史 token 的 K/V；为什么上下文越长显存越紧。
- MHA、MQA、GQA：先理解“多少个 query head 共享多少个 KV head”。
- MLA：把 K/V 压到 latent 表示，核心价值是显著降低 KV cache。
- DSA：用稀疏注意力/选择机制减少长上下文 attention 计算。

建议产出：

- `notes/01_transformer_flow.md`：一页解释 Transformer 推理流程。
- `notes/02_attention_variants.md`：用表格比较 MHA/MQA/GQA/MLA/DSA。
- 一张手绘或 Mermaid 图：`输入 token -> prefill -> KV cache -> decode loop -> logits`。

验收问题：

- 为什么 decode 阶段通常比 prefill 更受显存带宽影响？
- KV cache 里面存的是什么，按层、按 head、按 token 大概怎么增长？
- MLA 为什么不是简单地“少几个 KV head”，而是压缩到 latent 空间？
- DSA 主要解决长上下文里的哪类成本？

## 3. 第 2 周：推理框架主线

目标：知道端侧和服务端为什么用不同框架，以及每个框架的关键工程设计。

必学内容：

- `llama.cpp`
  - GGUF 文件格式、模型转换、量化权重加载。
  - `llama-cli`、`llama-server`、OpenAI-compatible API。
  - ggml/gguf 的定位：跨平台、轻依赖、CPU/GPU/Metal/Vulkan 等后端。
  - 端侧重点：小模型、低比特量化、CPU+GPU 混合、内存占用。
- `vLLM`
  - PagedAttention：把 KV cache 分块管理，降低碎片和浪费。
  - continuous batching：请求动态进出 batch，提升吞吐。
  - prefix caching、chunked prefill、speculative decoding 支持。
  - 服务端重点：高吞吐、多请求调度、显存管理。
- `SGLang`
  - RadixAttention：用 radix tree 管理和复用共享前缀的 KV cache。
  - structured generation、OpenAI API、多 GPU、投机推理生态。
  - 适合 agent、RAG、多轮对话、共享 system prompt/工具定义的场景。

建议实操：

```powershell
# 先只学习命令形态，具体模型按机器性能选择小模型 GGUF。
llama-cli -m model.gguf -p "Explain KV cache in one paragraph."
llama-server -m model.gguf --port 8080
```

如果本机没有合适 GPU，`vLLM` 和 `SGLang` 本周先以阅读架构图、启动参数和官方 quickstart 为主，不强行跑大模型。

建议产出：

- `notes/03_framework_compare.md`：比较 `llama.cpp`、`vLLM`、`SGLang`。
- `notes/04_llama_cpp_runbook.md`：记录模型下载、GGUF、量化、启动 server 的流程。

验收问题：

- 为什么端侧常用 `llama.cpp`，而高并发服务更常看 `vLLM`/`SGLang`？
- PagedAttention 和 RadixAttention 都和 KV cache 有关，但解决的问题有什么不同？
- continuous batching 为什么能提升吞吐，但不一定降低单请求延迟？

## 4. 第 3 周：量化必学

目标：知道量化怎么省显存、何时能加速、为什么 GPTQ/AWQ 是必学项。

必学内容：

- 基础概念：weight-only quantization、activation quantization、KV cache quantization、calibration、group size、scale、zero point、per-channel/per-token。
- GPTQ：一类 post-training quantization，用近似二阶信息做逐层/逐块权重量化，目标是低比特下保持精度。
- AWQ：根据 activation 分布找重要权重通道，通过保护/缩放 salient channels 来降低量化误差。
- GGUF 量化：`Q4_K_M`、`Q5_K_M`、`Q8_0` 等格式先知道含义和取舍，不急着背全。
- 量化评估：显存占用、tokens/s、TTFT、困惑度/任务准确率、输出质量主观对比。

建议实操：

- 找一个小模型的 FP16/BF16 与 Q4/Q5/Q8 GGUF，对比内存和速度。
- 阅读 `llama.cpp/tools/quantize` 文档，弄清“先转 GGUF，再量化”的流程。
- 做一张表：同一 prompt 下不同量化格式的速度、内存、输出质量。

建议产出：

- `notes/05_quantization_gptq_awq.md`：解释 GPTQ 和 AWQ 的差异。
- `experiments/quantization_benchmark_template.md`：保存 benchmark 表格模板。

验收问题：

- 量化为什么能省显存？为什么有时也能加速？
- GPTQ 和 AWQ 都是 PTQ，但校准思路有什么不同？
- weight quantization 和 KV cache quantization 分别影响什么？
- 端侧部署为什么常用 4-bit/5-bit，而不是一味追求 2-bit？

## 5. 第 4 周：其他轻量化方案

目标：建立概念地图，知道这些方案什么时候值得深入，而不是一开始平均用力。

建议了解：

- 低秩矩阵分解：用较低 rank 近似大矩阵，减少参数或计算；和 LoRA/SVD 的直觉相通。
- 蒸馏：大模型当 teacher，小模型学 logits、偏好、推理轨迹或任务输出。
- 剪枝：删除权重、通道、head、层或专家；区分 unstructured 与 structured pruning。
- MoE 与稀疏激活：不一定是轻量化，但能让每个 token 只激活部分参数。
- 编译与 kernel 优化：FlashAttention、CUDA graph、Triton kernel fusion 先知道术语和作用。

本周不要深陷训练细节。入门优先级是：

```text
量化 > KV/cache 优化 > 投机推理 > 蒸馏/剪枝/低秩分解
```

建议产出：

- `notes/06_model_compression_map.md`：一页概念地图。
- `notes/07_when_to_use_which_optimization.md`：按端侧、单机服务、高并发服务分别列方案。

验收问题：

- 低秩分解、蒸馏、剪枝分别改变模型的什么？
- 哪些方案主要减少显存，哪些主要减少计算，哪些主要改善调度？
- 为什么端侧最先落地的常常是量化，而不是剪枝？

## 6. 第 5 周：缓存命中与上下文复用

目标：理解“缓存命中”不只是普通工程缓存，而是复用已经算过的 KV cache。

必学内容：

- KV cache 生命周期：创建、增长、复用、驱逐。
- Prefix caching：相同前缀直接复用 KV cache，跳过共享 prompt 的 prefill。
- Block hashing：按 token block 建索引，判断能否复用。
- RadixAttention：把共享前缀组织成 radix tree，适合多轮对话、agent、RAG、few-shot prompt。
- Cache-aware scheduling：调度请求时考虑缓存命中率，减少重复 prefill。
- 影响命中的工程细节：system prompt 是否一致、工具定义是否稳定、RAG chunk 顺序、空格/模板/tokenizer 是否完全一致。

建议产出：

- `notes/08_kv_cache_and_prefix_cache.md`：解释 KV cache 与 prefix cache。
- `notes/09_cache_hit_playbook.md`：写一个“如何提高缓存命中”的 checklist。

验收问题：

- 什么情况下 prefix caching 完全命中？什么情况下只命中一部分？
- 为什么同样内容、不同模板或不同 tokenization 可能导致不能复用？
- PagedAttention、Automatic Prefix Caching、RadixAttention 分别在 cache 管理中处于什么层次？
- RAG/agent 场景为什么特别适合做 prefix cache？

## 7. 第 6 周：投机推理

目标：抓住 draft-verify 主线，了解几个主流算法，不需要一开始复现论文。

必学内容：

- Vanilla speculative decoding：小 draft model 先猜多个 token，大 target model 一次验证，接受前缀，拒绝处回退。
- 关键指标：draft latency、acceptance length、acceptance rate、verify cost、batch size、吞吐/延迟权衡。
- Medusa/MTP 类思路：让模型自己长出多个预测头或多 token 预测能力。
- EAGLE-3：从 feature-level EAGLE 演进到更直接的 token prediction 与多层特征融合。
- DSpark：半自回归 block drafter + confidence-scheduled variable-length verification，重点解决高并发下验证浪费。
- 什么时候不划算：draft 太慢、接受率太低、batch 太大导致 verify token 成本上涨、业务输出分布太难预测。

建议实操：

- 阅读 `llama.cpp` server speculative decoding 参数形态，理解 target model 与 draft model 的关系。
- 看 EAGLE-3 和 DSpark 的摘要、图和实验指标，不先啃全部证明。
- 用一页图画出：`draft K tokens -> target verify -> accept prefix -> resample/reject -> next step`。

建议产出：

- `notes/10_speculative_decoding.md`：投机推理算法总览。
- `notes/11_eagle3_dspark_compare.md`：比较 EAGLE-3 与 DSpark。
- `final_report.md`：入门总结，控制在 3000 字以内。

验收问题：

- 投机推理为什么理论上可以“无损”保持目标模型分布？
- speedup 为什么不只取决于 acceptance rate？
- EAGLE-3 和 DSpark 分别在 draft 质量、并行生成、验证调度上解决什么问题？
- 高并发服务里，为什么 DSpark 会强调 variable-length verification？

## 8. 四周压缩版

如果想最快完成入门，按下面路线压缩：

| 周次 | 主线 | 必须产出 |
|---|---|---|
| Week 1 | Transformer、KV cache、MHA/MQA/GQA/MLA/DSA | 2 页结构笔记 + 1 张推理流程图 |
| Week 2 | `llama.cpp`、`vLLM`、`SGLang` | 框架对比表 + `llama.cpp` 最小运行记录 |
| Week 3 | 量化、GPTQ、AWQ、轻量化概念地图 | 量化对比表 + GPTQ/AWQ 一页总结 |
| Week 4 | 缓存命中、RadixAttention、投机推理、EAGLE-3、DSpark | 最终 10 页汇报或 3000 字总结 |

压缩版的原则：先会讲、会画、会跑最小 demo；源码和论文细节放到第二轮。

## 9. 每周固定工作流

每周至少做 5 件事：

1. 读 2 篇一手资料或官方文档。
2. 写 1 篇 800-1200 字中文小结。
3. 画 1 张流程图或对比表。
4. 跑 1 个命令或阅读 1 个关键启动参数。
5. 复盘 3 个问题：瓶颈在哪里、省了什么、牺牲了什么。

建议记录模板：

```markdown
# 本周主题

## 我现在能讲清楚的

## 还模糊的概念

## 跑过的命令或读过的源码入口

## 一句话总结

## 下周要验证的问题
```

## 10. 最终入门汇报大纲

建议做成 8-10 页 PPT 或 Markdown：

1. LLM 推理总览：prefill、decode、KV cache。
2. Transformer 与注意力变体：MHA/MQA/GQA/MLA/DSA。
3. 推理框架对比：`llama.cpp`、`vLLM`、`SGLang`。
4. 量化：GPTQ、AWQ、GGUF 量化格式。
5. 其他轻量化：低秩、蒸馏、剪枝。
6. 缓存命中：prefix caching、RadixAttention、PagedAttention。
7. 投机推理：draft-verify、EAGLE-3、DSpark。
8. 自己的理解：端侧部署和高并发服务的优化路径有什么不同。
9. 后续深入路线：源码、benchmark、论文复现。

## 11. 资料入口

详细资料索引放在 [resources.md](resources.md)。

执行打卡清单放在 [weekly_checklist.md](weekly_checklist.md)。
