# LLM 推理入门资料索引

原则：优先读官方文档、官方仓库和论文。二手博客只作为辅助理解，不作为唯一依据。

## 模型结构

- Transformer 原始论文：Attention Is All You Need  
  https://arxiv.org/abs/1706.03762
- DeepSeek-V2：MLA 与 DeepSeekMoE  
  https://arxiv.org/abs/2405.04434
- DeepSeek-V3.2：DSA  
  https://arxiv.org/abs/2512.02556

## 推理框架

- `llama.cpp` 官方仓库  
  https://github.com/ggml-org/llama.cpp
- `llama.cpp` 量化工具说明  
  https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md
- `vLLM` 官方文档  
  https://docs.vllm.ai/
- PagedAttention 论文  
  https://arxiv.org/abs/2309.06180
- `vLLM` Automatic Prefix Caching  
  https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/
- `SGLang` 官方文档  
  https://docs.sglang.io/
- `SGLang` 官方仓库  
  https://github.com/sgl-project/sglang
- RadixAttention 与 SGLang 官方博客  
  https://www.lmsys.org/blog/2024-01-17-sglang/

## 量化

- GPTQ 论文  
  https://arxiv.org/abs/2210.17323
- GPTQ 官方实现  
  https://github.com/IST-DASLab/gptq
- AWQ 论文  
  https://arxiv.org/abs/2306.00978
- AWQ 官方实现  
  https://github.com/mit-han-lab/llm-awq

## 缓存命中

- PagedAttention 论文  
  https://arxiv.org/abs/2309.06180
- `vLLM` Prefix Caching 设计文档  
  https://docs.vllm.ai/en/stable/design/prefix_caching/
- `SGLang` RadixAttention 博客  
  https://www.lmsys.org/blog/2024-01-17-sglang/

## 投机推理

- Speculative Decoding 早期论文  
  https://arxiv.org/abs/2203.16487
- EAGLE 论文  
  https://arxiv.org/abs/2401.15077
- EAGLE-3 论文  
  https://arxiv.org/abs/2503.01840
- EAGLE 官方实现  
  https://github.com/SafeAILab/EAGLE
- DSpark 论文  
  https://arxiv.org/abs/2607.05147
- DSpark in SGLang 官方博客  
  https://www.lmsys.org/blog/2026-07-06-dspark-sglang/

## 建议阅读顺序

1. 先读 `README.md` 的计划，不打开论文。
2. 第一轮只读每篇论文的 abstract、模型图、核心方法图和 conclusion。
3. 第二轮再读方法细节：MLA、PagedAttention、RadixAttention、GPTQ、AWQ、EAGLE-3、DSpark。
4. 第三轮才进入源码：先看启动参数和文档，再看 scheduler、KV cache、quantization、speculative decoding 相关模块。
