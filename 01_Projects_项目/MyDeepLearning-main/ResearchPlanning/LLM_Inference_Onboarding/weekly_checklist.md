# LLM 推理入门执行清单

## Week 1：模型结构

- [ ] 读完 Attention Is All You Need 摘要、模型结构图和 decoder 相关部分。
- [ ] 写清 tokenization、embedding、RoPE、attention、MLP、logits 的顺序。
- [ ] 画出 prefill 与 decode 的差异。
- [ ] 写清 KV cache 存什么、为什么能省重复计算。
- [ ] 比较 MHA、MQA、GQA、MLA、DSA。
- [ ] 输出 `notes/01_transformer_flow.md`。
- [ ] 输出 `notes/02_attention_variants.md`。

## Week 2：推理框架

- [ ] 看 `llama.cpp` README，理解 GGUF、后端、量化、server。
- [ ] 跑一次 `llama-cli` 或至少写出完整命令。
- [ ] 跑一次 `llama-server` 或至少写出完整命令。
- [ ] 看 `vLLM` quickstart、PagedAttention、Automatic Prefix Caching。
- [ ] 看 `SGLang` quickstart、RadixAttention、structured generation。
- [ ] 输出 `notes/03_framework_compare.md`。
- [ ] 输出 `notes/04_llama_cpp_runbook.md`。

## Week 3：量化

- [ ] 读 GPTQ 摘要，写出“一次性 PTQ + 近似二阶信息”的直觉。
- [ ] 读 AWQ 摘要，写出“activation-aware + salient channel”的直觉。
- [ ] 看 `llama.cpp/tools/quantize` 文档，理解 GGUF 量化流程。
- [ ] 比较 Q4、Q5、Q8 的内存、速度、质量取舍。
- [ ] 记录 weight quantization、activation quantization、KV cache quantization 的区别。
- [ ] 输出 `notes/05_quantization_gptq_awq.md`。
- [ ] 输出 `experiments/quantization_benchmark_template.md`。

## Week 4：轻量化扩展

- [ ] 写清低秩矩阵分解减少什么。
- [ ] 写清蒸馏中的 teacher/student、logits、hard label、reasoning trace。
- [ ] 写清剪枝中的 unstructured 与 structured pruning。
- [ ] 了解 MoE/sparse activation 与轻量化的关系。
- [ ] 了解 FlashAttention、CUDA graph、Triton kernel fusion 的作用。
- [ ] 输出 `notes/06_model_compression_map.md`。
- [ ] 输出 `notes/07_when_to_use_which_optimization.md`。

## Week 5：缓存命中

- [ ] 画出 KV cache 从 prefill 到 decode 的生命周期。
- [ ] 写清 prefix caching 的命中条件。
- [ ] 理解 vLLM Automatic Prefix Caching 的 block 复用思路。
- [ ] 理解 SGLang RadixAttention 的 radix tree 复用思路。
- [ ] 写一份提高 cache hit rate 的工程 checklist。
- [ ] 输出 `notes/08_kv_cache_and_prefix_cache.md`。
- [ ] 输出 `notes/09_cache_hit_playbook.md`。

## Week 6：投机推理

- [ ] 写出 vanilla speculative decoding 的 draft-verify 流程。
- [ ] 记录 acceptance rate、accepted length、verify cost 的区别。
- [ ] 了解 Medusa/MTP 类多 token 预测思路。
- [ ] 读 EAGLE-3 摘要和官方 README。
- [ ] 读 DSpark 摘要和 SGLang 集成博客。
- [ ] 输出 `notes/10_speculative_decoding.md`。
- [ ] 输出 `notes/11_eagle3_dspark_compare.md`。
- [ ] 输出 `final_report.md`。

## 最终验收

- [ ] 能 5 分钟讲清 LLM 推理主流程。
- [ ] 能 5 分钟讲清 `llama.cpp` 为什么适合端侧。
- [ ] 能 5 分钟讲清 `vLLM` 与 `SGLang` 的关键优化。
- [ ] 能 5 分钟讲清 GPTQ 与 AWQ 的差异。
- [ ] 能 5 分钟讲清缓存命中和 prefix cache。
- [ ] 能 5 分钟讲清投机推理、EAGLE-3、DSpark。
- [ ] 完成 3000 字以内入门总结或 10 页以内汇报。
