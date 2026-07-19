# LLM 推理入门最终总结

## 1. 我对 LLM 推理主流程的理解

## 2. Transformer 与注意力变体

## 3. 推理框架对比

| 框架 | 核心定位 | 关键优化 | 适合场景 |
|---|---|---|---|
| llama.cpp | 端侧/本地轻量推理 | GGUF、低比特量化、多硬件后端 | 笔记本、手机、边缘设备 |
| vLLM | 高吞吐服务端推理 | PagedAttention、continuous batching、prefix caching | API 服务、高并发 |
| SGLang | 结构化生成与高性能服务 | RadixAttention、prefix reuse、speculative decoding | Agent、RAG、多轮对话 |

## 4. 量化与模型轻量化

## 5. 缓存命中

## 6. 投机推理

## 7. 端侧部署与服务端部署的差异

## 8. 后续深入计划
