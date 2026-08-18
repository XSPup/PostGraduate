# HiCache 端侧多级缓存调度：完整讲解档案

## 1. 这项研究到底在做什么

研究对象是大模型推理过程中产生的 KV Cache。

在多轮会话、固定系统提示、端侧 RAG 和 Agent 场景中，多个请求经常共享相同的长前缀。如果前缀对应的 KV 仍在快速内存中，系统可以跳过这部分 Prefill；如果 KV 被逐出，只能重新计算。

端侧的问题是：

- 加速器可直接使用的 KV 容量很小；
- LPDDR 带宽被模型计算和后台搬运共同使用；
- UFS/SSD 容量大，但访问延迟和尾延迟明显；
- 后台读写会增加能耗、温度和闪存写入；
- 因此“缓存命中”不一定比“重新计算”更快。

本研究准备借鉴 SGLang HiCache 的三级 KV 缓存、分页、Radix 前缀和异步状态机，但不照搬 CUDA、RDMA 和分布式后端实现。核心研究问题是：

> 在端侧实时状态下，如何决定一个 KV page 应该读取、等待、取消、重算、写回或逐出？

---

## 2. 五层汇报结构之间的关系

### 研究背景回答：为什么值得做

长前缀复用广泛存在，但 KV 容量随上下文线性增长。端侧无法让所有 KV 常驻，所以重复 Prefill 会增加 TTFT、能耗和温升。

### 研究内容回答：具体研究什么

研究三个问题：

1. HiCache 的哪些机制能迁移到端侧；
2. 一个 KV page 什么时候值得读取；
3. 新产生的 KV 什么时候值得写入持久层。

### 技术路线回答：怎样做

先做 microbenchmark 建模，再复现静态基线，之后实现 EAHS 在线调度，最后在真实设备上评价。

### 研究基础回答：为什么有可能做成

RadixAttention、HiCache Controller、异构内存/存储 offload、KV 量化和端侧运行时都提供了可复现基础。

### 预期成果回答：怎样判断是否完成

交付系统原型、算法、实验资产和论文结论，并用 TTFT、ITL、能耗、温度、写入量和失败场景形成可复现闭环。

---

## 3. 基础概念

### 3.1 Prefill

Prefill 并行处理整个输入序列，为每层注意力模块生成 Key 和 Value。输入越长，Prefill 计算和 KV 写入越多。

### 3.2 Decode

Decode 逐 token 生成。每一步都要读取历史 KV，所以通常对内存带宽敏感。后台搬运 KV 可能直接增加 inter-token latency。

### 3.3 TTFT

TTFT 是从请求到达至生成首 token 的延迟，可以粗略拆成：

$$
\begin{aligned}
T_{\mathrm{TTFT}}
={}&T_{\mathrm{queue}}
+T_{\mathrm{model\_load}}
+T_{\mathrm{KV\_load}} \\
&+T_{\mathrm{prefill}}
+T_{\mathrm{first\_decode}}
\end{aligned}
$$

其中，$T_{\mathrm{queue}}$ 是排队时间，$T_{\mathrm{model\_load}}$ 是模型准备时间，$T_{\mathrm{KV\_load}}$ 是 KV 数据读取时间，$T_{\mathrm{prefill}}$ 是输入计算时间，$T_{\mathrm{first\_decode}}$ 是生成首个 token 的时间。

分层 KV 缓存减少的是重复的 $T_{\mathrm{prefill}}$，同时可能增加 $T_{\mathrm{KV\_load}}$ 和后台搬运干扰。

### 3.4 ITL

ITL 是相邻生成 token 之间的延迟。只优化 TTFT 但使 ITL 增大，会导致“首 token 很快、后续输出卡顿”。端侧调度必须保护 Decode。

### 3.5 Prefix Cache

若新请求与历史请求拥有完全相同的 token 前缀，并且模型、tokenizer、位置编码和 KV 格式兼容，就可以复用该前缀的 KV。

### 3.6 RadixAttention

RadixAttention 使用 RadixTree 管理共享前缀。连续 token 前缀被组织成可匹配、可逐出、可复用的节点。

“连续”非常重要：如果前面某个 page 没有就绪，后面单独命中的 page 不能直接作为完整历史使用。

---

## 4. KV Cache 容量为什么会迅速增长

先定义各个变量：

| 符号 | 含义 |
|---|---|
| L | Transformer 层数 |
| H_kv | KV 注意力头数量 |
| d_h | 每个注意力头的维度 |
| n | 上下文中的 token 数量 |
| b | 每个数值占用的字节数，例如 FP16 为 2 字节 |

KV Cache 的总容量可以近似写成：

```text
KV Cache 总容量（字节）
= 2 × L × H_kv × d_h × n × b
```

最前面的 2 表示需要同时保存 Key 和 Value 两份数据。

单个 token 产生的 KV Cache 大小为：

```text
单个 token 的 KV 大小（字节）
= 2 × L × H_kv × d_h × b
```

以常见的 Llama 3 8B GQA、FP16 KV 配置作为量级示例：

```text
模型层数 L          = 32
KV 头数 H_kv        = 8
每个头的维度 d_h    = 128
每个数值的字节数 b  = 2
```

代入后得到：

```text
单个 token 的 KV 大小
= 2 × 32 × 8 × 128 × 2
= 131072 bytes
= 128 KiB/token
```
因此：

- 8K token 约 1 GiB；
- 32K token 约 4 GiB；
- 128K token 约 16 GiB。

注意：

- 这是量级示例，不是所有模型的固定值；
- GQA 已经减少了 KV 头数；
- MLA、滑动窗口、KV 量化会改变容量；
- 分页对齐、元数据和碎片会增加实际占用。

---

## 5. 前缀命中收益的数学推导

设请求由可复用前缀 \(k\) 个 token 和新增内容 \(s\) 个 token 组成。

无缓存时：

\[
T_{no\ cache}=C(k+s)
\]

命中并加载前缀 KV 时：

\[
T_{hit}=R(k)+C(s)+I(k)
\]

其中：

- \(C(x)\)：Prefill 计算 \(x\) 个 token 的时间；
- \(R(k)\)：前缀 KV 从当前层加载到可计算状态的完成时间；
- \(I(k)\)：加载对前台计算产生的干扰。

收益：

\[
\Delta T=C(k+s)-C(s)-R(k)-I(k)
\]

若 Prefill 对 token 近似线性：

\[
C(x)\approx ax+c
\]

则：

\[
\Delta T\approx ak-R(k)-I(k)
\]

基本 break-even：

\[
C(k)>R(k)+I(k)
\]

这条不等式解释了为什么高命中率不一定带来低延迟。

### 可能命中但变慢的情况

- UFS 带宽低或尾延迟高；
- 后台读取抢占 LPDDR；
- 预取完成得太晚；
- wait-complete 阻塞前台；
- page 太大，加载了大量未使用 token；
- 格式转换和反量化耗时高；
- 热节流使后续计算变慢。

---

## 6. HiCache 已经做了什么

### 6.1 三级结构

- L1：GPU/HBM，速度最快，直接参与计算；
- L2：Host DRAM，容量更大，保存本地可复用 KV；
- L3：外部或分布式存储，例如 Mooncake、3FS、NIXL。

### 6.2 HiRadixTree

HiRadixTree 保存连续前缀以及 L1/L2 位置信息。L3 元数据可以在请求到达时向后端查询。

### 6.3 请求流程

1. L1/L2 本地前缀匹配；
2. 未命中部分查询 L3；
3. 达到条件时 L3→L2 预取；
4. L2→L1 加载；
5. 执行剩余 Prefill 和 Decode；
6. 新 KV 按策略写回 L2/L3。

### 6.4 预取策略

- best_effort：前台可以执行时停止等待；
- wait_complete：全部完成后再执行；
- timeout：完成或超时后停止。

官方文档中的 timeout 近似：

\[
T_{timeout}=
\min
\left(
T_{max},
T_{base}+\alpha\frac{n_{fetch}}{1024}
\right)
\]

该策略主要按 token 数确定等待上限，没有直接估计实时带宽、干扰和能耗。

### 6.5 写策略

- write-through：每次访问都写到下一层；
- write-through-selective：超过访问阈值后才写；
- write-back：上层逐出时写。

端侧必须额外考虑 UFS 小块 I/O、写入量、能耗和温升。

### 6.6 源码层面的可迁移骨架

当前缓存控制器中值得关注：

- 独立 prefetch 和 backup 后台线程；
- 独立读写队列；
- 加载和写入设备流；
- 写操作合并；
- L2→L1 按层搬运，与计算重叠；
- Host Pool 容量保护；
- 停止、取消、备份和回收状态。

这些状态机和调度入口比具体 CUDA/RDMA 代码更值得迁移。

---

## 7. 为什么服务器实现不能照搬

| 组成 | 端侧复用性 | 处理方式 |
|---|---:|---|
| Radix 前缀元数据 | 高 | 保留并增加版本指纹 |
| page 粒度 | 高 | 保留，粒度由设备实测 |
| get/exists/set 接口 | 高 | 映射到 LPDDR/UFS 后端 |
| 预取和写策略状态机 | 高 | 改为在线自适应 |
| prefetch/backup 双队列 | 高 | 增加 phase 门控和抢占 |
| CUDA kernel/pinned memory | 低 | 按目标运行时重写 |
| RDMA/共享 L3 | 低 | 单端删除，端云场景另行研究 |
| PD 解耦/TP 通信 | 低 | 端侧通常没有同样集群结构 |

### 端侧新增约束

1. 统一内存：L1/L2 可能共享 LPDDR；
2. 前台竞争：Decode 本身 memory-bound；
3. 能耗：后台 I/O 和转换会耗电；
4. 热：持续任务可能触发降频；
5. UFS 长尾：带宽和固定开销受系统活动影响；
6. 闪存写入：write-through 可能写放大；
7. 隐私：持久 KV 需要隔离、加密和生命周期清理。

---

## 8. Edge-HiCache 设计

### 8.1 三层数据面

#### L1：加速器可见 KV 池

当前运行时可以直接使用，布局和精度已经准备好。

#### L2：LPDDR 温缓存

容量更大，但可能需要映射、显式复制或格式转换。

#### L3：UFS/SSD 持久前缀

保存跨请求、跨会话的高价值前缀，但延迟和写入代价最高。

### 8.2 EdgeRadix

保存：

- token 前缀；
- page 位置；
- 访问次数和热度；
- 最近访问时间；
- 模型和格式版本；
- 读写状态；
- deadline 和优先级。

### 8.3 cache key

建议至少包含：

\[
key=H(
model\_id,\ model\_revision,\ tokenizer\_hash,\
rope\_config,\ kv\_dtype,\ page\_layout,\ token\_ids
)
\]

否则模型更新、tokenizer 或位置编码变化后，旧 KV 可能造成静默错误。

### 8.4 最小运行时接口

- prefix_match；
- page_load；
- page_store；
- page_cancel；
- phase；
- timing/energy observation。

最小接口可以降低对具体 Android、Jetson 或 Ascend 运行时的耦合。

---

## 9. EAHS 调度算法

EAHS：Edge-Adaptive Hierarchical Scheduler。

### 9.1 page 状态变量

对候选 page \(i\) 定义：

- \(k_i\)：token 数；
- \(b_i\)：字节数；
- \(C_i\)：可避免的重算时间；
- \(R_i\)：加载完成时间；
- \(W_i\)：写入时间或成本；
- \(E_i^r,E_i^w\)：读写能量；
- \(I_i^r,I_i^w\)：对前台的干扰；
- \(p_i^r\)：本次预测性预取实际被使用的概率；
- \(p_i^H\)：未来时间窗内复用概率；
- \(d_i\)：读 deadline；
- \(slack_i=d_i-now-R_i\)；
- \(Wear_i\)：闪存写入机会成本。

### 9.2 全局目标

\[
\min J=
p95(TTFT)
+\alpha E_{request}
+\beta FlashWrite
+\gamma SLOmiss
+\eta p95(ITL)
\]

不同量纲不能直接硬相加。实际实现中应按基线或预算归一化。

也可以使用约束式：

\[
\min p95(TTFT)
\]

满足：

\[
\begin{aligned}
p95(ITL)&\le ITL_{base}(1+\epsilon)\\
E_{request}&\le E_{budget}\\
FlashWrite&\le W_{budget}\\
Mem(t)&\le C_{available}(t)
\end{aligned}
\]

### 9.3 读收益

\[
U_r(i)=p_i^rC_i-R_i-\lambda_EE_i^r-\lambda_II_i^r
\]

执行条件：

\[
U_r(i)>0
\quad\land\quad
now+R_i\le d_i
\]

需求加载时 \(p_i^r=1\)。预测性预取时 \(p_i^r<1\)。

### 9.4 写收益

\[
U_w(i)=
p_i^H(C_i-R_i)
-W_i
-\lambda_FWear_i
-\lambda_M\frac{b_i}{C_3}
-\lambda_II_i^w
\]

只有 \(U_w(i)>0\) 的 page 才进入写队列。

### 9.5 排序

读候选可以先按 deadline 分桶，再按价值密度排序：

\[
Priority_r(i)=
\frac{U_r(i)}{R_i+\varepsilon}
+\frac{\kappa}{slack_i+\varepsilon}
\]

第一项偏向单位时间收益高的 page，第二项偏向即将到期的 page。

### 9.6 连续前缀约束

- 从最长连续命中边界向后扩展；
- 前序 page 未就绪时，后序 page 不能单独消费；
- 后续 page 累计收益转负时停止；
- deadline 到达时只采用连续就绪部分。

### 9.7 phase 门控

| phase | 读 | 写 |
|---|---|---|
| Decode | 只保留即将到期的高价值读 | 暂停或抢占低优先级写 |
| Prefill | slowdown 小于阈值时限速重叠 | 仅高价值写 |
| Queue slack | 优先最早到期读 | 少量写 |
| Idle | 补齐温缓存 | 合并并排空写队列 |

### 9.8 在线流程

收到请求：

1. EdgeRadix 查找最长兼容连续前缀；
2. 构造缺失 page；
3. 预测 \(C,R,I,E,p,d\)；
4. 计算 \(U_r\)；
5. 收益非正或错过 deadline 时停止扩展；
6. 正收益候选进入读队列；
7. 采用连续完成部分，其余重算。

产生新 KV：

1. 估计未来复用；
2. 计算 \(U_w\)；
3. 只有正收益 page 入写队列；
4. 相邻 page 合并写；
5. Decode 到来时抢占低价值写。

---

## 10. break-even 示例

假设：

- 64 token/page；
- 128 KiB/token；
- page 大小 8 MiB；
- L3 有效带宽 1.5 GB/s；
- 固定开销 0.3 ms。

\[
R\approx
\frac{8\ \mathrm{MiB}}{1.5\ \mathrm{GB/s}}
+0.3\ \mathrm{ms}
\approx5.9\ \mathrm{ms}
\]

若避免重算时间 \(C=12\) ms：

\[
U\approx12-5.9=6.1\ \mathrm{ms}>0
\]

值得读。

若共享总线拥塞使完成时间变为 16 ms：

\[
U\approx12-16=-4\ \mathrm{ms}<0
\]

应该取消并重算。

这组数字只是解释算法，不是实验结果。

---

## 11. 研究基础与相关工作定位

### SGLang/RadixAttention

提供连续前缀匹配和运行时调度基础。

### HiCache

提供三级 KV、异步读写、预取策略和写策略。

### Mooncake

主要关注服务器集群中跨实例 KV 复用和分离式架构。

### NEO/FlexInfer

关注 CPU offload、异构资源和执行路径选择。

### KIVI

关注低比特 KV 量化，降低字节数。

### SparKV

关注端侧从云端加载 KV 与本地重算的动态选择。

### Bidaw

关注 Host-SSD 分层及计算—存储感知。

### 本研究的候选边界

不是“首次多级 KV”，而是：

- 单端本地持久前缀；
- 统一内存前台干扰；
- deadline/value 预取；
- phase-aware 选择性写回；
- 读写联合调度；
- 能耗、热和闪存写入预算。

该边界必须通过继续检索确认。

---

## 12. 实验设计

### 12.1 阶段 A：收益上限

测量：

- \(C(k)\)：不同前缀长度的 Prefill 时间；
- \(R(b)\)：不同 page size 的 L2/L3 加载时间；
- \(I(b,phase)\)：后台 I/O 对 Prefill/Decode 的干扰；
- \(E(b,phase)\)：能耗；
- UFS p50/p95/p99；
- 温度、频率和热节流；
- 量化/格式转换成本。

第一张关键结果应是 break-even 区域图。

### 12.2 阶段 B：原型

实现：

- EdgeRadix；
- L1/L2/L3 后端；
- 读写双队列；
- best_effort、wait_complete、timeout；
- write-through、selective、write-back；
- EAHS；
- 取消、部分采用和回收。

### 12.3 阶段 C：真实设备

工作负载：

- 固定 system prompt；
- 多轮会话；
- 端侧 RAG；
- Agent 工具说明；
- 热点前缀 Zipf；
- 冷请求；
- 不同前缀长度、到达率和并发度。

### 12.4 基线

1. 无前缀缓存；
2. 仅 L1 Radix；
3. 固定阈值 best-effort；
4. wait_complete；
5. timeout；
6. 三种写策略；
7. EAHS；
8. EAHS 去掉 deadline、value、phase、wear 的消融；
9. 平台允许时加入量化或动态 offload 方法。

### 12.5 指标

- p50/p95/p99 TTFT；
- p50/p95 ITL；
- 吞吐；
- 层级命中率；
- 有效加载字节；
- 无效预取字节；
- 取消和超时次数；
- 每请求能量；
- 温度和频率；
- UFS 写入字节；
- 缓存占用和逐出；
- \(C,R,I,E\) 预测误差。

### 12.6 暂定门槛

相对最优静态基线：

- p95 TTFT 降低至少 20%；
- p95 ITL 回退不超过 5%；
- 能耗不增加；
- 相对 write-through，持久层写入降低至少 30%。

这些是目标，不是已有结果。

---

## 13. 风险与止损条件

### 风险 1：真实端侧加载总是慢于重算

应对：转向 KV 量化或加载—重算选择。

### 风险 2：运行时不暴露 KV page 或异步接口

应对：选择更开放的平台，或先在 Jetson/桌面 GPU 验证控制器。

### 风险 3：统一内存层级不明显

应对：把 L1/L2 定义为运行时可见性和格式层次，不强行声称物理分层。

### 风险 4：复用率太低

应对：系统不写冷 KV，并报告冷 workload 无收益。

### 风险 5：与新文献高度重合

应对：缩小到闪存写入预算、连续前缀部分采用、真实统一内存干扰，或改成对现有方法的系统评估与改进。

### 风险 6：调度器本身开销过大

应对：第一版使用规则、滑动统计和轻量预测，不使用复杂神经网络。

---

## 14. 两天内的学习优先级

### 第一优先级：必须能解释

1. Prefill、Decode、TTFT、ITL；
2. KV 容量公式；
3. 前缀命中收益 \(C-R-I\)；
4. HiCache L1/L2/L3；
5. best_effort、wait_complete、timeout；
6. write-through、selective、write-back；
7. EAHS 读收益和写收益；
8. phase 门控；
9. 三阶段实验；
10. 为什么创新点不是“多级缓存”。

### 第二优先级：被问到时解释

1. RadixTree 和连续前缀；
2. cache key 与兼容性；
3. page size 权衡；
4. 部分采用和取消；
5. 能耗、热、闪存写入的评价方法；
6. Mooncake、SparKV、Bidaw、KIVI 的边界。

### 暂时不需要深挖

- HiCache 所有 CUDA kernel 细节；
- RDMA 后端实现；
- 每篇相关论文的全部实验数字；
- 复杂概率模型；
- 在没有目标硬件前编写完整端侧实现。

---

## 15. 参考资料

### 官方资料

- [HiCache Design](https://docs.sglang.io/docs/advanced_features/hicache_design)
- [HiCache Best Practices](https://docs.sglang.io/docs/advanced_features/hicache_best_practices)
- [SGLang cache_controller.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/cache_controller.py)
- [SGLang / RadixAttention](https://arxiv.org/abs/2312.07104)

### 相关工作

- [Mooncake, FAST 2025](https://www.usenix.org/conference/fast25/presentation/qin)
- [SparKV, 2026](https://arxiv.org/abs/2604.21231)
- [Bidaw, FAST 2026](https://www.usenix.org/conference/fast26/presentation/hu-shipeng)
- [KIVI](https://arxiv.org/abs/2402.02750)
- [NEO, MLSys 2025](https://proceedings.mlsys.org/paper_files/paper/2025/hash/66a026c0d17040889b50f0dfa650e5e0-Abstract-Conference.html)
- [FlexInfer, MLSys 2025](https://proceedings.mlsys.org/paper_files/paper/2025/hash/698cfaf72a208aef2e78bcac55b74328-Abstract-Conference.html)
- [Dynamic Heterogeneous Memory Management](https://arxiv.org/abs/2508.13231)

### 模型配置示例

- [Meta Llama 3 model.py](https://github.com/meta-llama/llama3/blob/main/llama/model.py)

---

## 16. 最终记忆版本

如果只能记住一条公式：

\[
\text{预取收益}
=
\text{避免重算}
-
\text{加载}
-
\text{前台干扰}
-
\text{能耗}
\]

如果只能记住三句话：

1. 机制可迁移：Radix、page、异步状态机和策略接口可以复用。
2. 实现需重构：端侧没有相同的服务器数据通路，并且资源竞争更强。
3. 调度是核心：只在计算节省大于数据移动与系统代价时读写 KV。

