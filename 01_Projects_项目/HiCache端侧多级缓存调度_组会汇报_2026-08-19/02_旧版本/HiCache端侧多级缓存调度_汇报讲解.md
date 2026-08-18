# 面向端侧大模型推理的分层 KV 缓存调度研究

> 组会日期：2026 年 8 月 19 日  
> 建议题目：面向端侧大模型推理的分层 KV 缓存调度研究——HiCache 机制迁移与端侧自适应预取 / 异步写回  
> 核心判断：HiCache 的控制思想可以迁移到端侧，但服务器数据通路不能照搬；最值得研究的是端侧约束下的读写联合调度。

---

## 0. 阅读和汇报口径

这份文档用于补充 PPT 放不下的推导、算法、实现细节和答辩问题。汇报时必须区分四种内容：

- 已有事实：SGLang HiCache 已提供 GPU—主机内存—外部存储的三级 KV 缓存，以及预取、加载、备份和写策略。
- 文献结论：论文或官方案例中的性能数字只能说“该工作报告……”，不能当作本课题结果。
- 本课题提案：Edge-HiCache 和 EAHS 是拟研究的架构与算法，目前还没有实测结论。
- 演示假设：PPT 第 13 页的带宽和时间只用于解释 break-even，必须用目标设备实测值替换。

整场汇报的因果链是：

1. 长前缀复用场景存在；
2. KV 容量随上下文线性增长，端侧无法全部常驻；
3. KV 被逐出后会造成重复 Prefill；
4. HiCache 提供了服务器侧的分层缓存和异步调度骨架；
5. 端侧的统一内存、共享带宽、能耗、热和闪存使固定策略失效；
6. 因此提出 Edge-HiCache 数据面和 EAHS 在线读写调度；
7. 先通过 microbenchmark 判断收益上限，再决定是否继续实现复杂算法。

---

## 1. 一句话说明研究方向

在多轮会话、固定系统提示、端侧 RAG 和 Agent 场景中，请求经常共享长前缀。加速器可用内存有限时，前缀 KV 会被逐出，后续请求必须重复 Prefill。

拟研究系统把较冷的前缀 KV 分层保存在 LPDDR 和 UFS/SSD 中，并由在线调度器回答：

- 哪些 KV 值得提前读？
- 什么时候读，是否能在 deadline 前完成？
- 读操作会不会抢占 Decode 所需内存带宽？
- 哪些新 KV 值得异步写入持久层？
- 什么时候应该暂停、取消、抢占或合并写入？

这不是单纯“增加一层缓存”，而是一个同时受延迟、能耗、热、共享带宽、容量和闪存写入约束的在线决策问题。

---

## 2. 背景：为什么 KV Cache 会成为端侧瓶颈

### 2.1 Prefill、Decode 与 TTFT

大模型推理通常分为：

- Prefill：并行处理输入 token，生成每层注意力模块的 Key/Value；
- Decode：逐 token 生成，每一步读取历史 KV，并追加新 KV。

首 token 延迟可分解为：

\[
T_{\mathrm{TTFT}}
=T_{\mathrm{queue}}
+T_{\mathrm{model/load}}
+T_{\mathrm{prefix\ KV\ load}}
+T_{\mathrm{prefill}}
+T_{\mathrm{first\ decode}}
\]

分层 KV 缓存试图降低重复的 \(T_{\mathrm{prefill}}\)，但会新增 \(T_{\mathrm{prefix\ KV\ load}}\) 和后台 I/O 干扰。因此，“命中”本身不是充分条件，真正条件是数据加载比重算更便宜。

### 2.2 KV 容量公式

设：

- \(L\)：Transformer 层数；
- \(H_{kv}\)：KV 头数；
- \(d_h\)：每头维度；
- \(n\)：上下文 token 数；
- \(b\)：每个元素的字节数。

忽略对齐、分页元数据和分配器碎片时：

\[
M_{KV}(n)=2LH_{kv}d_hnb
\]

系数 2 来自 K 和 V 两份张量。每 token 大小为：

\[
m_{token}=2LH_{kv}d_hb
\]

以常见的 Llama 3 8B GQA、FP16 KV 配置作为量级例子：

\[
L=32,\quad H_{kv}=8,\quad d_h=128,\quad b=2\ \mathrm{bytes}
\]

\[
m_{token}
=2\times32\times8\times128\times2
=131072\ \mathrm{bytes}
=128\ \mathrm{KiB/token}
\]

于是：

- 8K token：约 1 GiB；
- 32K token：约 4 GiB；
- 128K token：约 16 GiB。

必须说明：这只是给定模型配置下的量级示例。MLA、滑动窗口注意力、不同 GQA 比例、KV 量化和不同精度都会改变结果。

### 2.3 前缀命中的收益推导

设一轮请求由可复用前缀 \(k\) 和新问题 \(s\) 个 token 组成。

无缓存：

\[
T_{no\ cache}=C(k+s)
\]

命中并加载前缀 KV：

\[
T_{hit}=R(k)+C(s)+I
\]

其中：

- \(C(x)\)：Prefill 计算 \(x\) 个 token 的时间；
- \(R(k)\)：把前缀 KV 加载到可计算状态的完成时间；
- \(I\)：加载给前台计算造成的干扰。

收益：

\[
\Delta T=C(k+s)-C(s)-R(k)-I
\]

若一阶近似 \(C(x)\approx ax+c\)，则：

\[
\Delta T\approx ak-R(k)-I
\]

基本 break-even 条件：

\[
C(k)>R(k)+I
\]

还必须满足兼容性：模型、模型版本、tokenizer、位置编码、KV 数据类型、页面布局和 token 序列一致。任意一项不一致，都不能复用旧 KV。

### 2.4 为什么命中率不能单独代表性能

高命中率仍可能变慢：

- KV 来自慢速 UFS，加载时间超过重算；
- 后台加载抢占统一内存带宽，使 Decode 的 ITL 上升；
- wait-complete 为慢尾 I/O 阻塞请求；
- write-through 把无复用 KV 也写入闪存；
- 预取最终未使用的 page，浪费 I/O 和容量；
- 后台读写造成温升和降频。

因此评价必须同时看 TTFT、ITL、有效命中字节、无效预取、能耗、温度和写入量。

---

## 3. HiCache 的结构与调度机制

### 3.1 三层结构

SGLang HiCache 在 RadixAttention 基础上引入：

- L1 GPU/HBM：最快、最小，直接参与计算；
- L2 Host DRAM：较大，用于本地前缀复用；
- L3 外部或分布式存储：例如 Mooncake、3FS、NIXL。

HiRadixTree 记录连续 token 前缀及其 L1/L2 位置；L3 元数据在访问时向后端查询。

“连续前缀”是算法约束：注意力需要从序列起点开始的连续历史。离散命中的后缀 page 不能直接当作可消费前缀。

### 3.2 请求路径

一次请求可简化为：

1. 在 L1/L2 进行本地前缀匹配；
2. 对未命中部分查询 L3；
3. 满足条件时发起 L3→L2 预取；
4. 将所需 KV 从 L2 加载到 L1；
5. 执行剩余 Prefill 和 Decode；
6. 按策略把新 KV 写到 L2/L3。

官方最佳实践中，默认只有连续 L3 命中不少于 256 token 才触发预取。该规则适合作为默认值，但没有使用目标设备的 Prefill 速度、实时带宽、运行 phase、能耗和温度，这正是端侧自适应的切入点。

### 3.3 预取策略

HiCache 提供：

- best_effort：GPU 可以执行时停止等待；
- wait_complete：等待预取全部完成；
- timeout：完成或达到超时即停止。

官方 timeout 形式：

\[
T_{timeout}=\min
\left(
T_{max},
T_{base}+\alpha\frac{n_{fetch}}{1024}
\right)
\]

默认示例为基础 2 秒、每 1024 token 增加 0.1 秒、最大 30 秒。它是基于 token 数的规则，不直接判断“继续等是否比重算更划算”。

### 3.4 写策略

- write-through：每次访问都向下一层写，可靠但写放大最大；
- write-through-selective：访问超过阈值才写；
- write-back：逐出时写，写入较少，但需要处理延迟写和一致性。

端侧还要计入能耗、温升、UFS 占用和闪存寿命，所以不能只优化缓存命中率。

### 3.5 异步实现中值得迁移的骨架

当前缓存控制器源码体现了：

- 独立的 prefetch 和 backup 后台线程；
- 独立队列和加载/写入设备流；
- 写操作合并；
- L2→L1 按层搬运，与逐层计算重叠；
- Host Pool 容量保护；
- 预取停止、取消、备份和回收状态。

真正值得迁移的是 page 粒度、层级元数据、异步状态机、双队列以及策略接口，而不是 CUDA/RDMA 实现。

---

## 4. 端侧移植判断

结论：能迁移控制思想，不能照搬服务器数据通路。

| HiCache 组成 | 端侧复用性 | 处理方式 |
|---|---:|---|
| Radix 前缀元数据 | 高 | 保留，增加模型和格式指纹 |
| page 粒度 | 高 | 保留，page size 由实测确定 |
| get / exists / set 接口 | 高 | 映射到 LPDDR/UFS 后端 |
| 预取终止与写策略状态机 | 高 | 改成在线自适应 |
| prefetch / backup 双队列 | 高 | 增加 phase 门控和抢占 |
| CUDA kernel、pinned memory | 低 | 按端侧 NPU/GPU 运行时重写 |
| RDMA、共享 L3、TP all-reduce | 低 | 单端删除或重构 |
| PD 解耦 | 低到中 | 端侧通常没有同样的集群结构 |

### 4.1 建议的逻辑层次

- L1：加速器当前可以直接消费、格式与精度已就绪的 KV 池；
- L2：LPDDR 温缓存，可能需要映射、复制或格式转换；
- L3：UFS/SSD 持久前缀缓存；
- EdgeRadix：token 前缀、page 位置、热度、版本和状态元数据；
- EAHS：收集状态并输出 prefetch、cancel、write、evict、compress、recompute。

在统一内存 SoC 上，L1/L2 未必对应两块物理内存。它们可以是“可见性、格式、运行时状态和容量”的逻辑层级。

### 4.2 端侧新增约束

1. 共享带宽：Decode 往往 memory-bound，后台 I/O 会竞争 LPDDR。
2. 能耗与热：后台任务可能触发降频，扩大长尾。
3. UFS 长尾：小块随机 I/O 和系统活动造成波动。
4. 闪存写入：无选择 write-through 增加写放大。
5. 容量更小：需要更积极的选择、压缩和逐出。
6. 隐私和生命周期：持久 KV 需要用户隔离、加密、登出清理和版本失效。

---

## 5. 相关工作与候选研究空白

已有工作已经覆盖相邻问题：

- SGLang/HiCache：RadixAttention、三级 KV、预取和写策略；
- Mooncake：服务器集群的分离式 KVCache 与跨实例复用；
- SparKV（2026 预印本）：端侧在云端流式 KV 与本地重算之间做开销感知选择；
- Bidaw（FAST 2026）：Host—SSD 两层交互式推理和计算—存储感知；
- NEO/FlexInfer（MLSys 2025）：CPU offload、异构资源和 phase-aware 路径选择；
- KIVI：2-bit KV 量化，降低 KV 字节数。

所以不能声称“首次做多级 KV”“首次做端侧 KV 加载”或“首次做 phase-aware”。

当前更稳妥的候选贡献：

> 面向单端本地持久前缀复用，建立统一内存干扰模型，并对读预取和选择性写回进行 deadline/value/phase-aware 联合调度，同时显式考虑能耗、热和闪存写入预算。

开题前仍需继续检索并复现，重点关键词：

- on-device persistent prefix KV cache；
- unified-memory-aware KV offload；
- flash-aware KV cache；
- joint prefetch writeback scheduling for LLM inference；
- deadline-aware KV cache scheduling。

---

## 6. Edge-HiCache 数据面

建议 cache key 至少包含：

\[
key=H(
model\_id,\ model\_revision,\ tokenizer\_hash,\
rope\_config,\ kv\_dtype,\ page\_layout,\ token\_ids
)
\]

否则模型升级、tokenizer 或位置编码变化后，旧 KV 可能产生静默错误。

EAHS 持续观测：

- 连续前缀命中长度；
- 队列等待与请求 deadline；
- L2/L3 实测带宽、固定开销和尾延迟；
- Prefill、Decode、queue slack、idle phase；
- 加速器利用率与 LPDDR 压力；
- 能量、电量、温度和降频；
- L1/L2/L3 可用容量；
- 会话和前缀复用统计；
- 持久层写入预算。

---

## 7. EAHS 数学模型

### 7.1 page 状态

对候选 page \(i\) 定义：

- \(k_i\)：覆盖 token 数；
- \(b_i\)：字节数；
- \(C_i\)：重算该 page 可避免的前台时间；
- \(R_i\)：从当前层加载到可计算状态的完成时间；
- \(W_i\)：写入目标层的时间或代价；
- \(E_i^r,E_i^w\)：读写能量；
- \(I_i^r,I_i^w\)：对前台的干扰；
- \(p_i^r\)：预测性预取后实际使用概率，需求加载时可设为 1；
- \(p_i^H\)：未来时间窗 \(H\) 内再次复用概率；
- \(d_i\)：读取 deadline；
- \(slack_i=d_i-now-R_i\)；
- \(Wear_i\)：闪存写入机会成本。

### 7.2 全局目标

\[
\min J=
p95(TTFT)
+\alpha E_{request}
+\beta FlashWrite
+\gamma SLOmiss
+\eta p95(ITL)
\]

更适合工程实现的约束式：

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

权重或指标必须先按基线、预算归一化，不能直接把不同量纲硬相加。

### 7.3 读收益

\[
U_r(i)=p_i^rC_i-R_i-\lambda_EE_i^r-\lambda_II_i^r
\]

执行条件：

\[
U_r(i)>0
\quad\land\quad
now+R_i\le d_i
\]

并满足内存、I/O、能量和温度预算。

解释：

- \(p_i^rC_i\)：期望避免的重算；
- \(R_i\)：搬运完成时间；
- \(E_i^r\)：能耗；
- \(I_i^r\)：对 Decode/Prefill 的干扰。

### 7.4 写收益

\[
U_w(i)=
p_i^H(C_i-R_i)
-W_i
-\lambda_FWear_i
-\lambda_M\frac{b_i}{C_3}
-\lambda_II_i^w
\]

其中 \(b_i/C_3\) 表示占用持久层容量的机会成本。只有 \(U_w(i)>0\) 才进入写队列；相邻 page 应合并写以降低小块 I/O 和写放大。

### 7.5 排序

读候选先按 deadline 分桶，再按价值密度排序：

\[
Priority_r(i)=
\frac{U_r(i)}{R_i+\varepsilon}
+\frac{\kappa}{slack_i+\varepsilon}
\]

第一项偏向单位时间收益高的 page，第二项偏向即将到期的 page。实现前需归一化。

连续前缀约束：

- page \(i\) 之前若有未就绪前缀，\(i\) 不能独立消费；
- 调度从最长连续前缀边界向后扩展；
- 当后续 page 累计收益转负时停止，实现部分命中。

### 7.6 phase-aware 门控

| phase | 读策略 | 写策略 |
|---|---|---|
| Decode 活跃 | 只保留高价值、即将到期的读 | 暂停或抢占低优先级写 |
| Prefill 活跃 | slowdown 小于阈值时限速重叠 | 仅高价值写，严格限速 |
| Queue slack | 优先完成即将到期的读 | 少量写 |
| Idle | 补齐高价值温缓存 | 合并并排空写队列 |

核心不是“尽早搬”，而是“在 deadline 前完成且不伤害前台”。

### 7.7 在线算法步骤

收到请求：

1. EdgeRadix 查找最长兼容连续前缀。
2. 从连续命中边界开始构造 page 候选。
3. 对每个 page 预测 \(C,R,I,E,p,d\)。
4. 计算 \(U_r\)；若收益非正或预计错过 deadline，停止继续扩展。
5. 将候选按 deadline/value 放入读队列。
6. 到 deadline、全读完或继续等待不再划算时终止。
7. 只采用已经形成连续前缀的部分，其余重算。

phase 变化：

1. Decode：抢占写，取消已转负的低价值读。
2. Prefill：依据实测 slowdown 限制后台 I/O。
3. Queue slack：调度最早到期的读。
4. Idle：合并并排空高价值写。

产生新 KV：

1. 预测未来复用概率和避免重算价值。
2. 扣除写入、干扰、容量和磨损成本。
3. 只有 \(U_w>0\) 才进入写队列。

若候选数为 \(N\)，评分为 \(O(N)\)，优先队列构建约 \(O(N\log N)\)。真正难点是预测误差和抢占时机，而不是算法复杂度。

---

## 8. 第 13 页 break-even 示例

假设：

- page = 64 token；
- 128 KiB/token，所以 page = 8 MiB；
- L3 有效带宽 1.5 GB/s；
- 固定开销 0.3 ms。

\[
R\approx
\frac{8\ \mathrm{MiB}}{1.5\ \mathrm{GB/s}}
+0.3\ \mathrm{ms}
\approx5.9\ \mathrm{ms/page}
\]

若避免重算 \(C=12\) ms，暂不计能耗和干扰：

\[
U\approx12-5.9=6.1\ \mathrm{ms}>0
\]

值得读。

若总线拥塞使实际完成时间升至 16 ms：

\[
U\approx12-16=-4\ \mathrm{ms}<0
\]

应取消预取并重算。

这些数字不是实验结果，只解释为什么必须在线测量 \(R\) 和干扰，不能永久使用固定 256-token 阈值。

---

## 9. 实现路线

### 阶段 A：1—3 周，测收益上限

测量：

- Prefill 每 token/page 的 \(C(k)\)；
- L2/L3 在不同 page size 下的带宽、固定开销、p95/p99；
- Prefill/Decode 与后台读写并发时的 slowdown；
- KV 格式转换和量化/反量化开销；
- 温度、功耗和降频；
- 队列深度与 I/O 合并粒度。

拟合：

\[
\hat C(k,phase),\quad
\hat R(b,phase,q),\quad
\hat I(b,phase),\quad
\hat E(b,phase)
\]

若大多数真实场景满足 \(R\ge C\)，应及时缩小问题或转向 KV 量化/重算选择，不能继续堆调度复杂度。

### 阶段 B：4—8 周，单机原型

平台选择：

- Jetson/CUDA：较接近 HiCache 异步流机制；
- Android：考虑 MLC/llama.cpp 等运行时钩子；
- Ascend/MindIE：先确认 KV page、异步拷贝和缓存接口是否暴露。

最小接口只需：

- prefix_match；
- page_load/page_store；
- phase；
- timing/energy observation。

第一版先用精确 KV，避免同时引入量化误差；量化放到后续消融。

### 阶段 C：9—12 周，真实端侧验证

加入 UFS 持久层、能耗、温度、写入量、真实多轮会话/RAG/Agent trace、会话生命周期和安全清理。

---

## 10. 实验设计

### 10.1 基线

相同模型、page size、KV 精度和缓存容量下比较：

1. 无前缀缓存；
2. 仅 L1 Radix；
3. 固定阈值 best-effort；
4. wait-complete；
5. timeout；
6. write-through；
7. write-through-selective；
8. write-back；
9. EAHS；
10. 去掉 deadline、value、phase、wear 项的 EAHS 消融。

若平台兼容，再加入 KIVI、SparKV 或动态 offload 方法。

### 10.2 工作负载

- 固定 system prompt + 不同问题；
- 多轮会话；
- 端侧 RAG，共享文档前缀；
- Agent，共享工具说明和历史；
- 热点前缀 Zipf 分布；
- 冷请求对照；
- 不同前缀长度、到达率、并发度；
- Decode 期间 UFS I/O 干扰。

### 10.3 指标

- p50/p95/p99 TTFT；
- p50/p95 ITL；
- 吞吐；
- token/page/层级命中率；
- 有效加载和无效预取字节；
- 取消/超时次数；
- 每请求能量；
- 温度、频率和热节流；
- UFS 写入字节和合并率；
- 缓存占用、逐出；
- \(|\hat R-R|\)、\(|\hat C-C|\)。

### 10.4 暂定门槛

相对最优静态基线：

- p95 TTFT 降低至少 20%；
- p95 ITL 回退不超过 5%；
- 每请求能耗不增加；
- 相对 write-through，持久层写入降低至少 30%。

这些是研究目标，不是已有结果。

### 10.5 统计要求

- 报告中位数、p95/p99 和置信区间；
- 预热后测量；
- 固定设备初始温度或报告温度轨迹；
- 随机化请求顺序；
- 每组重复多轮；
- 同时报告高复用、低复用和无收益场景。

---

## 11. 15 分钟汇报节奏

正式讲述建议控制在 13 分 45 秒，保留约 1 分钟机动。

| 页 | 时间 | 本页结论 | 转场 |
|---:|---:|---|---|
| 1 | 20 秒 | 不是把 SGLang 搬到手机，而是研究端侧调度 | 先看问题为什么存在 |
| 2 | 45 秒 | 长前缀复用 + KV 容量不足导致重复 Prefill | 先量化 KV 多大 |
| 3 | 55 秒 | 8B 模型的长上下文 KV 也会达到 GiB 级 | 命中到底省什么 |
| 4 | 55 秒 | 收益是避免前缀全层计算，但要加载更便宜 | HiCache 如何做 |
| 5 | 60 秒 | L1/L2/L3 + HiRadixTree + 请求路径 | 关键在策略 |
| 6 | 65 秒 | 预取、等待、写回、线程和流构成调度入口 | 哪些能迁移 |
| 7 | 45 秒 | 复用元数据/page/状态机，不复用 CUDA/RDMA | 端侧多了什么约束 |
| 8 | 55 秒 | 统一内存、能耗、热、UFS 使后台 I/O 不免费 | 是否已有工作解决 |
| 9 | 65 秒 | 候选空白是本地持久前缀的读写联合调度 | 给出系统方案 |
| 10 | 65 秒 | 三层数据面 + 可观测、可抢占的 EAHS | 调度优化什么 |
| 11 | 70 秒 | 期望节省减迁移、干扰、能耗和写入代价 | 公式如何在线执行 |
| 12 | 75 秒 | 读按 deadline/value，写按 phase 门控 | 看一个 break-even |
| 13 | 55 秒 | 同一 page 空闲时值得读，拥塞时应取消 | 如何验证 |
| 14 | 75 秒 | 三阶段、完整基线、指标和止损条件 | 三句总结 |
| 15 | 20 秒 | 机制可迁移、实现需重构、调度是核心 | 请老师指导 |

---

## 12. 可能被问到的问题

### Q1：这不就是普通多级缓存吗？

回答：

> 多级缓存本身不是创新。HiCache、Mooncake 和 Host-SSD 工作都已存在。本课题研究的是端侧统一内存和持久闪存约束下，读预取与选择性写回如何联合调度。候选贡献是 deadline/value/phase 感知、前台干扰建模和写入预算。创新性仍需文献检索与基线复现确认。

### Q2：统一内存为什么还有 L1/L2？

> 这里是逻辑层级。L1 是当前运行时可直接消费、格式已就绪的 KV；L2 是仍需映射、转换或显式迁移的温缓存。层级由可见性、格式和容量共同定义。

### Q3：直接量化 KV 不就行了吗？

> 量化解决每 token 占多少字节，分层缓存解决哪些历史常驻、哪些低成本恢复。二者正交。第一版固定精度验证调度收益，再把 INT8/INT4 作为扩展和消融。

### Q4：为什么不直接使用 HiCache timeout？

> timeout 主要按 token 数构造上限，没有显式使用端侧实时带宽、Decode 干扰、能量、温度和 UFS 写入状态。EAHS 把这些状态纳入在线收益估计。

### Q5：复用概率怎么估计？

第一版采用低成本特征：会话内近期性、访问次数、指数衰减热度、请求类型、前缀长度和会话是否活跃。可以使用逻辑回归或轻量 GBDT。若预测开销过高，就不适合端侧。

### Q6：如何保证 KV 正确性？

> key 绑定模型版本、tokenizer、位置编码、精度和页面布局；只复用最长连续、完全兼容的前缀。模型更新、登出或配置变化时失效。命中与重算的 logits/输出误差必须经过测试。

### Q7：预取一半时 deadline 到了怎么办？

> 只采用已经连续就绪的前缀，其余取消并重算。离散后缀 page 不能当作有效命中。

### Q8：怎样测前台干扰？

> 对每个 phase 比较无后台 I/O 与不同带宽/队列深度下的 TTFT、ITL slowdown，拟合 \(I(b,phase,q)\)，运行时再用滑动窗口更新。Decode 期预算更严格。

### Q9：闪存磨损如何量化？

> 第一阶段用逻辑写入字节和合并率作为代理；平台能提供统计时再加入物理写放大或健康信息。算法采用写入预算约束，不声称精确预测寿命。

### Q10：没有重复前缀怎么办？

> 系统退化为普通推理，不持久化冷 KV。低 \(p_i^H\) 会使写收益转负。冷 workload 是必须报告的失败/无收益场景。

### Q11：为什么还要看 ITL？

> 分层缓存主要改善 TTFT，但端侧读写会抢占 Decode 内存带宽。如果只优化 TTFT，可能造成生成卡顿，所以 ITL 必须是约束或惩罚项。

### Q12：与截图中的 MindIE Motor / Mooncake 有什么关系？

> 截图方向强调 KV 亲和调度和服务器集群内跨实例复用。本课题吸收“利用历史 KV、避免重复 Prefill”的动机，但把边界缩到单端本地 LPDDR/UFS、统一内存竞争、能耗和选择性写回。二者相关，但不是同一个实现问题。

---

## 13. 汇报中容易说错的话

- 不说“HiCache 已支持手机”，说“HiCache 提供可借鉴的服务器控制机制”。
- 不说“预取一定降低 TTFT”，说“当 \(C>R+I\) 且能在 deadline 前完成时才有效”。
- 不说“EAHS 已经有效”，说“EAHS 是拟验证方案”。
- 不说“128K 一定需要 16 GiB”，说“在给定 Llama 3 8B GQA、FP16 KV 示例下约 16 GiB”。
- 不把论文或博客数字当自己的结果。
- 在检索与复现完成前不使用“首次”，使用“候选贡献”和“拟研究空白”。

---

## 14. 建议立即开始的工作

1. 确认可用端侧硬件与推理运行时。
2. 确认是否暴露 KV page、prefix match、异步 load/store 和 phase。
3. 选择一个 7B/8B 模型，固定精确 KV，测 \(C(k)\)、\(R(b)\)、\(I(b,phase)\)。
4. 用真实会话 trace 统计连续前缀长度和复用间隔。
5. 先实现 best-effort、wait-complete、timeout，再实现 EAHS。
6. 一个月内得到 break-even 区域图：横轴前缀/page，纵轴带宽或负载，颜色表示净收益。
7. 如果正收益区域很小，转向 KV 量化、端云加载—重算选择或更小范围的 L1/L2 调度。

---

## 15. 主要参考资料

### 官方文档与源码

- [SGLang HiCache Best Practices](https://docs.sglang.io/docs/advanced_features/hicache_best_practices)
- [SGLang HiCache Design](https://docs.sglang.io/docs/advanced_features/hicache_design)
- [SGLang cache_controller.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/cache_controller.py)
- [SGLang / RadixAttention](https://arxiv.org/abs/2312.07104)
- [SGLang HiCache Blog](https://www.lmsys.org/blog/2025-09-10-sglang-hicache/)

### 相关系统与论文

- [Mooncake, FAST 2025](https://www.usenix.org/conference/fast25/presentation/qin)
- [SparKV, 2026](https://arxiv.org/abs/2604.21231)
- [Dynamic Heterogeneous Memory Management for LLM Inference, 2025](https://arxiv.org/abs/2508.13231)
- [Bidaw, FAST 2026](https://www.usenix.org/conference/fast26/presentation/hu-shipeng)
- [NEO, MLSys 2025](https://proceedings.mlsys.org/paper_files/paper/2025/hash/66a026c0d17040889b50f0dfa650e5e0-Abstract-Conference.html)
- [FlexInfer, MLSys 2025](https://proceedings.mlsys.org/paper_files/paper/2025/hash/698cfaf72a208aef2e78bcac55b74328-Abstract-Conference.html)
- [KIVI](https://arxiv.org/abs/2402.02750)
- [KVCache: Cache in the Wild, USENIX ATC 2025](https://www.usenix.org/conference/atc25/presentation/wang-jiahao)

### 模型配置示例

- [Meta Llama 3](https://ai.meta.com/blog/meta-llama-3/)
- [Meta Llama 3 model.py](https://github.com/meta-llama/llama3/blob/main/llama/model.py)

---

## 最终收束

本课题最重要的逻辑不是“缓存越多越好”，而是：

\[
\boxed{
\text{只在预计计算节省}
>
\text{数据移动 + 前台干扰 + 能耗 + 写入机会成本}
\text{时执行}
}
\]

整场汇报收束为三句话：

1. 机制可迁移：Radix 元数据、page、异步状态机和策略接口值得复用。
2. 实现需重构：端侧没有相同的 CUDA/RDMA/集群数据通路，统一内存、热和闪存约束更强。
3. 调度是研究核心：用 deadline、value、phase 和写入预算决定何时预取、等待、取消和写回。
