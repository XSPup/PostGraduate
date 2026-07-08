# 未来网络 + AI GitHub 复现与创新项目候选

生成日期：2026-05-12

## 使用边界

这份清单的目标是帮你选出适合复现、二次开发、微调或包装成研究原型的开源项目。不要把别人的仓库简单改名当作自己的项目；更稳妥的做法是：保留原始许可证和论文引用，先做可复现实验，再加入清晰的新问题、新数据、新模型或新系统集成，最后用对比实验证明改进有效。

## 与导师方向的对应关系

黄韬老师早期公开方向集中在未来网络体系架构、内容中心网络、软件定义网络；相关未来网络实验平台工作基于 SDN/NFV、OpenStack、OpenDaylight，并强调跨域虚拟网络通信、虚拟网元管理、网络服务编排。近期相关文章进一步扩展到路由与交换、软件定义网络、内容分发网络、确定性网络、算力网络，以及自智算力网络、数字孪生、网络人工智能等方向。

因此，最适合你切入的主题不是泛泛的“AI + 通信”，而是下面几类：

- AI-native SDN / 网络路由优化
- GNN 网络性能建模与数字孪生
- 网络切片与算网资源编排
- 边缘计算任务卸载与联邦学习
- O-RAN xApp、网络安全与异常检测
- ICN / CDN / 内容缓存智能化
- 面向电信知识或网络遥测的模型微调

## 优先级结论

如果只选 3 条线，建议按下面顺序：

1. `RouteNet-Fermi + DRL-SDN Routing`：做“GNN 网络数字孪生 + 强化学习路由优化”。它最贴合 SDN、网络建模、AI for networking，容易形成论文式贡献。
2. `DeepSlicing + DROO/LyDROO`：做“算力网络中的切片与边缘计算联合编排”。它贴合算力网络、资源调度、端边云协同。
3. `MobiWatch / srsRAN RIC / FlexRIC + netFound`：做“O-RAN 网络遥测异常检测 xApp”。它工程感强，适合包装成系统演示，但环境搭建成本更高。

## 推荐仓库

| 优先级 | 仓库 | 方向 | 许可证/复用风险 | 适合做什么 |
|---|---|---|---|---|
| S | [BNN-UPC/RouteNet-Fermi](https://github.com/BNN-UPC/RouteNet-Fermi) | GNN 网络性能建模 | Apache-2.0，适合二次开发 | 网络数字孪生、时延/丢包/抖动预测、路由策略评估 |
| S | [knowledgedefinednetworking/a-deep-rl-approach-for-sdn-routing-optimization](https://github.com/knowledgedefinednetworking/a-deep-rl-approach-for-sdn-routing-optimization) | SDN 强化学习路由 | MIT，但依赖老版本 Python/Keras/OMNeT++ | 复现 DDPG 路由优化，改造成现代 RL/GNN 控制器 |
| S | [liuqiangus/DeepSlicing](https://github.com/liuqiangus/DeepSlicing) | 网络切片资源分配 | Apache-2.0，但代码量小、提交少 | 5G/6G 网络切片 DRL 资源分配基线 |
| S | [revenol/DROO](https://github.com/revenol/DROO) | MEC 任务卸载 | MIT，成熟度较高 | 边缘计算卸载、算网协同调度、无线供能 MEC |
| A | [revenol/LyDROO](https://github.com/revenol/LyDROO) | 稳定 MEC 卸载 | MIT | 在 DROO 基础上加入 Lyapunov 稳定性约束 |
| A | [NVlabs/sionna](https://github.com/NVlabs/sionna) | 6G 通信系统仿真 | Apache-2.0，工程基础好 | 6G/AI-RAN、无线数字孪生、信道/覆盖/系统级仿真 |
| A | [5GSEC/MobiWatch](https://github.com/5GSEC/MobiWatch) | O-RAN xApp + 深度学习安全 | Apache-2.0 | 5G/O-RAN 安全遥测异常检测、可解释 xApp |
| A | [SNL-UCSB/netFound](https://github.com/SNL-UCSB/netFound) | 网络遥测基础模型 | MIT | 用 PCAP/流量遥测做预训练或微调，支持异常检测/流量预测 |
| A | [srsran/oran-sc-ric](https://github.com/srsran/oran-sc-ric) | 简化 O-RAN SC RIC 环境 | AGPL-3.0，发布衍生品要谨慎 | 快速搭建 RIC + Python xApp 实验环境 |
| B | [openaicellular/flexric](https://github.com/openaicellular/flexric) | near-RT RIC / xApp SDK | OAI Public License，研究可用，商业/再发布需谨慎 | 低时延 RIC、E2/KPM/RC 数据采集、AI xApp 原型 |
| B | [icarus-sim/icarus](https://github.com/icarus-sim/icarus) | ICN 内容缓存仿真 | GPL-2.0+，衍生开源风险 | 内容中心网络/缓存策略/RL 缓存替换 |
| B | [echowei/DeepTraffic](https://github.com/echowei/DeepTraffic) | 网络流量分类/入侵检测 | MPL-2.0 | 加密流量分类、恶意流量检测、AIOps 安全基线 |
| B | [sintel-dev/sigllm](https://github.com/sintel-dev/sigllm) | LLM 时间序列异常检测 | MIT，通用工具 | 网络 KPI/流量序列异常检测与解释 |
| B | [netop-team/TeleQnA](https://github.com/netop-team/TeleQnA) | 电信领域 LLM 评测 | MIT | 构建电信知识问答评测、RAG/微调评估 |
| C | [Ali-maatouk/Tele-LLMs](https://github.com/Ali-maatouk/Tele-LLMs) | 电信大模型 | GitHub 仓库未见明确代码许可证，模型许可证需逐个查 | 可作为电信领域模型/数据来源线索，先核许可证再用 |

## 项目卡片

### 1. GNN 网络数字孪生 + RL 路由优化

推荐基线：

- [BNN-UPC/RouteNet-Fermi](https://github.com/BNN-UPC/RouteNet-Fermi)
- [knowledgedefinednetworking/a-deep-rl-approach-for-sdn-routing-optimization](https://github.com/knowledgedefinednetworking/a-deep-rl-approach-for-sdn-routing-optimization)

可以复现：

- RouteNet-Fermi 的网络时延、丢包、抖动预测。
- DDPG 在 SDN/OMNeT++ 环境里的路由权重优化。

可以创新：

- 把 RouteNet-Fermi 作为“数字孪生预测器”，先预测不同路由策略的性能，再让 RL 选择策略。
- 用 GNN/Graph Transformer 替换原始 DDPG 的扁平状态表示，让模型能泛化到不同拓扑。
- 把 reward 从单一 delay 扩展为 `delay + loss + jitter + energy + SLA violation` 多目标。
- 加入确定性网络场景：对 deadline、jitter bound、队列调度做约束。
- 做“从仿真到真实”的域随机化：训练时随机拓扑、流量矩阵、链路容量和故障。

适合项目名：

- `DT-GNN-RL: 面向自智网络的图神经数字孪生路由优化`
- `AI-Native SDN Routing with Graph-based Network Twin`

风险：

- SDN-RL 仓库较老，依赖 Python 3.6、Keras、OMNeT++ 4.6，建议用 WSL2/Ubuntu 或 Docker 复现。
- 真正创新点不能只停留在“换一个 RL 算法”，必须证明跨拓扑泛化、鲁棒性或多目标收益。

### 2. 算力网络切片与边缘计算联合编排

推荐基线：

- [liuqiangus/DeepSlicing](https://github.com/liuqiangus/DeepSlicing)
- [revenol/DROO](https://github.com/revenol/DROO)
- [revenol/LyDROO](https://github.com/revenol/LyDROO)

可以复现：

- 网络切片场景下的 DRL 资源分配。
- MEC 中根据无线信道和任务状态做在线卸载决策。
- Lyapunov-guided DRL 的稳定队列/长期约束优化。

可以创新：

- 把“网络切片资源”与“边缘算力资源”合成一个联合优化问题。
- 引入业务类型：eMBB、URLLC、mMTC、AIGC/LLM 推理任务。
- 用 Transformer 预测短期负载，再由 RL 做资源编排。
- 用多智能体强化学习让不同基站/边缘节点协同决策。
- 加入公平性、碳排放、SLA 违约惩罚，做多目标 Pareto 分析。

适合项目名：

- `SliceMEC-RL: 面向算力网络的切片-算力联合智能编排`
- `AI-assisted Computing Power Network Orchestration`

风险：

- DeepSlicing 代码小、提交少，适合做基线，但不要把它当作完整系统。
- DROO/LyDROO 偏 MEC 和无线资源，不等同于完整“算力网络”；你需要补上网络侧拓扑、带宽、时延和跨节点迁移建模。

### 3. O-RAN xApp 网络异常检测与自智运维

推荐基线：

- [5GSEC/MobiWatch](https://github.com/5GSEC/MobiWatch)
- [srsran/oran-sc-ric](https://github.com/srsran/oran-sc-ric)
- [openaicellular/flexric](https://github.com/openaicellular/flexric)
- [SNL-UCSB/netFound](https://github.com/SNL-UCSB/netFound)
- [sintel-dev/sigllm](https://github.com/sintel-dev/sigllm)

可以复现：

- MobiWatch 的 O-RAN xApp 异常检测流程。
- srsRAN 的 Docker 化 near-RT RIC + Python xApp demo。
- FlexRIC 的 E2/KPM/RC 数据采集和 Python/C xApp。
- netFound 的网络流量基础模型微调或特征提取。

可以创新：

- 将 netFound 预训练特征接入 MobiWatch，用真实/仿真遥测做微调。
- 做“检测 + 解释 + 控制”闭环：发现异常后由 xApp 输出可解释原因和控制建议。
- 对比传统 AutoEncoder、LSTM、Transformer、netFound、LLM time-series detector 的效果。
- 加入 TeleQnA/电信标准知识做 RAG，让系统能解释 3GPP/O-RAN 指标含义。
- 做鲁棒性实验：低样本、噪声遥测、跨场景迁移、未知攻击。

适合项目名：

- `NetTwinGuard: 面向 O-RAN 的可解释网络异常检测 xApp`
- `Telemetry Foundation Model for Self-Intelligent RAN Operations`

风险：

- O-RAN 环境搭建重，优先用 srsRAN 的简化 RIC 或 MobiWatch 教程，不建议一开始就上完整 O-RAN SC Kubernetes。
- srsRAN RIC 是 AGPL-3.0；如果发布衍生系统，要遵守开源义务。
- FlexRIC 使用 OAI Public License，研究测试合适，但非研究用途和再发布要认真审查许可证。

### 4. ICN / CDN 内容缓存智能优化

推荐基线：

- [icarus-sim/icarus](https://github.com/icarus-sim/icarus)

可以复现：

- ICN 中不同缓存策略的命中率、路径长度、时延表现。
- 不同拓扑和内容流行度分布下的缓存性能。

可以创新：

- 用强化学习或上下文 bandit 学习缓存替换策略。
- 用图神经网络建模内容请求在拓扑上的传播。
- 加入 CDN/边缘缓存场景：热视频、AIGC 内容、低时延应用。
- 做“内容缓存 + 算力缓存”联合：缓存模型权重、embedding、推理结果。

适合项目名：

- `AI-ICNCache: 面向内容中心网络的智能缓存优化`
- `Joint Content and Compute Caching for Future Networks`

风险：

- Icarus 是 GPL-2.0+，如果你直接修改并发布衍生代码，通常也要按 GPL 开源。
- 仓库本身不是深度学习项目，你需要自己补 AI 策略模块。

### 5. 网络遥测基础模型与电信 LLM 微调

推荐基线：

- [SNL-UCSB/netFound](https://github.com/SNL-UCSB/netFound)
- [netop-team/TeleQnA](https://github.com/netop-team/TeleQnA)
- [Ali-maatouk/Tele-LLMs](https://github.com/Ali-maatouk/Tele-LLMs)
- [sintel-dev/sigllm](https://github.com/sintel-dev/sigllm)

可以复现：

- netFound 对 PCAP/流级网络数据的表征学习。
- TeleQnA 对电信知识问答模型的评测。
- SigLLM 对时序异常的 prompt/detector 两类 pipeline。

可以创新：

- 微调一个“小型电信网络运维助手”：输入 KPI/日志/拓扑摘要，输出异常判断、原因解释、处置建议。
- 把 TeleQnA 作为知识能力评测，把网络 KPI 异常检测作为任务能力评测。
- 用 LoRA/QLoRA 只微调小模型，不从头训练。
- 做“结构化遥测 + 文本知识”的多模态 RAG：指标来自网络，解释来自标准/论文/运维手册。

适合项目名：

- `TeleOps-LLM: 面向未来网络运维的轻量电信大模型微调`
- `KPI-RAG: 基于电信知识增强的网络异常解释系统`

风险：

- Tele-LLMs 仓库本身没有明显许可证声明，模型和数据在 Hugging Face 上需要逐项检查 license。
- LLM 项目容易变成包装 demo，必须设计可量化评测：准确率、召回率、误报率、解释一致性、推理延迟。

## 最推荐的落地路线

### 路线 A：偏论文，稳妥

题目：`面向自智网络的 GNN 数字孪生辅助强化学习路由优化`

组合：

- RouteNet-Fermi 负责网络性能预测。
- DRL-SDN Routing 负责路由策略学习。
- 你新增 GNN state encoder、多目标 reward、跨拓扑泛化实验。

最低可交付：

- 能跑通 2 个原仓库的基础实验。
- 能生成一组拓扑/流量矩阵。
- 能对比 OSPF/ECMP/原始 DDPG/你的 GNN-RL。
- 指标包括平均时延、95% tail delay、丢包、SLA 违约率、训练收敛速度。

### 路线 B：偏算力网络，贴近近期方向

题目：`面向算力网络的切片-边缘任务联合智能编排`

组合：

- DeepSlicing 负责切片资源分配基线。
- DROO/LyDROO 负责边缘计算卸载基线。
- 你新增联合状态、联合动作、SLA/能耗/公平性约束。

最低可交付：

- 单节点 MEC 卸载复现。
- 多切片资源分配复现。
- 合并成一个简化算网环境。
- 对比启发式、DQN/DDPG、Lyapunov-guided RL、你的方法。

### 路线 C：偏系统，展示效果强

题目：`面向 O-RAN 的可解释网络遥测异常检测 xApp`

组合：

- srsRAN RIC 或 FlexRIC 提供 xApp 运行环境。
- MobiWatch 提供安全异常检测思路。
- netFound/SigLLM 提供模型层。

最低可交付：

- RIC/xApp demo 能跑。
- 能读取或模拟 KPM/流量/日志指标。
- 能输出异常标签、置信度、解释文本、处置建议。
- 能以 dashboard 或 CLI 展示异常检测闭环。

## 合规与写法建议

你以后写简历、开题或论文时，建议这样表述：

- “基于开源项目 X 复现了论文基线，并在 Y 场景下扩展了 Z 模块。”
- “新增了跨拓扑泛化、多目标优化、可解释检测、算网联合建模等实验。”
- “代码遵循原项目许可证，保留引用，并开源新增模块或给出复现实验脚本。”

不要这样写：

- “独立开发某某系统”，但核心代码其实来自原仓库。
- “提出全新算法”，但只是把 DQN 换成 PPO 且没有充分实验。
- “微调成自己的模型”，但没有说明数据、许可证、原模型来源和评测。

## 参考来源

- 黄韬等，未来网络技术发展趋势，DOAJ 条目：https://doaj.org/article/4ceb0f92c0e54a739de91d4ef1cb800f
- 黄韬等，面向自智算力网络的数字孪生：https://www.joconline.com.cn/zh/article/doi/10.11959/j.issn.1000-436x.2025064/
- 基于 SDN/NFV 的未来网络实验平台：https://www.telecomsci.com/zh/article/doi/10.11959/j.issn.1000-0801.2017097/
- RouteNet-Fermi：https://github.com/BNN-UPC/RouteNet-Fermi
- SDN DRL Routing：https://github.com/knowledgedefinednetworking/a-deep-rl-approach-for-sdn-routing-optimization
- DeepSlicing：https://github.com/liuqiangus/DeepSlicing
- DROO：https://github.com/revenol/DROO
- LyDROO：https://github.com/revenol/LyDROO
- Sionna：https://github.com/NVlabs/sionna
- MobiWatch：https://github.com/5GSEC/MobiWatch
- srsRAN ORAN SC RIC：https://github.com/srsran/oran-sc-ric
- FlexRIC：https://github.com/openaicellular/flexric
- Icarus：https://github.com/icarus-sim/icarus
- DeepTraffic：https://github.com/echowei/DeepTraffic
- netFound：https://github.com/SNL-UCSB/netFound
- SigLLM：https://github.com/sintel-dev/sigllm
- TeleQnA：https://github.com/netop-team/TeleQnA
- Tele-LLMs：https://github.com/Ali-maatouk/Tele-LLMs

## B站资料包项目的开源替代仓库

这些项目大多是机器学习入门练习，不是未来网络方向本身。它们的价值在于补齐 Python、数据清洗、建模、可视化、训练评估和简单部署能力。不要为了这类“资料包”付费太多；同类内容基本都能在 GitHub、Kaggle、Microsoft Learn、官方教程里找到。

### 项目对照表

| 序号 | B站资料包条目 | 推荐开源仓库 | 主要学习点 | 和未来网络方向的关系 |
|---|---|---|---|---|
| 1 | 鸢尾花分类项目 | [Ruban2205/Iris_Classification](https://github.com/Ruban2205/Iris_Classification) | sklearn 分类、KNN/SVM/逻辑回归、Streamlit 简单部署 | 只适合入门，后续可迁移到“网络流量分类” |
| 2 | Python 创建表情符号 | [carpedm20/emoji](https://github.com/carpedm20/emoji) | Python 包使用、文本到 emoji 映射、CLI/小工具 | 不是 AI 项目，只能练 Python 和交互脚本 |
| 3 | 贷款预测项目 | [Sajid030/Lending-Club-Loan-Prediction](https://github.com/Sajid030/Lending-Club-Loan-Prediction) | 表格分类、特征工程、类别不平衡、Streamlit 部署 | 可迁移到“网络故障/异常是否发生”的二分类 |
| 4 | 住房价格预测项目 | [MYoussef885/House_Price_Prediction](https://github.com/MYoussef885/House_Price_Prediction) | 回归、XGBoost、特征处理、误差指标 | 可迁移到“网络时延/吞吐量预测” |
| 5 | MNIST 数字识别项目 | [pytorch/examples/mnist](https://github.com/pytorch/examples/tree/main/mnist) | PyTorch 训练循环、CNN、图像分类 | 深度学习基本功，不是研究项目 |
| 6 | 股价预测项目 | [huseinzol05/Stock-Prediction-Models](https://github.com/huseinzol05/Stock-Prediction-Models) | 时间序列预测、LSTM、交易模拟 | 可迁移到“网络流量/KPI 时间序列预测” |
| 7 | 泰坦尼克生存预测 | [mrankitgupta/titanic-survival-prediction-93-xgboost](https://github.com/mrankitgupta/titanic-survival-prediction-93-xgboost) | Kaggle 分类全流程、EDA、XGBoost | 只适合练完整 ML pipeline |
| 8 | 葡萄酒质量检测项目 | [sharmaroshan/Wine-Quality-Predictions](https://github.com/sharmaroshan/Wine-Quality-Predictions) | UCI 数据集、回归/分类、可视化 | 可迁移到“QoS 等级预测” |
| 9 | 假新闻检测项目 | [safe-graph/GNN-FakeNews](https://github.com/safe-graph/GNN-FakeNews) | NLP、图神经网络、社交传播图建模 | 可迁移到“图神经网络 + 网络拓扑/告警传播” |
| 10 | 音乐流派分类项目 | [ruohoruotsi/LSTM-Music-Genre-Classification](https://github.com/ruohoruotsi/LSTM-Music-Genre-Classification) | 音频特征、MFCC、LSTM、PyTorch/Keras | 可迁移到“无线频谱/信号特征分类” |
| 11 | 比特币价格预测项目 | [SC4RECOIN/LSTM-Crypto-Price-Prediction](https://github.com/SC4RECOIN/LSTM-Crypto-Price-Prediction) | 金融时间序列、LSTM、趋势分类 | 可迁移到“网络负载趋势预测”，但不要用于投资判断 |
| 12 | Uber 数据分析项目 | [FiveThirtyEight/uber-tlc-foil-response](https://github.com/fivethirtyeight/uber-tlc-foil-response) | 出行数据分析、时空聚合、可视化 | 可迁移到“流量时空分布分析” |
| 13 | 客户细分项目 | [pramodkondur/Customer-Segmentation-RFM-CLV](https://github.com/pramodkondur/Customer-Segmentation-RFM-CLV) | KMeans/RFM/CLV、聚类分析 | 可迁移到“网络用户/业务类型聚类” |
| 14 | 情感分析项目 | [bentrevett/pytorch-sentiment-analysis](https://github.com/bentrevett/pytorch-sentiment-analysis) | NLP、RNN/CNN/Transformer/BERT 文本分类 | 可迁移到“电信工单/日志文本分类” |
| 15 | 语音情感识别项目 | [Meghashyam-adimallam/speech-emotion-recognition](https://github.com/Meghashyam-adimallam/speech-emotion-recognition) | MFCC、BiLSTM、音频分类、Streamlit | 可迁移到“信号序列分类”，但和未来网络关系较弱 |
| 16 | 非法捕鱼监测项目 | [geetakingle/Detecting-Illegal-Fishing](https://github.com/geetakingle/Detecting-Illegal-Fishing) | AIS 轨迹、异常检测、半监督学习 | 和“网络异常检测/轨迹异常检测”方法相通 |
| 17 | 电影推荐系统项目 | [prakruti-joshi/Movie-Recommendation-System](https://github.com/prakruti-joshi/Movie-Recommendation-System) | 协同过滤、SVD、混合推荐、MovieLens | 可迁移到“网络资源/服务推荐” |
| 18 | 车牌识别项目 | [ankandrew/fast-alpr](https://github.com/ankandrew/fast-alpr) | YOLO/ONNX/OCR、目标检测 + 文本识别 | 计算机视觉工程项目，和未来网络主线关系弱 |

### AI 学习路线图和精选资料的免费替代

如果卖家资料包里还包含“AI 学习路线图”，优先看这些免费仓库：

| 仓库 | 适合阶段 | 用法 |
|---|---|---|
| [microsoft/ML-For-Beginners](https://github.com/microsoft/ML-For-Beginners) | 机器学习入门 | 按回归、分类、聚类、NLP、时间序列顺序刷 |
| [microsoft/AI-For-Beginners](https://github.com/microsoft/AI-For-Beginners) | AI 概念和基础实践 | 用来补 AI 基础、搜索、知识表示、神经网络 |
| [microsoft/generative-ai-for-beginners](https://github.com/microsoft/generative-ai-for-beginners) | 生成式 AI 入门 | 学 prompt、RAG、agent、应用搭建 |
| [ageron/handson-ml3](https://github.com/ageron/handson-ml3) | 系统学习 ML/DL | 配合《Hands-On Machine Learning》第三版 notebook |
| [rasbt/machine-learning-book](https://github.com/rasbt/machine-learning-book) | sklearn + PyTorch 基础 | 适合从传统 ML 过渡到深度学习 |

### 建议学习顺序

如果你的目标仍然是未来网络 + AI，不建议把 18 个项目全部刷一遍。更合理的顺序是：

1. 先做 1、3、4、5：掌握 sklearn、表格数据、回归/分类、PyTorch 训练循环。
2. 再做 6、11、13：掌握时间序列预测和聚类，这两类最容易迁移到网络 KPI、流量预测、用户行为分析。
3. 再做 9、14：掌握 GNN 和 NLP，为网络拓扑建模、告警关联、电信工单分析做准备。
4. 最后选 16 或 17：练异常检测或推荐系统，分别对应自智网络运维和资源/服务推荐。

### 可以包装成你自己项目的方向

这些入门项目不要直接改名当作品。可以这样升级：

- 把“住房价格预测”改造成“网络链路时延预测”：输入链路容量、队列长度、流量矩阵，输出 delay/loss/jitter。
- 把“股价/比特币预测”改造成“网络流量预测”：用 LSTM/Transformer 预测未来 5 分钟或 1 小时流量。
- 把“客户细分”改造成“业务流/用户画像聚类”：按流量、时延敏感度、带宽需求划分业务类型。
- 把“假新闻 GNN”改造成“网络拓扑告警传播 GNN”：节点是路由器/基站/网元，边是拓扑或依赖关系。
- 把“非法捕鱼异常检测”改造成“网络遥测异常检测”：AIS 轨迹换成 KPI 时间序列或网元行为轨迹。
- 把“推荐系统”改造成“边缘节点/网络切片/服务函数推荐”：为任务选择最合适的计算节点或网络路径。

### 许可证提醒

上表中有些仓库许可证明确，例如 MIT、Apache-2.0 或 GPL；也有部分仓库页面未清晰显示许可证。后续真正 clone、复现或公开自己的改版项目时，先做三件事：

1. 检查仓库根目录是否有 `LICENSE`。
2. 在 README 里保留原仓库链接和论文/数据集引用。
3. 自己新增的代码、实验记录、数据处理脚本和结果图单独放清楚，避免把“整理资料”包装成“原创项目”。
