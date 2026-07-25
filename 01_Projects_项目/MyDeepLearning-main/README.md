# MyDeepLearning 学习工作区

该目录用于长期保存机器学习、深度学习以及网络与 AI 方向的学习和研究材料。稳定材料、个人练习、研究规划和临时实验分开管理，避免把所有 notebook 与脚本堆在同一层。

## 目录说明

| 目录 | 用途 | Git 约定 |
|---|---|---|
| `LearningNotes/` | 课程笔记、可复用示例和长期学习材料 | 稳定内容可跟踪 |
| `LearningNotes/d2l-pytorch-notebooks/` | 《动手学深度学习》PyTorch notebook 学习套件 | 已筛选并跟踪；详见其 README |
| `MyHandwrittenNotebookwithLIMU/` | 个人跟学、手写练习 | 有学习价值的版本可跟踪 |
| `ResearchPlanning/` | 学习路线、研究方向和实验计划 | 跟踪规划与结论 |
| `OpenSourceProjectPractice/` | 开源项目复现和实践 | 当前按本地实践区管理 |
| `ScratchNotebooks/` | 临时 notebook 草稿 | 默认不跟踪 |
| `ScratchScripts/` | 临时或单文件实验脚本 | 默认不跟踪 |
| `data/` | 本地数据集与中间数据 | 默认不跟踪 |
| `JupyterNotebooks/` | 外部导入的多框架原始 notebook 集合 | 保留本地源材料，不整体发布 |
| `.obsidian/` | 本工作区的 Obsidian 配置 | 不随意移动或改名 |

## PyTorch 学习材料

可发布的 PyTorch 套件位于 `LearningNotes/d2l-pytorch-notebooks/`。它从本地多框架原始集合中单独整理，MXNet、Paddle 和 TensorFlow 版本仍只保留在 `JupyterNotebooks/`，没有纳入本次 Git 跟踪。

整理时保留 notebook、配套图片、参考文献和安装脚本，排除了明确的运行生成物：模型参数、提交结果 CSV 和序列化演示输出。原始文件仍保留在本地导入目录中。

## 工作约定

- 稳定课程材料和长期笔记放进 `LearningNotes/`。
- 个人跟学 notebook 放进 `MyHandwrittenNotebookwithLIMU/`。
- 研究路线、选题与实验设计放进 `ResearchPlanning/`。
- 一次性试验放进 `ScratchNotebooks/` 或 `ScratchScripts/`，成熟后再迁入稳定目录。
- 模型权重、数据集、缓存、日志和 `.ipynb_checkpoints/` 不提交。
- 提交时按目标子目录暂存，避免把整个本地导入区一起加入 Git。
