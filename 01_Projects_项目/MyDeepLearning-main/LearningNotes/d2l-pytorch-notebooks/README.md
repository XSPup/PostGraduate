# D2L PyTorch Notebooks

这里是从本地多框架 notebook 集合中筛选出的 PyTorch 学习版本，作为 `MyDeepLearning-main` 的长期、可追踪学习材料。

## 内容

- 142 个章节与索引 notebook；
- notebook 所需的配套图片和图表；
- `d2l.bib`、`setup.py` 等辅助源文件。

建议从根目录的 `index.ipynb` 开始，再按 `chapter_*` 目录学习。运行环境和依赖可能随上游课程版本变化，首次运行前请先阅读 `chapter_installation/index.ipynb`。

## 筛选范围

本目录只包含 PyTorch 版本。MXNet、Paddle 和 TensorFlow 版本未复制到此处，也未纳入 Git。

以下明确的运行生成物没有进入本目录，但本地原始集合保持不变：

- 模型参数文件 `mlp.params`；
- 两个实验生成的 `submission.csv`；
- `my_mlp`、`mydict`、`x-file`、`x-files` 等序列化演示输出。

为减少仓库体积并避免发布机器相关的运行状态，已清除复制版本中的 cell 输出和执行计数；代码、Markdown 与 notebook 结构保持不变。后续运行 notebook 时，请不要提交模型权重、数据集、缓存、日志、cell 输出或 `.ipynb_checkpoints/`。
