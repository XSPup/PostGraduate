# 8 个月 ML 到 DL 学习计划

适用对象：已经完成机器学习入门实践，具备概率论和线性代数基础，希望在 8 个月内从传统机器学习系统过渡到深度学习、PyTorch、Transformer 和小型 AI 项目。

当前日期：2026-05-12。

## 0. 总体策略

这 8 个月不要把目标定成“学完所有 AI”。更合理的目标是：

1. 完整走完吴恩达机器学习主线。
2. 掌握 Python 数据处理和 sklearn 项目流程。
3. 完成 3-5 个能展示的 ML/DL 项目。
4. 从传统 ML 平稳过渡到神经网络和 PyTorch。
5. 在最后阶段建立 Transformer / LLM 的基本直觉，而不是一开始硬冲 Transformer。

你的主线应该是：

```text
吴恩达 ML -> sklearn 项目 -> 神经网络基础 -> PyTorch -> CNN/RNN/Attention -> Transformer 入门
```

每周建议投入：

```text
课程学习：40%
代码实操：40%
总结复盘：20%
```

如果一周只有 8-10 小时，也按这个比例执行，不要只看课。

## 1. 8 个月阶段总览

| 阶段 | 时间 | 主线目标 | 必做产出 |
|---|---:|---|---|
| Month 1 | 第 1-4 周 | 完成传统 ML 基础闭环 | notebooks + 一个 sklearn baseline |
| Month 2 | 第 5-8 周 | 系统掌握 sklearn、树模型、模型评估 | 一个完整 tabular 分类项目 |
| Month 3 | 第 9-12 周 | 无监督学习、特征工程、项目规范 | 一个聚类/降维项目 |
| Month 4 | 第 13-16 周 | 神经网络基础，从零实现 MLP | NumPy/PyTorch MLP 项目 |
| Month 5 | 第 17-20 周 | PyTorch 系统入门，训练循环 | 图像分类项目 |
| Month 6 | 第 21-24 周 | CNN、序列模型、实验管理 | CNN 项目 + 序列分类项目 |
| Month 7 | 第 25-28 周 | Attention、Transformer、NLP 入门 | 文本分类或小型 Transformer 实验 |
| Month 8 | 第 29-32 周 | 综合项目、部署、作品集整理 | 1 个 ML/DL capstone + GitHub 作品集 |

## 2. Month 1：完成传统 ML 基础闭环

目标：把你已经开始的线性回归、逻辑回归、KNN、正则化真正收束成体系。

主线课程：

- 吴恩达机器学习课程：线性回归、逻辑回归、正则化、分类基础。
- 辅助文档：scikit-learn User Guide 中的 model selection 和 metrics。

第 1 周：

- 复盘 `01_linear_regression.ipynb`。
- 确认能解释 MSE、RSS、梯度下降、学习率。
- 把线性回归核心函数不看答案重写一遍。

第 2 周：

- 复盘 `02_logistic_regression.ipynb`。
- 能解释 sigmoid、交叉熵、threshold、假阳性、假阴性。
- 做一次 threshold 对 precision/recall 的影响实验。

第 3 周：

- 复盘 `03_regularization_overfitting.ipynb`。
- 能解释 train/validation/test、欠拟合、过拟合、Ridge、Lasso。
- 改 `degree` 和 `alpha`，记录结果。

第 4 周：

- 完成 `04_sklearn_classification_baseline.ipynb` 和 `05_knn_iris_visual.ipynb`。
- 学会完整 sklearn 分类流程。
- 写一份 `2026-05-xx-ml-foundation-summary.md` 总结。

阶段产出：

- 5 个入门 notebook 跑通。
- 1 篇阶段总结。
- 能用自己的话讲清楚：回归、分类、KNN、正则化、过拟合。

## 3. Month 2：sklearn、树模型和完整分类项目

目标：从“理解算法”进入“完成一个小型机器学习项目”。

主线课程：

- 吴恩达机器学习课程：决策树、模型评估、误差分析。
- scikit-learn：DecisionTree、RandomForest、model_selection、metrics。

第 5 周：

- 学决策树。
- 理解信息增益、树深度、过拟合。
- notebook：`06_decision_tree_iris_or_cancer.ipynb`。

第 6 周：

- 学随机森林和集成学习直觉。
- 对比 Logistic Regression、KNN、Decision Tree、Random Forest。
- 学交叉验证 `cross_val_score`。

第 7 周：

- 做完整 tabular 分类项目。
- 推荐数据集：Breast Cancer、Titanic、Heart Disease 三选一。
- 必须包含：EDA、缺失值检查、train/test、baseline、模型对比、混淆矩阵、classification report。

第 8 周：

- 项目整理成 GitHub 风格。
- README 写清楚问题、数据、方法、结果、错误分析、下一步。
- 把 notebook 中稳定代码整理到 `src/` 或脚本中。

阶段产出：

- 一个完整分类项目。
- 一个模型对比表。
- 一份项目报告。

## 4. Month 3：数据处理、特征工程、无监督学习

目标：补齐真实项目中最常见的数据处理能力。

主线资源：

- Kaggle Learn：Python、Pandas、Data Visualization、Intro to ML。
- sklearn：preprocessing、Pipeline、PCA、KMeans。

第 9 周：

- Pandas 系统复习。
- 必会：读取 CSV、筛选、分组、缺失值、类别编码、合并、透视表。
- 小练习：对一个 CSV 做 EDA 报告。

第 10 周：

- 特征工程。
- 内容：标准化、归一化、one-hot、缺失值填充、Pipeline。
- notebook：`07_feature_engineering_pipeline.ipynb`。

第 11 周：

- 无监督学习。
- 学 KMeans、PCA。
- 项目：Iris 或 customer segmentation 聚类可视化。

第 12 周：

- 做一个无监督学习小项目。
- 推荐：客户分群、鸢尾花 PCA 可视化、图像颜色聚类。
- 重点不是分数，而是解释结果。

阶段产出：

- 一个 Pandas EDA notebook。
- 一个特征工程 pipeline notebook。
- 一个聚类/降维项目。

## 5. Month 4：神经网络基础，从 ML 过渡到 DL

目标：理解神经网络不是魔法，而是“线性变换 + 非线性激活 + 损失函数 + 梯度下降”的组合。

主线课程：

- 吴恩达 Deep Learning Specialization 第 1 门：Neural Networks and Deep Learning。
- 你已有的 `NNBasics`、`interactive-neuron-demo` 可作为补充。

第 13 周：

- 学神经元、激活函数、前向传播。
- 手写单层神经元做 AND/OR。
- 复盘 XOR 为什么线性模型做不好。

第 14 周：

- NumPy 手写两层 MLP。
- 任务：XOR 或简单二分类数据。
- 必须写出 forward、loss、backward、update。

第 15 周：

- 学反向传播直觉。
- 不要求一次推完所有矩阵公式，但要知道梯度从 loss 往前传。
- notebook：`08_numpy_mlp_from_scratch.ipynb`。

第 16 周：

- 把 NumPy MLP 和 sklearn MLP/PyTorch MLP 做对比。
- 写一份总结：传统 ML 到神经网络的差别。

阶段产出：

- NumPy MLP notebook。
- XOR 或 toy classification 可视化。
- 一篇“我如何理解神经网络”的学习报告。

## 6. Month 5：PyTorch 系统入门

目标：从手写 NumPy 过渡到 PyTorch，掌握深度学习工程基本写法。

主线资源：

- PyTorch 官方 Learn the Basics。
- DeepLearning.AI Deep Learning Specialization 第 2 门可并行看。

第 17 周：

- 学 Tensor、Dataset、DataLoader。
- 搞清楚 NumPy array 和 torch Tensor 的区别。
- notebook：`09_pytorch_tensor_dataloader.ipynb`。

第 18 周：

- 学 autograd。
- 能解释 `.backward()` 在做什么。
- 把线性回归用 PyTorch 重写。

第 19 周：

- 写完整训练循环。
- 内容：model、loss、optimizer、train loop、eval loop。
- 项目：Fashion-MNIST MLP 分类。

第 20 周：

- 加入 checkpoint、训练曲线、混淆矩阵。
- 学会保存和加载模型。

阶段产出：

- PyTorch 线性回归。
- PyTorch MLP 分类项目。
- 一个可复现训练脚本。

## 7. Month 6：CNN、序列模型和实验管理

目标：进入典型深度学习任务，开始接触图像和序列。

主线课程：

- Deep Learning Specialization：CNN 部分。
- PyTorch Tutorials：Computer Vision / Transfer Learning。

第 21 周：

- 学 CNN 基础：卷积、池化、通道、padding、stride。
- 项目：CIFAR-10 或 Fashion-MNIST CNN。

第 22 周：

- 做 CNN 实验对比。
- 对比 MLP vs CNN。
- 加入数据增强。

第 23 周：

- 学序列模型基础：RNN/LSTM/GRU 的直觉。
- 项目：简单文本分类或时间序列预测。

第 24 周：

- 做实验管理。
- 至少记录：参数、loss 曲线、accuracy、错误样本、改进计划。

阶段产出：

- 一个 CNN 图像分类项目。
- 一个序列建模入门项目。
- 一份实验记录模板。

## 8. Month 7：Attention、Transformer、NLP 入门

目标：正式接触 Transformer，但只要求建立结构直觉和小规模实践。

主线资源：

- Hugging Face Course。
- Stanford CS224N 作为进阶参考，不作为当前主线硬啃。

第 25 周：

- 学词向量、tokenization、embedding。
- 项目：中文/英文文本分类 baseline。

第 26 周：

- 学 attention 直觉。
- 能解释 query、key、value 的作用。
- 做一个小 attention 可视化实验。

第 27 周：

- 学 Transformer encoder 基本结构。
- 使用 Hugging Face fine-tune 一个文本分类模型。

第 28 周：

- 做一个小型 NLP 项目。
- 推荐：论文摘要分类、情感分类、新闻分类。
- 输出：模型、评估、错误分析、README。

阶段产出：

- 文本分类项目。
- 一个 Hugging Face fine-tuning notebook。
- 一篇 Transformer 入门总结。

## 9. Month 8：综合项目与作品集

目标：把 7 个月的学习转成能展示的作品，而不是只留下零散 notebook。

第 29 周：

- 选择 capstone 项目。
- 推荐方向：
  - 通信 AI：调制识别、网络流量预测、基站 KPI 异常检测。
  - 通用 ML：Titanic/Heart Disease 完整建模。
  - DL：图像分类或文本分类。
  - LLM 应用：通信论文 RAG 问答原型。

第 30 周：

- 项目实现。
- 必须包含：数据说明、baseline、模型对比、评估指标。

第 31 周：

- 项目工程化。
- 加 README、requirements、运行命令、结果图、错误分析。

第 32 周：

- 作品集整理。
- 把最好的 3-5 个项目整理成 GitHub 仓库或本地索引。
- 写 8 个月总结：学了什么、会做什么、下一阶段学什么。

阶段产出：

- 一个 capstone 项目。
- 一个项目索引 README。
- 一份 8 个月学习总结。

## 10. 项目清单

按优先级完成，不要求全部做完。

### 必做项目

1. Iris KNN 可视化多分类。
2. sklearn 完整分类 baseline。
3. Breast Cancer / Heart Disease 分类项目。
4. Pandas EDA + 特征工程 pipeline。
5. NumPy MLP from scratch。
6. PyTorch MLP 分类。
7. CNN 图像分类。
8. Hugging Face 文本分类。

### 选做项目

1. KMeans 客户分群。
2. PCA 可视化。
3. 网络流量预测。
4. RadioML 调制识别。
5. 通信论文 RAG 问答。

## 11. 每周固定工作流

每周按这个节奏执行：

1. 看课：2-4 小时。
2. 写 notebook：3-6 小时。
3. 改参数和画图：1-2 小时。
4. 写学习报告：30-60 分钟。
5. 周末复盘：整理 README 或项目索引。

每个 notebook 最后必须回答：

1. 这个任务是什么？
2. 输入是什么，输出是什么？
3. 用了什么模型？
4. 用了什么指标？
5. 出错最多的地方是什么？
6. 下一步怎么改？

## 12. 学习资源

主线资源：

- DeepLearning.AI Machine Learning Specialization：https://www.deeplearning.ai/specializations/machine-learning
- DeepLearning.AI Deep Learning Specialization：https://www.deeplearning.ai/specializations/deep-learning/
- PyTorch Learn the Basics：https://docs.pytorch.org/tutorials/beginner/basics/index.html
- scikit-learn Model Selection and Evaluation：https://scikit-learn.org/stable/model_selection
- Kaggle Learn Python：https://www.kaggle.com/learn/python
- Hugging Face Course：https://huggingface.co/course

进阶参考：

- fast.ai Practical Deep Learning：https://course.fast.ai/
- Stanford CS224N：https://web.stanford.edu/class/cs224n/

## 13. 阶段判断标准

### 2 个月后

你应该能独立完成一个 sklearn 分类项目，并能解释指标。

### 4 个月后

你应该能手写一个简单 MLP，并理解反向传播的基本流程。

### 6 个月后

你应该能用 PyTorch 完成一个图像分类项目，并保存训练结果。

### 8 个月后

你应该能完成一个端到端 ML/DL 项目，写清楚数据、模型、评估、错误分析和下一步改进。

## 14. 重要提醒

不要因为 Transformer 和 LLM 很火就跳过传统 ML。Transformer 的训练、评估和调试仍然离不开你现在学的这些东西：

```text
数据处理、特征理解、损失函数、梯度下降、正则化、过拟合、验证集、评估指标、错误分析
```

你现在应该做的是把吴恩达课程完整学完，然后逐步进入 DL。这样 8 个月后你不会只是会调用模型，而是能真正理解一个 AI 项目从数据到模型再到评估的完整过程。
