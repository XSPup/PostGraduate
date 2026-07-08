# 项目学习报告：Regularization and Overfitting Notebook

## 1. 基本信息

- 日期：2026-05-11
- 项目名称：Andrew Ng Machine Learning Notebooks - 03 Regularization and Overfitting
- GitHub 链接：暂无，当前为本地课程同步实操
- 本地路径：D:\MyDeepLearning\andrew-ng-ml-notebooks
- 主要 notebook：D:\MyDeepLearning\andrew-ng-ml-notebooks\03_regularization_overfitting.ipynb
- 数据来源：sklearn `load_diabetes()` 内置数据集
- 项目类型：吴恩达机器学习基础同步实操 / 回归 / 欠拟合 / 过拟合 / 正则化
- 难度：中等，比前两个 notebook 更偏模型诊断和实验分析

## 2. 我为什么学这个项目

前两个 notebook 分别完成了线性回归和逻辑回归。今天进入第三个 notebook，重点不是再学习一个新的预测模型，而是理解模型为什么会表现不好，以及如何判断是欠拟合还是过拟合。

昨天逻辑回归实验中已经出现了一个重要现象：继续增加 epoch 后准确率不再明显提升。这说明模型效果不只取决于训练次数，还和模型表达能力、数据质量、特征处理、正则化和泛化能力有关。

本次实验就是为了系统理解这些概念。

## 3. 本次学习目标

- 理解正则化的直觉：预测错误 + 模型复杂度惩罚。
- 理解 `lambda` / `alpha` 是正则化强度。
- 理解 L2/Ridge 和 L1/Lasso 的区别。
- 理解 sklearn `load_diabetes()` 数据集和昨天 Pima 糖尿病二分类数据集的区别。
- 理解 `degree` 代表多项式模型的复杂度。
- 学会比较训练集、验证集、测试集 MSE。
- 观察高阶多项式模型如何过拟合。
- 观察 Ridge/Lasso 如何改善验证集表现。
- 总结“训练集很好、验证集很差”时的排查和改进办法。

## 4. 我实际做了什么

### 4.1 先理解正则化

一开始看到吴恩达课程中逻辑回归正则化公式时比较混乱，尤其是：

```text
lambda
||w||^2
L1
L2
omit b
```

后续将其简化理解为：

```text
正则化 = 在训练时惩罚复杂模型，防止模型为了讨好训练集而学得太极端。
```

原来的训练目标：

```text
最小化预测错误
```

加入正则化后：

```text
最小化预测错误 + 模型复杂度惩罚
```

其中：

- `lambda` 或 sklearn 中的 `alpha` 表示惩罚力度。
- L2/Ridge 会惩罚参数平方和，让权重整体变小，模型更平滑。
- L1/Lasso 会惩罚参数绝对值和，可能让部分权重变成 0。

### 4.2 区分两个 diabetes 数据集

今天明确了一个容易混淆的问题：

昨天第二个 notebook 使用的是：

```text
Pima Indians Diabetes Dataset
```

目标值是：

```text
Outcome = 0 或 1
```

所以昨天是分类任务。

今天第三个 notebook 使用的是：

```python
load_diabetes()
```

目标值是连续数值，表示一年后的疾病进展指标，所以今天是回归任务。

当前实验为了方便画图，只取了 `bmi` 一个特征：

```text
bmi -> disease progression target
```

这里的 `bmi` 是已经预处理/标准化后的相对值，因此有正有负。正数可以理解为高于平均水平，负数可以理解为低于平均水平。

### 4.3 划分训练集、验证集、测试集

本次填写参数：

```python
TEST_SIZE = 0.2
VAL_SIZE = 0.25
```

含义：

- 20% 数据作为测试集。
- 剩下 80% 中再拿 25% 作为验证集。
- 最终大约是 60% train、20% validation、20% test。

今天进一步理解了三者区别：

```text
训练集：用来训练模型。
验证集：用来选择模型复杂度和参数。
测试集：最后只用一次，用来估计最终效果。
```

### 4.4 理解 degree

`degree` 表示多项式阶数，也就是模型允许曲线弯曲到什么程度。

例如：

```text
degree = 1: y = w1*x + b，一条直线
degree = 2: y = w1*x + w2*x^2 + b，抛物线
degree = 15: 使用 x 到 x^15，曲线可以非常复杂
```

今天观察到 `degree=15` 的曲线直接“起飞”，把其他 degree 的曲线压得看不见。这是一个很直观的过拟合现象。

## 5. 项目结构

当前相关结构如下：

```text
D:\MyDeepLearning
  andrew-ng-ml-notebooks
    01_linear_regression.ipynb
    02_logistic_regression.ipynb
    03_regularization_overfitting.ipynb
    04_sklearn_classification_baseline.ipynb

  learning-records
    reports
      2026-05-11-regularization-overfitting-notebook.md
```

## 6. 核心代码理解

### 6.1 读取数据

```python
diabetes = load_diabetes()
X_all = diabetes.data
y = diabetes.target
feature_names = diabetes.feature_names
```

理解：

`X_all` 包含全部特征，`y` 是连续目标值。

### 6.2 只取 bmi 特征

```python
bmi_index = feature_names.index("bmi")
X = X_all[:, [bmi_index]]
```

理解：

这里只取 `bmi` 一个特征，是为了能画二维图，更直观看到曲线如何欠拟合或过拟合。

### 6.3 多项式模型

```python
PolynomialFeatures(degree=degree, include_bias=False)
```

理解：

这个步骤会把一个输入 `x` 扩展成：

```text
x, x^2, x^3, ..., x^degree
```

degree 越高，模型越复杂，越容易贴合训练集，也越容易过拟合。

### 6.4 Ridge 和 Lasso

```python
Ridge(alpha=RIDGE_ALPHA)
Lasso(alpha=LASSO_ALPHA, max_iter=100000)
```

理解：

- Ridge 是 L2 正则化，惩罚权重平方和。
- Lasso 是 L1 正则化，惩罚权重绝对值和。
- `alpha` 表示正则化强度。

当前参数：

```python
HIGH_DEGREE = 15
RIDGE_ALPHA = 1.0
LASSO_ALPHA = 0.05
```

## 7. 遇到的问题和解决办法

### 问题 1：不理解正则化公式

- 原因：公式中同时出现了 `lambda`、L1、L2、范数和损失函数。
- 解决办法：先用文字理解为“预测错误 + 模型复杂度惩罚”。
- 学到什么：不用一开始强行吃透所有符号，先抓住概念直觉。

### 问题 2：误以为 diabetes 数据集都是 0/1 标签

- 原因：昨天的 Pima diabetes 数据集是二分类，今天的 sklearn diabetes 数据集是回归。
- 解决办法：明确今天的目标值是连续的疾病进展指标。
- 学到什么：同名数据集也可能任务不同，要先看目标值 `y` 是连续值还是类别。

### 问题 3：bmi 为什么有正有负

- 原因：sklearn 的 diabetes 特征已经被预处理/标准化。
- 解决办法：将其理解为相对于平均水平的偏高或偏低。
- 学到什么：机器学习数据中的特征值不一定是原始现实单位。

### 问题 4：degree=15 曲线起飞

- 原因：高阶多项式模型过于复杂，容易在边缘或局部剧烈波动。
- 解决办法：可以临时去掉 degree=15，或限制 y 轴范围观察其他曲线。但保留这个现象有助于理解过拟合。
- 学到什么：模型复杂度太高不一定更好，可能只是更会贴合噪声。

### 问题 5：如何回答“训练集很好、验证集很差”

当前总结为：

```text
训练好、验证差通常是过拟合。
优先尝试：检查数据泄漏、降低模型复杂度、加正则化、增加数据、清理数据、减少噪声特征、交叉验证。
神经网络中还可以使用 dropout、weight decay 和 early stopping。
```

## 8. 运行结果

### 8.1 线性 baseline

```text
Linear Regression
  train MSE=4012.69, RMSE=63.35, R2=0.355
  val   MSE=3430.10, RMSE=58.57, R2=0.374
  test  MSE=4074.72, RMSE=63.83, R2=0.231
```

### 8.2 不同 degree 的多项式模型

```text
Polynomial degree=1
  train MSE=4012.69
  val   MSE=3430.10
  test  MSE=4074.72

Polynomial degree=3
  train MSE=3982.23
  val   MSE=3540.81
  test  MSE=4100.57

Polynomial degree=8
  train MSE=3846.40
  val   MSE=3825.51
  test  MSE=11331.14

Polynomial degree=15
  train MSE=3739.88
  val   MSE=110626.82
  test  MSE=27193710.93
```

观察：

- degree 从 1 增加到 15，训练集 MSE 逐步下降。
- 但验证集和测试集并没有同步变好。
- degree=15 出现严重过拟合。

### 8.3 正则化对比

```text
High degree no regularization
  train MSE=3739.88
  val   MSE=110626.82
  test  MSE=27193710.93

Ridge regularization
  train MSE=3956.67
  val   MSE=3528.75
  test  MSE=6148.86

Lasso regularization
  train MSE=3946.85
  val   MSE=3513.27
  test  MSE=7424.65
```

结论：

- 无正则化高阶模型严重过拟合。
- Ridge 和 Lasso 都显著降低了验证集 MSE。
- 正则化略微牺牲训练集 MSE，但明显改善泛化能力。

## 9. 当前理解

今天对过拟合和正则化的理解如下：

模型太简单时，例如 degree 很低，训练误差和验证误差都可能较高，这更像欠拟合。

模型太复杂时，例如 degree=15，训练误差可能下降，但验证误差和测试误差可能大幅上升，这就是过拟合。

不能只看训练集 MSE，因为训练集是模型见过的数据。真正要判断模型是否能泛化，需要看验证集和测试集表现。

正则化的作用是限制模型复杂度。它会让训练集误差略微变高，但可能显著降低验证集误差，使模型更稳定。

## 10. 下一步计划

- 回到 notebook 中再观察 `Train vs Validation Error` 和 `Regularization Comparison` 两张图。
- 尝试修改 `HIGH_DEGREE`、`RIDGE_ALPHA`、`LASSO_ALPHA`，观察正则化强度变化。
- 学习交叉验证，理解为什么单次 train/val/test 切分有偶然性。
- 后续进入 sklearn classification baseline，开始把前面学到的训练、验证、评估流程用到完整分类项目中。
