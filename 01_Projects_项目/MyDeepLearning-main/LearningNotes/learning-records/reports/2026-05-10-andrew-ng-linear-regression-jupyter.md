# 项目学习报告：Andrew Ng ML - Linear Regression Jupyter Notebook

## 1. 基本信息

- 日期：2026-05-10
- 项目名称：Andrew Ng Machine Learning Notebooks - 01 Linear Regression
- GitHub 链接：暂无，当前为本地课程同步实操
- 本地路径：D:\MyDeepLearning\andrew-ng-ml-notebooks
- 主要 notebook：D:\MyDeepLearning\andrew-ng-ml-notebooks\01_linear_regression.ipynb
- 数据路径：D:\MyDeepLearning\simple-linear-regression-notebook\data.csv
- 项目类型：吴恩达机器学习基础同步实操 / Jupyter Notebook / 线性回归
- 难度：入门，但涉及机器学习核心基础概念

## 2. 我为什么学这个项目

我现在正在学习吴恩达机器学习基础课程。为了避免只看视频、不动手，我开始用 Jupyter Notebook 同步复现课程中的核心概念。

本次实验选择线性回归作为第一个 notebook，是因为线性回归包含机器学习中最基础的一组概念：模型假设、损失函数、梯度下降、学习率、参数更新和模型评估。这些概念后面会反复出现在逻辑回归、神经网络和深度学习中。

## 3. 本次学习目标

- 安装并启动 JupyterLab。
- 理解 Jupyter Notebook 和 VS Code 的区别。
- 创建吴恩达机器学习课程同步实操目录。
- 读取已有 CSV 数据，并用更清晰的变量名表示数据含义。
- 手写线性回归的预测函数、MSE 和梯度下降。
- 观察不同学习率对 loss 曲线的影响。
- 对比手写梯度下降和 sklearn `LinearRegression` 的结果。
- 写出本次实验总结，明确当前理解和下一步计划。

## 4. 我实际做了什么

### 4.1 完成 JupyterLab 环境检查

在命令行中检查 Jupyter 安装情况：

```cmd
jupyter --version
```

输出中包含：

```text
jupyterlab       : 4.5.7
notebook         : 7.5.6
ipykernel        : 7.2.0
```

说明 JupyterLab、Notebook 和 Python kernel 已经安装成功。

随后运行：

```cmd
jupyter lab
```

Jupyter Server 正常启动，并可以在浏览器中打开 notebook。

### 4.2 理解 Jupyter Notebook 和 VS Code 的区别

本次明确了：

- Jupyter Notebook 适合学习、实验、可视化、公式记录和边运行边观察。
- VS Code 更适合写正式项目、脚本、工程代码、README 和多文件结构。

当前阶段应该以 Jupyter 为主，等代码稳定后再逐步整理成 `.py` 脚本。

### 4.3 创建课程实操目录

创建了本地课程同步目录：

```text
D:\MyDeepLearning\andrew-ng-ml-notebooks
```

目录中包含：

```text
01_linear_regression.ipynb
02_logistic_regression.ipynb
03_regularization_overfitting.ipynb
04_sklearn_classification_baseline.ipynb
README.md
requirements.txt
.gitignore
```

其中 `01_linear_regression.ipynb` 已经完成第一轮学习，后面三个 notebook 作为后续课程模板。

### 4.4 重写数据读取代码

最初 notebook 中使用了候选路径和自动生成数据的写法。因为我明确知道数据文件就在：

```text
D:\MyDeepLearning\simple-linear-regression-notebook\data.csv
```

所以将数据读取逻辑改成更直接的版本：

```python
from pathlib import Path
import numpy as np

data_path = Path(r"D:\MyDeepLearning\simple-linear-regression-notebook\data.csv")

points = np.genfromtxt(data_path, delimiter=",")

hours_studied = points[:, 0]
test_scores = points[:, 1]

print("Data path:", data_path)
print("Data shape:", points.shape)
points[:5]
```

这样变量含义更清晰：

- `hours_studied`：学习时长
- `test_scores`：考试分数

### 4.5 手写线性回归核心函数

本次实验中理解并使用了以下函数：

```python
def predict(x, w, b):
    return w * x + b
```

该函数表示线性回归假设函数：

```text
y_hat = w * x + b
```

其中：

- `w` 是斜率，也可以理解为权重。
- `b` 是截距，也可以理解为偏置。

损失函数使用 MSE：

```python
def compute_cost(x, y, w, b):
    y_hat = predict(x, w, b)
    return np.mean((y_hat - y) ** 2)
```

我的理解：

- 残差是 `预测值 - 真实值`。
- 残差平方和 RSS 是所有残差平方加起来。
- MSE 是 RSS 除以样本数量。
- 吴恩达课程中常见的代价函数 `J(w,b)=RSS/(2m)` 和 MSE 本质上衡量同一件事，只是多除了一个 `2`，方便求导。

### 4.6 理解 MSE 和交叉熵的适用场景

本次明确了：

```text
回归任务常用 MSE、MAE、Huber Loss。
分类任务常用交叉熵。
```

线性回归预测的是连续值，例如考试分数，所以本次使用 MSE。

交叉熵更适合分类任务，因为分类模型通常输出类别概率，例如：

```text
P(y=1|x)=0.83
```

因此当前实验不用交叉熵，而是用 MSE。

### 4.7 观察学习率对训练的影响

实验中比较了不同学习率：

```python
learning_rates = [0.00001, 0.0001, 0.001]
```

观察结果：

- `lr=0.00001`：下降较慢。
- `lr=0.0001`：相对稳定。
- `lr=0.001`：MSE 迅速变得极大，说明梯度下降发散。

我的理解：

```text
学习率不是越大越好。
学习率太小，收敛慢。
学习率合适，稳定下降。
学习率太大，会越过最优点并发散。
```

使用对数坐标可以更清楚地观察数值差距很大的 loss 曲线：

```python
plt.yscale("log")
```

### 4.8 对比手写梯度下降和 sklearn

使用 sklearn 训练线性回归：

```python
model = LinearRegression()
model.fit(hours_studied.reshape(-1, 1), test_scores)

sklearn_pred = model.predict(hours_studied.reshape(-1, 1))
sklearn_mse = mean_squared_error(test_scores, sklearn_pred)
```

其中：

- `reshape(-1, 1)` 是因为 sklearn 要求输入特征是二维数组。
- `model.coef_[0]` 是 sklearn 学到的斜率。
- `model.intercept_` 是 sklearn 学到的截距。

本次实验得到的结果为：

```text
manual w:  1.4766, manual b:  0.1484, manual mse:  112.5795
sklearn w: 1.3224, sklearn b: 7.9910, sklearn mse: 110.2574
```

我的理解：

手写梯度下降和 sklearn 的结果不完全一样。主要原因是 sklearn 的 `LinearRegression` 通常直接求最小 MSE 的最优解，而手写梯度下降是一步一步迭代逼近最优解。

当前手写结果中：

```text
manual b = 0.1484
sklearn b = 7.9910
```

截距差距明显，说明手写梯度下降还没有完全收敛。由于原始数据中 `x` 的尺度会影响梯度下降速度，`w` 和 `b` 的学习速度可能不一致。后续可以通过增加 epoch 或特征缩放改进。

## 5. 项目结构

当前相关结构如下：

```text
D:\MyDeepLearning
  andrew-ng-ml-notebooks
    README.md
    requirements.txt
    .gitignore
    01_linear_regression.ipynb
    02_logistic_regression.ipynb
    03_regularization_overfitting.ipynb
    04_sklearn_classification_baseline.ipynb

  simple-linear-regression-notebook
    data.csv
    linear-regression-demo.ipynb
    README.md

  learning-records
    reports
      2026-05-10-andrew-ng-linear-regression-jupyter.md
```

## 6. 核心代码理解

### 6.1 读取数据

```python
data_path = Path(r"D:\MyDeepLearning\simple-linear-regression-notebook\data.csv")
points = np.genfromtxt(data_path, delimiter=",")
hours_studied = points[:, 0]
test_scores = points[:, 1]
```

理解：

`points` 是一个二维数组。第一列是输入特征，第二列是目标值。

### 6.2 预测函数

```python
def predict(x, w, b):
    return w * x + b
```

理解：

这就是一元线性回归的模型假设函数。

### 6.3 MSE

```python
def compute_cost(x, y, w, b):
    y_hat = predict(x, w, b)
    return np.mean((y_hat - y) ** 2)
```

理解：

MSE 越低，说明在同一数据集上预测值越接近真实值。但训练集 MSE 低不代表模型泛化一定好，后面还要学习 train/validation/test。

### 6.4 梯度下降

```python
def compute_gradients(x, y, w, b):
    m = len(x)
    y_hat = predict(x, w, b)
    error = y_hat - y

    dw = (2 / m) * np.sum(error * x)
    db = (2 / m) * np.sum(error)
    return dw, db
```

理解：

`dw` 和 `db` 表示当前损失函数分别对 `w` 和 `b` 的偏导数。梯度下降通过下面的方式更新参数：

```python
w = w - learning_rate * dw
b = b - learning_rate * db
```

### 6.5 sklearn 对比

```python
model = LinearRegression()
model.fit(hours_studied.reshape(-1, 1), test_scores)
```

理解：

sklearn 是机器学习算法工具库。手写实现用于理解原理，sklearn 用于实际项目中快速、稳定地训练模型。

## 7. 遇到的问题和解决办法

### 问题 1：Jupyter Markdown 双击后不渲染

- 原因：双击 Markdown cell 后进入编辑状态，显示原始 Markdown 和 LaTeX。
- 解决办法：按 `Shift + Enter` 运行该 cell，或者按 `Esc` 退出编辑状态。
- 学到什么：Jupyter cell 有编辑状态和渲染状态。

### 问题 2：画图时变量名混用

错误倾向：

```python
y_pred = predict(hours_studied, w, b)
plt.scatter(x, y)
```

- 原因：前面已经把 `x` 和 `y` 改名为 `hours_studied` 和 `test_scores`，后面画图时仍然使用旧变量名。
- 解决办法：统一变量名：

```python
plt.scatter(hours_studied, test_scores, alpha=0.8, label="data")
plt.plot(hours_studied, y_pred, color="red", label="manual gradient descent")
```

- 学到什么：变量名一旦改得更语义化，后续代码也要同步改。

### 问题 3：学习率曲线只能看到一条线

- 原因：不同学习率的 MSE 数值差距过大，尤其 `lr=0.001` 发散后把其他曲线压在底部。
- 解决办法：使用对数坐标：

```python
plt.yscale("log")
```

- 学到什么：机器学习中观察 loss 曲线时，对数坐标很有用。

### 问题 4：手写梯度下降和 sklearn 拟合线不重合

- 原因：手写梯度下降没有完全收敛，尤其是截距 `b` 和 sklearn 差距明显。
- 解决办法：可以增加 epoch，或者后续学习特征缩放。
- 学到什么：sklearn 的 `LinearRegression` 通常直接求最优解，而手写梯度下降依赖学习率、epoch 和数据尺度。

## 8. 运行结果

当前已完成：

- JupyterLab 启动成功。
- NumPy、Pandas、Matplotlib、sklearn 均可在 notebook 中导入。
- 成功读取固定路径 CSV 数据。
- 成功画出数据散点图。
- 成功手写预测函数、MSE、梯度下降。
- 成功画出训练 loss 曲线。
- 成功观察学习率过大导致发散。
- 成功用 sklearn 训练线性回归，并与手写结果对比。
- 完成 notebook 最后一节总结。

本次关键输出：

```text
manual w:  1.4766, manual b:  0.1484, manual mse:  112.5795
sklearn w: 1.3224, sklearn b: 7.9910, sklearn mse: 110.2574
```

结论：

```text
MSE 越低说明当前数据集上误差越小。
sklearn MSE 更低，说明它找到了更优的线性回归解。
手写梯度下降结果还可以继续优化。
```

## 9. 当前理解

目前我对线性回归的理解如下：

线性回归用一条直线拟合数据：

```text
y_hat = w * x + b
```

模型训练的目标是找到合适的 `w` 和 `b`，让预测值 `y_hat` 尽量接近真实值 `y`。

MSE 是均方误差，英文全称是 `Mean Squared Error`。它的含义是平均每个样本的残差平方。MSE 和残差平方和 RSS 的关系是：

```text
MSE = RSS / 样本数量
```

对于当前回归任务，使用 MSE 是合理的。交叉熵更适合分类任务，因为分类任务通常输出概率。

梯度下降不是一次性得到最优答案，而是通过不断更新参数逐步靠近最优点。学习率决定每一步走多大。学习率过小会慢，学习率过大会发散。

sklearn 是传统机器学习中常用的 Python 工具库。手写实现帮助理解原理，sklearn 帮助实际项目中高效建模。

## 10. 下一步计划

- 用 `Kernel -> Restart Kernel and Run All Cells` 检查 `01_linear_regression.ipynb` 是否能从头完整运行。
- 保存并整理第一个 notebook，确认总结区已补全。
- 尝试加入特征缩放实验，观察手写梯度下降是否更接近 sklearn。
- 开始 `02_logistic_regression.ipynb`，学习 sigmoid、二分类、交叉熵和决策边界。
- 继续跟随吴恩达机器学习基础课程，保持“看课 + notebook 实操 + 学习报告”的节奏。
