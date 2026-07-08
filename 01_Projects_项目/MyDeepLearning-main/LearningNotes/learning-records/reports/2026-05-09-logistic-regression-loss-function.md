# 项目学习报告：吴恩达课程 - Logistic Regression Loss Function

## 1. 基本信息

- 日期：2026-05-09
- 项目名称：吴恩达课程 - Logistic Regression Loss Function
- GitHub 链接：暂无，本次主要学习课程板书和公式推导
- 本地路径：D:\MyDeepLearning\learning-records\reports\2026-05-09-logistic-regression-loss-function.md
- 项目类型：机器学习理论学习 / 逻辑回归 / 损失函数 / 梯度下降
- 难度：入门到中等，主要难点在公式含义和符号理解

## 2. 我为什么学这个项目

我正在学习吴恩达课程中逻辑回归的下一小节。昨天已经通过一个小项目理解了线性回归中的 MSE、残差平方和、梯度下降和参数更新。今天的目标是把这个理解迁移到逻辑回归，弄清楚为什么分类任务不用 MSE，而使用交叉熵损失，以及逻辑回归中的梯度下降到底在最小化什么。

## 3. 本次学习目标

- 理解逻辑回归的单样本 loss function 为什么写成 `-[y log(y_hat) + (1-y) log(1-y_hat)]`。
- 区分信息熵、交叉熵、log loss、MSE、残差平方和这些概念。
- 理解逻辑回归中的 cost function `J(w,b)` 是所有样本 loss 的平均值。
- 理解梯度下降本身不是某一种损失函数，而是一种最小化 `J(w,b)` 的方法。
- 理解 `a`、`y_hat`、`z`、`w`、`b`、`dw`、`db` 在吴恩达板书里的含义。

## 4. 我实际做了什么

1. 复习了逻辑回归的预测公式：

```text
y_hat = σ(w^T x + b)
σ(z) = 1 / (1 + e^(-z))
```

其中 `y_hat` 是模型预测 `y=1` 的概率。

2. 对比了昨天线性回归项目中的损失函数：

```python
total_cost += (y - (m * x + b)) ** 2
return total_cost / N
```

明确了：

```text
total_cost = 残差平方和 RSS
total_cost / N = 均方误差 MSE
```

3. 学习了逻辑回归的单样本 loss：

```text
L(y_hat, y) = - [ y log(y_hat) + (1-y) log(1-y_hat) ]
```

4. 学习了逻辑回归的整体 cost function：

```text
J(w,b) = 1/m * Σ L(y_hat^(i), y^(i))
```

也就是：

```text
J(w,b) = -1/m * Σ [ y^(i)log(y_hat^(i)) + (1-y^(i))log(1-y_hat^(i)) ]
```

5. 讨论了如果损失函数有多个局部最低点，梯度下降可能只能找到局部最低点，不一定找到全局最低点。

6. 进一步理解了吴恩达课程中为什么逻辑回归不用平方误差，而使用交叉熵：逻辑回归加交叉熵的 cost function 更适合优化，通常是凸函数。

7. 学习了 m 个样本上的逻辑回归训练流程：

```text
z^(i) = w^T x^(i) + b
a^(i) = y_hat^(i) = σ(z^(i))
J(w,b) = 所有样本 loss 的平均值
dw_j = J 对 w_j 的偏导数
db = J 对 b 的偏导数
```

## 5. 项目结构

本次不是一个新的代码项目，而是一次课程理论学习。相关学习资产如下：

- `D:\MyDeepLearning\learning-records\reports\2026-05-09-logistic-regression-loss-function.md`：今天的阶段性学习报告。
- `D:\MyDeepLearning\learning-records\templates\project-report-template.md`：固定学习报告模板。
- `D:\MyDeepLearning\simple-linear-regression-notebook`：昨天用来对比 MSE 和梯度下降的线性回归项目。

## 6. 核心代码理解

今天主要是公式理解，没有新增代码。但把昨天代码和今天公式做了对照。

昨天线性回归项目的损失函数：

```python
def compute_cost(b, m, points):
    total_cost = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        total_cost += (y - (m * x + b)) ** 2
    return total_cost/N
```

我的理解：

```text
线性回归预测连续值，所以用预测值和真实值的距离来衡量错误。
这里的 loss/cost 是平均残差平方，也就是 MSE。
```

今天逻辑回归课程的损失函数：

```text
L(y_hat, y) = - [ y log(y_hat) + (1-y) log(1-y_hat) ]
```

我的理解：

```text
逻辑回归预测的是概率，所以 loss 衡量的是模型给真实类别的概率有多低。
如果模型非常自信地预测错了，loss 会非常大。
```

当 `y = 1` 时：

```text
L = -log(y_hat)
```

这时希望 `y_hat` 越接近 1 越好。

当 `y = 0` 时：

```text
L = -log(1-y_hat)
```

这时希望 `y_hat` 越接近 0 越好。

## 7. 遇到的问题和解决办法

### 问题 1：逻辑回归 loss 为什么看起来像信息熵？

- 原因：逻辑回归的 loss 本质上是二分类交叉熵，也属于 `- 概率 * log(概率)` 这一类形式。
- 解决办法：区分信息熵和交叉熵。
- 学到什么：

```text
信息熵：衡量一个分布自身的不确定性。
交叉熵：衡量用预测分布表示真实分布时有多糟。
逻辑回归 loss：二分类场景下的交叉熵 / log loss。
```

### 问题 2：逻辑回归的梯度下降是不是在使用平均信息熵？

- 原因：公式形式和信息熵相似，容易把交叉熵误认为信息熵。
- 解决办法：明确更准确的说法是“平均交叉熵损失”，不是“平均信息熵”。
- 学到什么：

```text
昨天线性回归：梯度下降最小化 MSE。
今天逻辑回归：梯度下降最小化平均交叉熵 / log loss。
梯度下降本身只是优化方法，不绑定某一种损失函数。
```

### 问题 3：如果函数有多个局部最低点怎么办？

- 原因：梯度下降只根据当前位置的斜率往下降方向走，如果目标函数坑坑洼洼，可能停在局部最低点。
- 解决办法：理解凸函数和非凸函数的区别。
- 学到什么：

```text
线性回归 + MSE：通常是凸函数。
逻辑回归 + 交叉熵：通常是凸函数。
逻辑回归 + 平方误差：可能非凸，不推荐。
深度神经网络：通常非凸，会有更复杂的优化问题。
```

### 问题 4：m 个样本那一页符号太多，看不懂

- 原因：吴恩达把单样本 loss、全体样本平均 cost、预测值 `a`、参数梯度 `dw/db` 放在同一页，符号密度很高。
- 解决办法：把它拆成四步：

```text
1. 对每个样本算 z。
2. 对每个样本算 a = y_hat。
3. 对所有样本的 loss 求平均，得到 J。
4. 求 J 对 w 和 b 的偏导，用于梯度下降更新参数。
```

- 学到什么：`a^(i)`、`y_hat^(i)`、`σ(z^(i))` 在这页里基本是同一个东西，都是第 `i` 个样本预测为 1 的概率。

## 8. 运行结果

今天没有运行新的代码，但完成了以下理论理解：

- 搞清楚昨天项目里的 LF 是 MSE，也就是平均残差平方。
- 搞清楚今天逻辑回归里的 LF 是交叉熵 / log loss。
- 搞清楚 cost function `J(w,b)` 是多个样本 loss 的平均值。
- 搞清楚梯度下降是在最小化 `J(w,b)`，而不是固定最小化某一种特定损失。
- 能初步解释为什么分类问题更适合交叉熵，而不是 MSE。

## 9. 当前理解

我现在的理解是：损失函数负责定义“错得有多严重”，梯度下降负责根据这个错误去调整参数。

昨天的线性回归预测的是连续数值，所以用：

```text
MSE = 平均 (真实值 - 预测值)^2
```

今天的逻辑回归预测的是二分类概率，所以用：

```text
Binary Cross Entropy = -平均 [ y log(y_hat) + (1-y)log(1-y_hat) ]
```

这两个 loss 都会变成一个整体 cost function `J`。训练模型的目标不是让某一个样本完美，而是让所有训练样本的平均损失尽量小。

## 10. 下一步计划

- 继续学习吴恩达后面关于逻辑回归梯度计算的推导，重点看 `dz = a - y` 是怎么来的。
- 找一个小型逻辑回归代码项目，把今天的公式对应到真实代码里。
- 尝试自己写一个极小版逻辑回归：只用 NumPy、几个二维点、二分类标签。
- 把昨天线性回归的 `compute_cost()` 和未来逻辑回归的 `compute_loss()` 放在一起对比。
- 如果今天后续继续学习，把新的内容追加到这份报告，而不是新建重复文件。

## 11. 今日后续学习补充：向量化与神经网络入门

### 11.1 向量化是什么

今天继续学习了吴恩达课程中的向量化。我的理解是：

```text
向量化 = 用矩阵和向量运算代替显式 for 循环。
```

它和 for 循环在数学理论上做的是同一件事，区别主要在：

```text
for 循环：一个样本一个样本算，直观但慢。
向量化：一批样本一起算，代码短，速度快。
```

逻辑回归中，原来可能写成：

```python
for i in range(m):
    z[i] = np.dot(w.T, X[:, i]) + b
    a[i] = sigmoid(z[i])
```

向量化后可以写成：

```python
Z = np.dot(w.T, X) + b
A = sigmoid(Z)
```

这不是换算法，而是换了一种更高效的计算表达方式。

### 11.2 np.dot 的作用

今天还学习了 `np.dot()`。我的理解是：它根据输入形状不同，可以表示向量点积、矩阵乘向量、矩阵乘矩阵。

最重要的机器学习用法是：

```python
z = np.dot(w.T, x) + b
```

它对应数学公式：

```text
z = w1*x1 + w2*x2 + ... + wn*xn + b
```

也就是把每个输入特征乘以对应权重，再全部加起来。

### 11.3 神经网络里的节点和连线

今天开始进入神经网络部分。最开始我不理解图里的节点和连线，后来整理成：

```text
节点：保存一个数值，通常叫 activation。
连线：保存一个权重 weight。
训练：不断调整连线上的权重。
```

一个神经元做两步：

```text
z = w1*x1 + w2*x2 + b
a = sigmoid(z)
```

如果有隐藏层，就是把多个这样的神经元堆起来：

```text
第一层：
Z[1] = W[1]X + b[1]
A[1] = activation(Z[1])

第二层：
Z[2] = W[2]A[1] + b[2]
A[2] = activation(Z[2])
```

这里 `A[2]` 就是最终预测值 `y_hat`。

### 11.4 NNBasics 项目尝试

为了理解神经网络，我下载并运行了一个很小的 GitHub 项目：

```text
https://github.com/ceasedfonts/NNBasics
```

本地路径：

```text
D:\MyDeepLearning\NNBasics
```

这个项目训练一个小神经网络学习 XOR：

```text
[0,0] -> 0
[0,1] -> 1
[1,0] -> 1
[1,1] -> 0
```

运行结果：

```text
Input: [0 0], Predicted Output: [0.03272906], Expected Output: [0]
Input: [0 1], Predicted Output: [0.93088766], Expected Output: [1]
Input: [1 0], Predicted Output: [0.93088767], Expected Output: [1]
Input: [1 1], Predicted Output: [0.09215892], Expected Output: [0]
```

我的理解：

```text
输出接近 0 就判断为 0。
输出接近 1 就判断为 1。
这个小网络已经基本学会了 XOR。
```

但是这个项目马上进入了前向传播、反向传播、隐藏层权重更新，对当前阶段来说有点难。

### 11.5 环境管理调整

今天还调整了项目环境管理方式：

```text
每个项目保留代码 + requirements.txt。
不用长期保留 .venv。
需要运行时再创建环境。
```

已经删除了以下虚拟环境：

```text
D:\MyDeepLearning\simple-linear-regression-notebook\.venv
D:\MyDeepLearning\NNBasics\.venv
```

并为项目保留了依赖清单：

```text
D:\MyDeepLearning\simple-linear-regression-notebook\requirements.txt
D:\MyDeepLearning\NNBasics\requirements.txt
```

以后重建环境的基本流程是：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 11.6 创建互动单神经元 Demo

因为 `NNBasics` 对当前阶段有点难，所以创建了一个更简单的本地互动页面：

```text
D:\MyDeepLearning\interactive-neuron-demo\index.html
```

它只讲一个神经元：

```text
z = w1*x1 + w2*x2 + b
a = sigmoid(z)
```

这个页面可以通过滑块调节：

```text
x1, x2, w1, w2, b
```

实时观察：

```text
输入节点怎么影响 z
连线权重怎么影响输出
偏置 b 怎么整体推动 z
sigmoid 怎么把 z 变成 0 到 1
为什么 a >= 0.5 时判断为 1
```

这个互动 demo 更适合作为理解神经网络节点和连线的第一步。

## 12. 更新后的下一步计划

- 先反复玩 `interactive-neuron-demo`，把一个神经元的输入、权重、偏置、sigmoid 输出理解清楚。
- 暂时不急着深入反向传播，先把前向传播完全理解。
- 回头再看 `NNBasics`，先只看 `forward()` 函数，不急着看 `train()`。
- 等一个神经元理解稳定后，再看两个隐藏神经元如何组合成一个小网络。
- 今天学习结束后，可以把这份报告再整理成最终版。
