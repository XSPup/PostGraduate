# 项目学习报告：D2L PyTorch - 线性回归从零实现

## 1. 基本信息

- 日期：2026-05-22
- 学习主题：Dive into Deep Learning（PyTorch 版）- 线性回归的从零实现
- GitHub 链接：本次为本地 notebook 学习
- 本地路径：E:\MyDeepLearning\MyHandwrittenNotebookwithLIMU\5LinearRegression.ipynb
- 参考内容：D2L PyTorch `chapter_linear-networks/linear-regression-scratch.ipynb`
- 项目类型：深度学习基础 / PyTorch / Jupyter Notebook / 手写训练循环
- 难度：中等偏难，第一次完整串起训练流程
- 当前状态：已跑通训练循环，loss 正常下降，已在 notebook 末尾补充总结

## 2. 为什么学这一节

这一节是从传统机器学习线性回归过渡到深度学习训练流程的重要节点。它不只是再次学习线性回归公式，而是用 PyTorch 手写一遍训练系统的核心部件：数据迭代器、模型、损失函数、自动求导、优化器和训练循环。

本节的价值在于：以后即使使用高级框架封装，仍然能知道训练背后发生了什么。模型不是神秘地“自己变好”，而是每个 batch 经过预测、计算损失、反向传播和参数更新之后逐步逼近真实规律。

## 3. 本次学习目标

- 理解人工合成线性数据集的生成方式。
- 理解 `features`、`labels`、`w`、`b` 的形状关系。
- 手写小批量数据迭代器 `data_iter`。
- 区分 `epoch`、`batch`、`batch_size`。
- 手写线性回归模型 `linreg`。
- 手写平方损失函数 `squared_loss`。
- 理解 `backward()`、`grad`、`zero_()` 的作用。
- 手写小批量随机梯度下降 `sgd`。
- 跑通完整训练循环，并观察 loss 下降和参数误差。

## 4. 实际做了什么

1. 导入 `random`、`torch` 和 `d2l`。
2. 定义 `synthetic_data(w, b, num_examples)`，生成带噪声的人工线性数据。
3. 检查 `features.shape == torch.Size([1000, 2])` 和 `labels.shape == torch.Size([1000, 1])`。
4. 绘制 `features[:, 1]` 和 `labels` 的散点图，观察人工数据的线性趋势。
5. 定义 `data_iter`，实现随机打乱索引并按小批量返回样本。
6. 通过打印第一个 batch，观察 `X` 和 `y` 的实际内容。
7. 初始化参数：

```python
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
```

8. 定义线性模型：

```python
def linreg(X, w, b):
    return X @ w + b
```

9. 定义平方损失函数：

```python
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
```

10. 定义小批量随机梯度下降：

```python
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
```

11. 设置训练超参数：

```python
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss
```

12. 跑通完整训练循环，并打印每个 epoch 后的平均 loss。
13. 打印 `w` 和 `b` 与真实参数之间的估计误差。
14. 在 notebook 末尾新增本节总结，记录重难点和常见错误。

## 5. 项目结构

```text
E:\MyDeepLearning
  MyHandwrittenNotebookwithLIMU
    5LinearRegression.ipynb

  ReferencesJupyter
    pytorch
      chapter_linear-networks
        linear-regression-scratch.ipynb

  LearningNotes
    learning-records
      reports
        2026-05-22-d2l-linear-regression-scratch.md
      PROJECT_INDEX.md
```

## 6. 核心代码理解

### 6.1 人工数据生成

```python
y = torch.matmul(X, w) + b
y += torch.normal(0, 0.01, y.shape)
return X, y.reshape((-1, 1))
```

理解：

- `X` 是输入特征矩阵。
- `w` 是真实权重。
- `b` 是真实偏置。
- 噪声模拟真实数据中的观测误差。
- `reshape((-1, 1))` 把标签变成列向量，方便后续和预测值对齐。

### 6.2 小批量数据迭代器

```python
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]
```

理解：

- 每次调用 `data_iter` 时，先把样本索引打乱一次。
- 一个 epoch 内按打乱后的顺序切 batch。
- 一个样本在同一个 epoch 内通常只会被使用一次。
- `yield` 每次产出一个 batch，而不是一次性返回全部数据。

### 6.3 线性模型

```python
def linreg(X, w, b):
    return X @ w + b
```

理解：

- `X @ w` 是矩阵乘法。
- `X` 的形状是 `[batch_size, 2]`。
- `w` 的形状是 `[2, 1]`。
- 输出形状是 `[batch_size, 1]`。
- `b` 通过广播加到每个样本的预测值上。

### 6.4 平方损失

```python
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
```

理解：

- `y_hat` 是预测值。
- `y` 是真实标签。
- `y.reshape(y_hat.shape)` 让真实标签形状和预测值一致。
- `** 2 / 2` 是平方损失，除以 2 是为了后续求导更简洁。

### 6.5 SGD

```python
param -= lr * param.grad / batch_size
param.grad.zero_()
```

理解：

- `param.grad` 是当前 batch 反向传播得到的梯度。
- `lr` 控制每次更新的步长。
- 除以 `batch_size` 是把 batch 内总梯度变成平均梯度。
- `param -= ...` 是原地更新参数，不能写成 `param = param - ...`。
- `param.grad.zero_()` 清空梯度，避免下一次 `backward()` 时梯度累加。

## 7. 遇到的问题和解决办法

### 问题 1：`detach().numpy()` 写法混乱

错误写法：

```python
labels, detach().numpy
```

或：

```python
labels.detach().numpy
```

原因：

- `detach()` 和 `numpy()` 都是张量对象的方法，需要用点连接。
- 方法后面如果要真正执行，必须加括号。

正确写法：

```python
labels.detach().numpy()
```

学到：

```text
点表示“属于谁”，括号表示“执行一下”，逗号表示“下一个参数”。
```

### 问题 2：散点图中 x 和 y 尺寸不一致

错误原因：

- `labels.detach().numpy` 少写了 `()`，导致传入的不是数组结果。

正确写法：

```python
d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
```

学到：

- 方法名本身和方法执行后的结果不是一回事。
- `numpy` 和 `numpy()` 差别很大。

### 问题 3：重启或跳过前面 cell 后变量不存在

报错：

```text
NameError: name 'features' is not defined
```

原因：

- Jupyter 当前 kernel 里还没有运行生成 `features` 和 `labels` 的 cell。

解决：

- 从导入、数据生成、函数定义开始按顺序重新运行。

学到：

- notebook 里的变量存在于当前 kernel 内存中。
- 代码写在上面不代表已经运行过。

### 问题 4：`data_iter` 索引多包了一层方括号

错误写法：

```python
yield features[[batch_indices]], labels[batch_indices]
```

警告：

```text
Using a non-tuple sequence for multidimensional indexing is deprecated
```

正确写法：

```python
yield features[batch_indices], labels[batch_indices]
```

学到：

- `batch_indices` 本身已经是一组索引，不需要再写成 `[batch_indices]`。

### 问题 5：平方损失函数里把 `shape` 当函数调用

错误写法：

```python
return (y_hat - y.shape(y_hat.shape)) ** 2 / 2
```

报错：

```text
TypeError: 'torch.Size' object is not callable
```

原因：

- `y.shape` 是查看形状，不是改变形状。
- `reshape(...)` 才是改变形状。

正确写法：

```python
return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
```

学到：

```text
shape 是属性，reshape 是方法。
```

### 问题 6：SGD 中把原地更新写成重新赋值

错误写法：

```python
param = param - lr * param.grad / batch_size
param.grad.zero_()
```

报错：

```text
AttributeError: 'NoneType' object has no attribute 'zero_'
```

原因：

- `param = param - ...` 创建了一个新张量，只是让局部变量 `param` 指向新对象，没有原地更新原来的 `w` 和 `b`。
- 新张量的 `.grad` 可能是 `None`。

正确写法：

```python
param -= lr * param.grad / batch_size
param.grad.zero_()
```

学到：

- 手写优化器时，参数更新需要改原来的参数对象。
- PyTorch 中 `zero_()` 末尾的 `_` 通常表示原地操作。

## 8. 运行结果

训练循环已经成功跑通，输出为：

```text
epoch 1,loss 0.027798
epoch 2,loss 0.000096
epoch 3,loss 0.000049
```

说明模型在训练过程中确实学到了更接近真实规律的参数，loss 明显下降。

最终参数误差：

```text
w的估计误差: tensor([ 2.4021e-04, -7.4625e-05], grad_fn=<SubBackward0>)
b的估计误差: tensor([0.0005], grad_fn=<RsubBackward1>)
```

说明学到的 `w` 和 `b` 已经非常接近生成数据时使用的真实参数。

## 9. 当前理解

目前对本节的理解可以总结为：

线性回归从零实现不是只写一个公式，而是完整搭建一个最小训练系统。这个系统的流程是：

```text
生成/读取数据
-> 随机取 batch
-> 用模型预测
-> 计算损失
-> backward 自动求梯度
-> SGD 更新参数
-> 清空梯度
-> 重复多个 epoch
```

这一节虽然模型很简单，但它已经包含了深度学习训练的基本骨架。后面学习神经网络时，模型会更复杂，损失函数和优化器可能会换成框架封装，但核心思路仍然是预测、算损失、求梯度、更新参数。

当前掌握情况：

- 能解释 `features`、`labels`、`w`、`b` 的形状关系。
- 能理解 `epoch` 和 `batch` 的区别。
- 能看懂训练循环每一行大概在做什么。
- 对 `grad`、`backward()`、`zero_()` 的关系有了第一层理解。
- 对 PyTorch 里“原地更新”和“重新赋值”的差别有了实际报错经验。

还不够稳的地方：

- 对自动求导的内部机制还只是会用级理解。
- 对张量形状变化还需要继续练习。
- 对 SGD 的数学推导还没有完全熟练。
- 对 Jupyter cell 执行顺序仍需要保持警惕，改完函数后必须重新运行对应 cell。

## 10. 下一步计划

- 重新从头运行一次 `5LinearRegression.ipynb`，确认整篇 notebook 能顺序执行。
- 复述训练循环：用自己的话解释 `l.sum().backward()`、`sgd([w, b], lr, batch_size)`、`zero_()`。
- 对照 D2L 下一节“线性回归的简洁实现”，比较手写版和框架版的差别。
- 专门补一小段 PyTorch 自动求导练习，巩固 `requires_grad`、`grad`、`backward`、`no_grad`。
- 继续记录容易混淆的语法：点、逗号、括号、`shape`、`reshape`、原地操作 `_`。
