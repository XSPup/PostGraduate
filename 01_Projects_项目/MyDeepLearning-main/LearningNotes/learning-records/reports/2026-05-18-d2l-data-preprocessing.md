# 项目学习报告：D2L PyTorch - 数据预处理

## 1. 基本信息

- 日期：2026-05-18
- 学习主题：Dive into Deep Learning（PyTorch 版）- 数据预处理
- 本地路径：[pandas.ipynb](/D:/MyDeepLearning/d2l-zh/pytorch/chapter_preliminaries/pandas.ipynb)
- 手写练习文件：[2DataPreprocessing.ipynb](/D:/MyDeepLearning/MyHandwrittenNotebookwithLIMU/2DataPreprocessing.ipynb)
- 学习方式：边运行边提问，按概念逐个拆解并补充中文注释
- 当前状态：本节已完成，今天后续还会继续下一节

## 2. 今天这一节学了什么

今天完成了 D2L 中“数据预处理”这一节的第一轮学习和手写练习，重点不是追求一次写出完整答案，而是弄清每一行代码在做什么、数据格式为什么会变化、以及 notebook 反复运行时为什么会报错。

本节的核心流程已经跑通：

1. 用 `os` 和 `open(..., 'w')` 手动创建一个小型 CSV 文件。
2. 用 `pandas.read_csv()` 读入原始数据。
3. 观察并统计缺失值 `NaN`。
4. 用 `iloc` 把输入特征和输出标签分开。
5. 用 `pd.get_dummies(..., dummy_na=True)` 对类别列做独热编码，并把缺失值也编码进去。
6. 用 `to_numpy(dtype=float)` 和 `torch.tensor(...)` 把表格数据转换成张量。
7. 完成练习：构造 5 行 5 列的新 CSV，统计每列缺失值数量，找出缺失值最多的列并删除。

## 3. 我今天真正弄懂的点

### 3.1 `os`、路径和写文件

不再只是照抄：

```python
os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
```

已经理解为：

- `os.path.join(...)` 是拼路径。
- `..` 表示上一级目录。
- `os.makedirs(..., exist_ok=True)` 是创建文件夹，已存在也不报错。
- `open(..., 'w')` 是写入模式，如果文件已存在会覆盖原内容。

### 3.2 `inputs` 和 `outputs` 的切分

已经能看懂：

```python
inputs = data.iloc[:, 0:2]
outputs = data.iloc[:, 2:3]
```

并且理解：

- `iloc` 是按位置取数据。
- `:` 表示所有行。
- `0:2` 表示取第 0、1 列，不包含第 2 列。
- `2` 和 `2:3` 的显示效果不同。
- `data.iloc[:, 2]` 得到的是 `Series`，不按表格形式显示列名。
- `data.iloc[:, 2:3]` 得到的是 `DataFrame`，会保留表头 `Price`。

### 3.3 `pd.get_dummies(...)` 的作用

已经理解下面这句不是“神秘转换”，而是把文字类别转换成模型更容易使用的 0/1 列：

```python
inputs = pd.get_dummies(inputs, dummy_na=True)
```

其中：

- `get_dummies()` 会把类别列拆成多列。
- `dummy_na=True` 表示把缺失值 `NaN` 也单独当成一种类别。
- `Alley` 会变成像 `Alley_Pave`、`Alley_nan` 这样的列。

### 3.4 `DataFrame`、`NumPy 数组`、`Tensor`

今天这部分虽然最开始很抽象，但已经建立了一个能用的直觉：

- `DataFrame`：给人看和做数据整理的表格。
- `NumPy 数组`：更偏向数值计算的纯数字矩阵。
- `Tensor`：PyTorch 用来训练模型的数据格式，可以先理解为“深度学习里的数组”。

所以这两句的理解已经清楚了：

```python
X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(outputs.to_numpy(dtype=float))
```

也就是：

```text
DataFrame -> NumPy 数组 -> PyTorch Tensor
```

## 4. 练习部分完成情况

### 4.1 自己构造了更大的原始数据集

按练习要求创建了一个 5 行 5 列的 CSV：

- `NumRooms`
- `Alley`
- `Floor`
- `Age`
- `Price`

并且故意加入了多处 `NA`，方便继续练习缺失值处理。

### 4.2 学会统计每一列有多少个 `NaN`

已经会用：

```python
data.isna().sum()
```

得到结果：

```text
NumRooms    2
Alley       3
Floor       2
Age         1
Price       0
```

### 4.3 学会找出缺失值最多的列

已经会用：

```python
index = data.isna().sum().idxmax()
```

并确认缺失值最多的是：

```text
Alley
```

### 4.4 学会删除缺失值最多的列

已经成功运行：

```python
data = pd.read_csv(data_file)
index = data.isna().sum().idxmax()
data = data.drop(columns=[index])
print(data)
```

同时也发现了一个 notebook 很典型的问题：

- 第一次运行 `drop(...)` 可以正常删除。
- 如果不重读原始数据，反复运行同一个 cell，就会因为 `Alley` 已经被删掉而报 `KeyError`。

这是今天一个很重要的实际收获。

## 5. 今天遇到的问题和解决方式

### 问题 1：Jupyter 内核和 notebook 启动混乱

今天前半段实际花了不少时间在环境问题上，包括：

- 浏览器里 notebook 双击打不开。
- 选择内核后提示找不到 kernel。
- `.venv` 里缺少 `ipykernel`。
- `.venv` 里缺少 `pandas`。
- `.venv` 里缺少 `notebook`，导致 `python -m notebook` 一开始无法启动。

后面都已逐步修好，现在已经明确：

- `pandas` 已安装到 `.venv`
- `ipykernel` 已安装并注册
- `notebook` 已安装到 `.venv`

### 问题 2：变量名和数据结构不够直观

今天提问较多的地方集中在：

- `as pd` 为什么这样写
- `iloc` 到底是什么意思
- `Series` 和 `DataFrame` 为什么显示不同
- `Tensor` 和 `NumPy 数组` 有什么区别

这些问题都已经逐步理顺，说明今天的学习不是“看懂结果”，而是开始真正理解每个中间步骤。

## 6. 当前掌握程度

目前对本节的掌握大致可以这样评价：

- 能看懂并解释原始示例中的大部分代码。
- 能独立复现创建 CSV、读取数据、分离输入输出、编码类别特征、转张量的基本流程。
- 能完成“删除缺失值最多的列”这个练习的主干逻辑。
- 对 notebook 里“重复运行同一 cell 导致状态变化”的问题已经有了实际体会。

还不够稳的地方：

- 对 `NumPy` 和 `Tensor` 的区别目前还是“够用级理解”，还不算真正熟练。
- 练习的第 2 问“将预处理后的数据集转换为张量格式”今天还没有完整往下写完。
- 今天这一节结束后，下一节内容还没开始正式记录。

## 7. 下一步最自然的衔接

当前最自然的下一步不是回头重学，而是顺着今天这一节继续做两件事：

1. 把练习第 2 问完整写完：
   - 先删掉缺失值最多的列
   - 再把剩余特征做数值化处理
   - 最后转换成张量
2. 继续进入下一节新内容时，保持今天这种“每次只搞懂一个概念”的节奏

## 8. 预留：今天后续下一节记录

下面这部分故意留空，等你今天继续学下一节后再往里补。

### 8.1 下一节主题

- D2L PyTorch - 线性代数（入门部分）
- 重点围绕：向量、矩阵、按轴求和、非降维求和、点积、矩阵乘法

### 8.2 下一节我做了什么

- 开始进入 D2L 的“线性代数”部分，并在手写 notebook 里跟着敲基础例子。
- 重新区分了向量和矩阵的形状表示，知道了一维张量和二维张量在显示形式上的差别。
- 学到了 `A.sum(axis=1, keepdims=True)` 这种写法，理解了“按某个轴求和”和“保留维度”分别是什么意思。
- 专门弄清楚了“非降维求和”这个概念：求和后不是把那个轴删掉，而是把它保留成长度为 1 的轴。
- 理清了“点积”和“矩阵乘法”的关系：点积可以看成矩阵乘法的特殊情况，但两者不能直接完全等同。
- 弄清楚了 PyTorch 里 `torch.mm(A, B)` 的 `mm` 是 `matrix multiply`，表示二维矩阵乘法。
- 进一步确认了数学里的 `AB`，在 PyTorch 代码里通常要写成 `A @ B`，而不是直接写 `AB`。

### 8.3 下一节遇到的问题

- 一开始对 `axis=1` 的理解不够直观，不容易一下子看出它到底是在“按行求和”还是“按列求和”。
- `keepdims=True` 这个参数刚看到时比较抽象，不容易明白“为什么求和之后还要保留这个维度”。
- 容易把“点积”“矩阵乘法”“逐元素乘法”混在一起，尤其是刚看到 `A @ B`、`A * B`、`torch.mm(A, B)` 这些不同写法时。
- 数学记号和代码记号之间有落差。数学里写 `AB` 很自然，但在 Python / PyTorch 里不能直接这么写。

### 8.4 下一节的关键收获

- `axis` 决定“沿哪个方向做汇总”，而不是随便写的参数；以后看到 `sum(axis=...)` 要先看清楚是在保留哪些维度、压缩哪些维度。
- `keepdims=True` 的核心作用不是“语法好看”，而是为了让结果继续保持合适的形状，方便后面做广播运算。
- `A.sum(axis=1)` 和 `A.sum(axis=1, keepdims=True)` 的数值可能一样，但张量形状不同，这会直接影响后续计算能不能顺利进行。
- 点积本质上可以理解成“行向量乘列向量”，而矩阵乘法可以理解成“很多个点积组合起来”。
- 在 PyTorch 里，`A @ B` 通常表示矩阵乘法，`torch.mm(A, B)` 是专门做二维矩阵乘法，而 `A * B` 是逐元素相乘，不是矩阵乘法。
- 这部分虽然还只是入门，但已经开始接触到以后深度学习里非常常见的张量形状、广播和矩阵计算基础。
