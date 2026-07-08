# 项目学习报告：simple-linear-regression-notebook

## 1. 基本信息

- 日期：2026-05-08
- 项目名称：simple-linear-regression-notebook
- GitHub 链接：https://github.com/philipphundertmark/simple-linear-regression-notebook
- 本地路径：D:\MyDeepLearning\simple-linear-regression-notebook
- 项目类型：机器学习入门 / 线性回归 / Jupyter Notebook
- 难度：入门，小数据集，适合轻薄本

## 2. 我为什么学这个项目

我正在学习吴恩达机器学习/深度学习课程中的回归思想。这个项目用很小的数据集，从零实现简单线性回归，可以帮助我把公式 `y = mx + b`、损失函数、梯度下降和代码运行流程联系起来。

## 3. 本次学习目标

- 学会从 GitHub 把一个小项目下载到本地。
- 学会在项目目录中创建并激活 Python 虚拟环境。
- 学会在 VS Code 中打开 notebook 并选择项目自己的 Python 环境。
- 理解 `data.csv`、散点图、`m`、`b`、MSE、梯度下降的基本含义。

## 4. 我实际做了什么

1. 检查了 Python 和 pip：

```cmd
python --version
pip --version
```

2. 第一次直接输入：

```cmd
cd D:\MyDeepLearning
```

但因为使用的是 Windows `cmd`，没有真正切换到 D 盘，导致仓库被下载到了 `C:\Users\A`。

3. 用正确方式切换到了 D 盘：

```cmd
cd /d D:\MyDeepLearning
```

4. 重新克隆项目：

```cmd
git clone https://github.com/philipphundertmark/simple-linear-regression-notebook.git
```

5. 删除了误下载到 C 盘的副本：

```cmd
rmdir /s /q C:\Users\A\simple-linear-regression-notebook
```

6. 进入项目并创建/激活虚拟环境：

```cmd
cd simple-linear-regression-notebook
python -m venv .venv
.venv\Scripts\activate
```

7. 安装依赖时发现旧版 `requirements.txt` 不兼容 Python 3.13，于是改为安装现代版本依赖：

```cmd
pip install numpy matplotlib pandas scikit-learn scipy notebook
```

8. 准备在 VS Code 中继续运行和理解 notebook：

```cmd
code .
```

## 5. 项目结构

- `data.csv`：数据集，两列数据，第一列是学习小时数，第二列是考试分数。
- `linear-regression-demo.ipynb`：主要学习文件，包含读取数据、画散点图、定义损失函数、梯度下降和画拟合线。
- `requirements.txt`：项目原始依赖文件，但版本太老，不适合 Python 3.13。
- `README.md`：项目说明文档。

## 6. 核心代码理解

读取数据并画散点图：

```python
points = genfromtxt('data.csv', delimiter=',')
x = array(points[:,0])
y = array(points[:,1])
plt.scatter(x, y)
plt.show()
```

我的理解：这段代码把 `data.csv` 读成数组，第一列作为横坐标 `x`，第二列作为纵坐标 `y`，然后用散点图展示“学习时间”和“考试分数”的关系。

修改散点颜色：

```python
plt.scatter(x, y, color='#8B3A3A')
```

注意：Matplotlib 使用 `color`，不是 `colour`。

超参数：

```python
learning_rate = 0.0001
initial_b = 0
initial_m = 0
num_iterations = 10
```

我的理解：`m` 和 `b` 是模型要学习的参数；`learning_rate`、初始值、迭代次数是训练前人为设定的超参数。`initial_m = 0` 且 `initial_b = 0` 时，初始直线是 `y = 0`，也就是 x 轴。

损失函数：

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

我的理解：`total_cost` 是残差平方和 RSS，`total_cost / N` 是均方误差 MSE。MSE 衡量当前直线预测得有多差，训练目标是让 MSE 越来越小。

## 7. 遇到的问题和解决办法

### 问题 1：仓库没有出现在 D 盘文件夹里

- 原因：在 `cmd` 中输入 `cd D:\MyDeepLearning` 不会自动切换盘符，实际还停留在 `C:\Users\A`。
- 解决办法：使用：

```cmd
cd /d D:\MyDeepLearning
```

- 学到什么：Windows `cmd` 跨盘符切换目录时要加 `/d`。

### 问题 2：`pip install -r requirements.txt` 报错

- 原因：项目很老，`requirements.txt` 里有 `functools32`，它只支持 Python 2.7；而当前 Python 是 3.13.5。
- 解决办法：不再使用旧的 `requirements.txt`，改为安装现代依赖：

```cmd
pip install numpy matplotlib pandas scikit-learn scipy notebook
```

- 学到什么：老项目的依赖文件不一定适合现代 Python，要看报错内容判断。

### 问题 3：Matplotlib 修改颜色时报错

错误写法：

```python
plt.scatter(x, y, colour='red')
```

正确写法：

```python
plt.scatter(x, y, color='red')
```

- 原因：Matplotlib 参数名使用美式英语 `color`。
- 学到什么：库函数参数名必须严格匹配。

## 8. 运行结果

当前已经完成：

- 成功 clone GitHub 仓库到 D 盘。
- 成功创建并激活 `.venv` 虚拟环境。
- 成功安装现代版本依赖。
- 已经理解读取 CSV、画散点图、修改散点颜色、超参数、MSE 的基本含义。

后续还需要继续运行完整 notebook，观察：

- `Optimized b`
- `Optimized m`
- `Minimized cost`
- `Cost per iteration`
- 最终拟合直线图

## 9. 当前理解

这个项目的核心是用一条直线拟合散点：

```text
y_hat = m * x + b
```

其中 `x` 是学习小时数，`y_hat` 是预测分数。模型先从 `m = 0, b = 0` 开始，也就是从 x 轴开始。然后通过计算 MSE 判断当前直线有多差，再用梯度下降不断调整 `m` 和 `b`，让直线越来越贴近数据点。

## 10. 下一步计划

- 继续阅读并理解 `step_gradient()`，重点理解 `m_gradient` 和 `b_gradient`。
- 把 `num_iterations` 从 10 改成 100、1000，观察 MSE 和拟合线的变化。
- 尝试修改散点颜色、拟合线颜色、图标题，让图更容易阅读。
- 用自己的话解释“为什么梯度下降能让 MSE 下降”。
- 完成后再找一个小型逻辑回归项目，和吴恩达 Logistic Regression 板书对应起来。
