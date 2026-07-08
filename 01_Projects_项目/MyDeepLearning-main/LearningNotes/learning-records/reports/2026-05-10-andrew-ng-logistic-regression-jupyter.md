# 项目学习报告：Andrew Ng ML - Logistic Regression Jupyter Notebook

## 1. 基本信息

- 日期：2026-05-10
- 项目名称：Andrew Ng Machine Learning Notebooks - 02 Logistic Regression
- GitHub 链接：暂无，当前为本地课程同步实操
- 数据来源：https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv
- 本地路径：D:\MyDeepLearning\andrew-ng-ml-notebooks
- 主要 notebook：D:\MyDeepLearning\andrew-ng-ml-notebooks\02_logistic_regression.ipynb
- 项目类型：吴恩达机器学习基础同步实操 / Jupyter Notebook / 逻辑回归 / 二分类
- 难度：入门到中级，涉及分类、概率、交叉熵、阈值和模型评估

## 2. 我为什么学这个项目

第一个 notebook 已经完成了线性回归，主要解决连续数值预测问题。本次实验进入逻辑回归，用来理解分类问题。

线性回归预测的是分数、房价这类连续值；逻辑回归预测的是类别，例如“是否患病”“是否录取”“是否为垃圾邮件”。这个实验承接前面的线性模型思想，但加入 sigmoid 和交叉熵，是从回归进入分类的关键一步。

## 3. 本次学习目标

- 理解二分类问题的基本形式。
- 了解 Pima Indians Diabetes 数据集的背景。
- 学会用 pandas 从 GitHub raw CSV 读取小型数据集。
- 理解 `X` 是特征，`y` 是标签。
- 理解为什么逻辑回归要使用 sigmoid。
- 理解为什么分类任务使用交叉熵，而不是 MSE。
- 手写逻辑回归的预测概率、交叉熵、梯度下降和类别预测。
- 调整 `LEARNING_RATE`、`EPOCHS`、`THRESHOLD` 并观察结果。
- 对比手写逻辑回归和 sklearn `LogisticRegression`。

## 4. 我实际做了什么

### 4.1 补全第二个 notebook 框架

在以下文件中完成逻辑回归实验框架：

```text
D:\MyDeepLearning\andrew-ng-ml-notebooks\02_logistic_regression.ipynb
```

框架包含：

- 读取 GitHub CSV 数据
- 查看数据基本信息
- 划分训练集和测试集
- 特征标准化
- sigmoid 函数
- 手写逻辑回归
- 交叉熵损失
- 梯度下降
- accuracy、confusion matrix、classification report
- sklearn 对比
- 实验总结

### 4.2 理解实验背景

本次使用的是 Pima Indians Diabetes 数据集。每一行代表一个人，输入特征包括：

| 特征 | 含义 |
|---|---|
| `Pregnancies` | 怀孕次数 |
| `Glucose` | 血糖指标 |
| `BloodPressure` | 血压 |
| `SkinThickness` | 皮肤厚度 |
| `Insulin` | 胰岛素 |
| `BMI` | 身体质量指数 |
| `DiabetesPedigreeFunction` | 糖尿病家族遗传相关指标 |
| `Age` | 年龄 |

标签是：

```text
Outcome
```

其中：

```text
0：没有糖尿病
1：有糖尿病
```

所以这个任务是二分类：

```text
身体指标 -> 是否有糖尿病
```

### 4.3 填写关键参数

当前 notebook 保存的主要参数为：

```python
TEST_SIZE = 0.2
LEARNING_RATE = 0.05
EPOCHS = 10000
THRESHOLD = 0.5
```

含义：

- `TEST_SIZE = 0.2`：20% 数据作为测试集。
- `LEARNING_RATE`：梯度下降每次参数更新的步长。
- `EPOCHS`：训练轮数。
- `THRESHOLD = 0.5`：概率大于等于 0.5 时预测为 1，否则预测为 0。

### 4.4 对学习率和训练轮数进行调试

一开始使用过较小学习率和较少训练轮数：

```python
LEARNING_RATE = 0.001
EPOCHS = 1000
```

结果：

```text
final train loss: 0.6007
Manual train accuracy: 76.71%
Manual test accuracy: 70.78%
```

后来增加训练轮数并调整学习率，最终结果变为：

```text
final train loss: 0.4662
Manual train accuracy: 79.48%
Manual test accuracy: 71.43%
```

这说明之前训练还不够充分。增加训练轮数和调整学习率后，模型进一步收敛。

### 4.5 观察阈值变化

尝试过：

```python
THRESHOLD = 0.4
THRESHOLD = 0.5
THRESHOLD = 0.6
```

当前实验中，`0.4` 和 `0.6` 的 accuracy 都不如 `0.5`。说明在当前模型和测试集上，`0.5` 对 accuracy 更合适。

但医疗场景不能只看 accuracy，还需要关注：

- 假阳性：本来没有病，但预测有病。
- 假阴性：本来有病，但预测没有病。

尤其在疾病筛查中，假阴性通常风险更大。

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

  learning-records
    reports
      2026-05-10-andrew-ng-linear-regression-jupyter.md
      2026-05-10-andrew-ng-logistic-regression-jupyter.md
```

## 6. 核心代码理解

### 6.1 读取数据

```python
DATA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
df = pd.read_csv(DATA_URL, names=columns)
```

理解：

`pd.read_csv()` 可以直接读取 GitHub raw CSV。因为这个数据集没有表头，所以需要用 `names=columns` 手动指定列名。

### 6.2 划分特征和标签

```python
X = df.drop(columns="Outcome").to_numpy(dtype=float)
y = df["Outcome"].to_numpy(dtype=float)
```

理解：

- `X` 是输入特征，包含 8 个身体指标。
- `y` 是标签，只包含 `0` 和 `1`。

### 6.3 sigmoid

```python
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))
```

理解：

sigmoid 把任意实数压到 `0~1` 之间。逻辑回归先计算：

```text
z = Xw + b
```

再计算：

```text
p = sigmoid(z)
```

这个 `p` 可以理解为模型认为样本属于正类 `1` 的概率。

### 6.4 交叉熵

```python
loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
```

理解：

分类任务关注的是模型给真实类别的概率是否足够高。如果真实标签是 `1`，模型却只给出很低概率，交叉熵会很大。

这也是本实验不用 MSE 的原因：MSE 更适合连续数值预测，交叉熵更适合概率分类。

### 6.5 梯度下降

```python
dw = (X.T @ error) / m
db = np.sum(error) / m
```

理解：

逻辑回归也可以用梯度下降训练。`dw` 和 `db` 表示当前损失函数对参数的梯度。每次更新：

```python
w = w - learning_rate * dw
b = b - learning_rate * db
```

### 6.6 概率转类别

```python
return (probabilities >= threshold).astype(int)
```

理解：

逻辑回归输出的是概率。要得到最终类别，需要设置阈值。默认 `threshold=0.5`：

- 概率大于等于 0.5，预测为 1。
- 概率小于 0.5，预测为 0。

## 7. 遇到的问题和解决办法

### 问题 1：不理解数据集背景

- 原因：数据列名和医学指标比较陌生。
- 解决办法：把实验简化理解为“根据 8 个身体指标判断是否有糖尿病”。
- 学到什么：机器学习实验先要理解任务，而不是一上来就看公式。

### 问题 2：`assert TEST_SIZE is not 0.2` 写错

错误写法：

```python
assert TEST_SIZE is not 0.2
```

这个意思变成了“TEST_SIZE 不能是 0.2”。

正确写法：

```python
assert TEST_SIZE is not None
```

- 学到什么：`assert` 是检查条件是否成立。这里真正要检查的是参数有没有从 `None` 改成具体值。

### 问题 3：学习率太低导致训练不足

一开始：

```python
LEARNING_RATE = 0.001
EPOCHS = 1000
```

训练损失较高，手写模型测试准确率略低于 sklearn。

调整后：

```python
EPOCHS = 10000
```

训练损失降低到约 `0.4662`，手写模型测试准确率达到 `71.43%`，和 sklearn 一致。

- 学到什么：如果学习率小，训练轮数不够时模型可能还没收敛。

### 问题 4：继续增加 epoch 后准确率不再提升

- 原因：模型基本已经收敛，继续训练同一个逻辑回归模型不会明显改变分类边界。
- 学到什么：当优化已经到头时，提升效果要考虑特征处理、模型表达能力、阈值策略或更强模型，而不是只增加 epoch。

### 问题 5：是否需要神经网络

当前理解：

逻辑回归本质上是线性分类器。它只能学习一种线性决策关系。如果身体指标和糖尿病之间存在复杂非线性关系，逻辑回归表达能力可能不足。

但数据集只有 768 行，直接使用神经网络也可能过拟合。更合理的后续顺序是：

1. 学习正则化和过拟合。
2. 尝试决策树、随机森林等传统模型。
3. 再尝试小型神经网络。

## 8. 运行结果

当前主要输出：

```text
w shape: (8,)
b: -0.8807
final train loss: 0.4662
```

手写逻辑回归：

```text
Manual train accuracy: 79.48%
Manual test accuracy:  71.43%
```

混淆矩阵：

```text
[[82 18]
 [26 28]]
```

含义：

- 真实为 0 且预测为 0：82
- 真实为 0 但预测为 1：18
- 真实为 1 但预测为 0：26
- 真实为 1 且预测为 1：28

与 sklearn 对比：

```text
Manual test accuracy:  71.43%
sklearn test accuracy: 71.43%
```

说明手写逻辑回归在当前参数下已经基本达到 sklearn 的分类结果。

## 9. 当前理解

逻辑回归虽然名字里有“回归”，但它主要用于分类。它和线性回归的共同点是都会先做线性组合：

```text
z = Xw + b
```

区别是逻辑回归会把 `z` 输入 sigmoid，得到 `0~1` 之间的概率：

```text
p = sigmoid(z)
```

然后再用阈值把概率转成类别：

```text
p >= 0.5 -> 1
p < 0.5 -> 0
```

本次实验让我理解了：

- 回归任务常用 MSE。
- 分类任务常用交叉熵。
- 学习率太小会收敛慢。
- epoch 增加到一定程度后，准确率不再提升，说明模型本身或数据本身成为限制。
- accuracy 不是唯一指标，医疗类任务还要关注假阴性和假阳性。

## 10. 下一步计划

- 在 notebook 中补充不同阈值下 precision、recall、F1 的对比表。
- 开始 `03_regularization_overfitting.ipynb`，学习过拟合、欠拟合、正则化和 train/validation/test。
- 后续用同一个糖尿病数据集尝试决策树和随机森林，观察是否比逻辑回归更适合。
- 在学习神经网络前，先理解模型复杂度和过拟合，否则容易误以为“模型越复杂越好”。
