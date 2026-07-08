# Andrew Ng Machine Learning Notebooks

这个文件夹用于记录吴恩达机器学习基础课程的同步实操。建议学习方式：

1. 先看课程对应小节。
2. 在 notebook 中复现公式和代码。
3. 每个 notebook 最后写 5-10 行总结。
4. 学完一章后，把核心代码整理成 `.py` 脚本或项目。

## 文件说明

| 文件 | 内容 |
|---|---|
| `01_linear_regression.ipynb` | 线性回归、MSE、梯度下降、sklearn 对比 |
| `02_logistic_regression.ipynb` | 逻辑回归、sigmoid、交叉熵、决策边界 |
| `03_regularization_overfitting.ipynb` | 过拟合、正则化、学习曲线、train/val/test |
| `04_sklearn_classification_baseline.ipynb` | sklearn 分类 baseline、指标、混淆矩阵、ROC-AUC |

## 启动方式

在 PowerShell 中运行：

```powershell
cd /d D:\MyDeepLearning
jupyter lab
```

然后打开 `andrew-ng-ml-notebooks` 文件夹。

## 建议节奏

- 线性回归：1-2 天
- 逻辑回归：2-3 天
- 正则化与过拟合：2-3 天
- sklearn baseline：3-5 天

不要只运行代码。每个实验都要改参数、画图、写观察结论。
