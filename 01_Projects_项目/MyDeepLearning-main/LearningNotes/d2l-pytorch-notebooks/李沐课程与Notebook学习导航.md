# 李沐 D2L 课程与 PyTorch Notebook 学习导航

视频播放列表按“单节课”拆分，而本目录按书的“章”组织。找 notebook 时，不要按视频序号猜文件：先看视频主题，再在下表中打开同名主题对应的 `.ipynb`。

> **从基础开始，不要先学 Transformer。** 推荐先完成预备知识、线性神经网络和多层感知机；有了这些基础，再进入卷积网络、循环网络和注意力机制。

## 初学者推荐顺序

以下路径均相对于当前 `d2l-pytorch-notebooks/` 目录。

| 阶段 | 课程主题 | 对应 notebook | 本阶段目标 |
|---|---|---|---|
| 0 | 安装与 Jupyter 使用 | `chapter_installation/index.ipynb`；需要时看 `chapter_appendix-tools-for-deep-learning/jupyter.ipynb` | 能打开、运行和重启 notebook |
| 1 | 数据操作与预处理 | `chapter_preliminaries/ndarray.ipynb`；`chapter_preliminaries/pandas.ipynb` | 熟悉张量、形状、索引和缺失值处理 |
| 2 | 线性代数、微积分、自动微分、概率 | `chapter_preliminaries/linear-algebra.ipynb`；`chapter_preliminaries/calculus.ipynb`；`chapter_preliminaries/autograd.ipynb`；`chapter_preliminaries/probability.ipynb` | 看懂后续模型中的矩阵运算、梯度和概率 |
| 3 | 线性回归 | `chapter_linear-networks/linear-regression.ipynb` → `chapter_linear-networks/linear-regression-scratch.ipynb` → `chapter_linear-networks/linear-regression-concise.ipynb` | 理解损失函数、梯度下降，以及从零实现与框架实现的对应关系 |
| 4 | Softmax 回归与 Fashion-MNIST | `chapter_linear-networks/softmax-regression.ipynb`；`chapter_linear-networks/image-classification-dataset.ipynb`；`chapter_linear-networks/softmax-regression-scratch.ipynb`；`chapter_linear-networks/softmax-regression-concise.ipynb` | 从回归过渡到多分类和完整训练循环 |
| 5 | 多层感知机（MLP） | `chapter_multilayer-perceptrons/mlp.ipynb` → `chapter_multilayer-perceptrons/mlp-scratch.ipynb` → `chapter_multilayer-perceptrons/mlp-concise.ipynb` | 理解隐藏层、激活函数和更深网络 |
| 6 | 模型选择与正则化 | `chapter_multilayer-perceptrons/underfit-overfit.ipynb`；`chapter_multilayer-perceptrons/weight-decay.ipynb`；`chapter_multilayer-perceptrons/dropout.ipynb` | 识别过拟合，并掌握权重衰减和 Dropout |
| 7 | 数值稳定性与反向传播 | `chapter_multilayer-perceptrons/numerical-stability-and-init.ipynb`；`chapter_multilayer-perceptrons/backprop.ipynb` | 理解初始化、梯度传播和训练稳定性 |
| 8 | 深度学习计算基础 | `chapter_deep-learning-computation/model-construction.ipynb`；`chapter_deep-learning-computation/parameters.ipynb`；`chapter_deep-learning-computation/custom-layer.ipynb`；`chapter_deep-learning-computation/read-write.ipynb`；`chapter_deep-learning-computation/use-gpu.ipynb` | 会组织模型、管理参数、保存模型并使用 GPU |
| 9 | 卷积神经网络基础 | `chapter_convolutional-neural-networks/why-conv.ipynb`；`chapter_convolutional-neural-networks/conv-layer.ipynb`；`chapter_convolutional-neural-networks/padding-and-strides.ipynb`；`chapter_convolutional-neural-networks/channels.ipynb`；`chapter_convolutional-neural-networks/pooling.ipynb`；`chapter_convolutional-neural-networks/lenet.ipynb` | 理解卷积、通道、池化和经典 CNN |
| 10 | 现代卷积网络 | `chapter_convolutional-modern/alexnet.ipynb`；`chapter_convolutional-modern/vgg.ipynb`；`chapter_convolutional-modern/nin.ipynb`；`chapter_convolutional-modern/googlenet.ipynb`；`chapter_convolutional-modern/batch-norm.ipynb`；`chapter_convolutional-modern/resnet.ipynb`；`chapter_convolutional-modern/densenet.ipynb` | 认识现代 CNN 的关键结构演进 |
| 11 | 序列与循环神经网络 | `chapter_recurrent-neural-networks/sequence.ipynb`；`chapter_recurrent-neural-networks/text-preprocessing.ipynb`；`chapter_recurrent-neural-networks/language-models-and-dataset.ipynb`；`chapter_recurrent-neural-networks/rnn-scratch.ipynb`；`chapter_recurrent-neural-networks/rnn-concise.ipynb` | 理解序列数据、语言模型和 RNN |
| 12 | GRU、LSTM 与机器翻译 | `chapter_recurrent-modern/gru.ipynb`；`chapter_recurrent-modern/lstm.ipynb`；`chapter_recurrent-modern/encoder-decoder.ipynb`；`chapter_recurrent-modern/machine-translation-and-dataset.ipynb`；`chapter_recurrent-modern/seq2seq.ipynb` | 掌握更稳定的序列模型和编码器—解码器 |
| 13 | 注意力机制 | `chapter_attention-mechanisms/attention-cues.ipynb`；`chapter_attention-mechanisms/nadaraya-waston.ipynb`；`chapter_attention-mechanisms/attention-scoring-functions.ipynb`；`chapter_attention-mechanisms/bahdanau-attention.ipynb` | 先理解“为什么需要注意力”及其基本计算 |
| 14 | 自注意力与 Transformer | `chapter_attention-mechanisms/multihead-attention.ipynb`；`chapter_attention-mechanisms/self-attention-and-positional-encoding.ipynb`；最后看 `chapter_attention-mechanisms/transformer.ipynb` | 在完成前面基础后，再学习 Transformer |

不必一次学完全部 notebook。初次学习建议先完成 **阶段 0–8**，再根据兴趣选择 CNN（阶段 9–10）或序列模型（阶段 11–14）。

## 已知示例：Dropout 视频

你提供的 Bilibili 链接 `BV1EY6FBwE8P?p=33` 对应 Dropout 课。配套打开：

`chapter_multilayer-perceptrons/dropout.ipynb`

学习前最好已经看过：

1. `chapter_multilayer-perceptrons/mlp.ipynb`
2. `chapter_multilayer-perceptrons/underfit-overfit.ipynb`
3. `chapter_multilayer-perceptrons/weight-decay.ipynb`

本节重点不是背 API，而是理解：训练时为什么随机丢弃隐藏单元、输出为什么需要缩放，以及测试时为什么通常关闭 Dropout。

## 一节视频的使用方法

1. 看视频标题中的主题词，例如“线性回归”“权重衰减”“Dropout”。
2. 在上表找到对应阶段和 notebook；若有多个文件，按箭头或表内顺序打开。
3. 第一遍先跟着视频阅读 Markdown 和公式，不急着修改代码。
4. 第二遍从上到下运行单元格，并用自己的话记录输入、模型、损失和输出。
5. 最后只改一个小变量做实验，例如学习率、隐藏单元数或 Dropout 概率，记录结果后再进入下一节。

若视频标题与文件名不完全一致，以主题含义为准；本导航不绑定不确定的播放列表序号，避免课程重新编排后失效。
