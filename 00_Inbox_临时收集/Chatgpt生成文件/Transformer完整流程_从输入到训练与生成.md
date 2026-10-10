---
title: Transformer 完整流程——从输入、注意力到训练与生成
aliases:
  - Transformer 完整流程
tags:
  - 深度学习
  - Transformer
  - 注意力机制
  - NLP
created: 2026-09-29
status: 完成
---

# Transformer 完整流程——从输入、注意力到训练与生成

> [!abstract] 一句话直觉
> Transformer 会让序列中的每个 token 根据任务需要去“查看”其他 token，把有用信息按权重汇总到自己的新表示中；多层重复后，再用这些表示完成理解、翻译或生成。

本文先讲 2017 年论文 **Attention Is All You Need** 中的完整 Encoder–Decoder Transformer，然后再说明 BERT、GPT 等模型删掉了哪些部分、保留了哪些部分。

为了让所有矩阵有具体落点，全文主要使用翻译例子：

```text
源语言：I love you .
目标语言：我 爱 你 。
```

---

## 0. 先看完整路线

### 0.1 训练时的完整流程

```text
源句子 I love you .
        ↓
分词与 Token ID
        ↓
词嵌入 + 位置编码
        ↓
Encoder × N 层
        ↓
Encoder 输出（memory）──────────────────────┐
                                            │
目标句子右移：<BOS> 我 爱 你 。              │
        ↓                                   │
目标词嵌入 + 位置编码                        │
        ↓                                   │
Decoder 的带因果遮挡自注意力                 │
        ↓                                   │
Decoder–Encoder 交叉注意力 ←────────────────┘
        ↓
Decoder 的前馈网络
        ↓
Decoder × N 层
        ↓
线性层映射到整个词表
        ↓
每个位置得到下一个 token 的概率
        ↓
与正确答案比较，计算交叉熵损失
        ↓
反向传播 + 优化器更新参数
```

### 0.2 推理时的完整流程

```text
I love you . → Encoder 只运行一次 → 得到源句 memory

<BOS>        → Decoder → 预测“我”
<BOS> 我     → Decoder → 预测“爱”
<BOS> 我 爱  → Decoder → 预测“你”
……
直到预测出 <EOS> 或达到最大长度
```

训练时可以并行计算目标序列的所有位置；推理时因为下一个 token 依赖已经生成的 token，所以通常必须逐步生成。

---

## 1. 先统一符号和矩阵形状

| 符号 | 含义 | 常见值或形状 |
|---|---|---|
| (B) | batch size，一批样本数量 | 例如 32 |
| (L_s) | 源序列长度 | `I love you .` 中为 4 |
| (L_t) | 目标序列长度 | 随训练或生成阶段变化 |
| (V) | 词表大小 | 例如 30,000、50,000 |
| (d_{model}) 或 (d) | 模型主维度，每个 token 的主向量长度 | 原论文为 512 |
| (h) | 注意力头数 | 原论文常用 8 |
| (d_k) | 每个头中 Query、Key 的维度 | 常见为 (d_{model}/h) |
| (d_v) | 每个头中 Value 的维度 | 常见与 (d_k) 相同 |
| (d_{ff}) | FFN 中间隐藏层维度 | 原论文为 2048 |
| (N) | Encoder 层数和 Decoder 层数 | 原论文各 6 层 |

为方便阅读，正文大多暂时省略 batch 维。实际代码中的输入通常是：

$$
X\in\mathbb{R}^{B\times L\times d_{model}}
$$

省略 batch 后写成：

$$
X\in\mathbb{R}^{L\times d_{model}}
$$

> [!important] (d_{model}) 与 (d_k) 的区别
> - (d_{model})：模型中每个 token 的主表示维度。
> - (d_k)：单个注意力头观察 token 时，Query 和 Key 使用的维度。
> - 在平均分头的常见设计中，(d_k=d_{model}/h)。这是一种常用设计，不是数学上唯一允许的选择。

例如：

$$
d_{model}=512,\quad h=8,\quad d_k=d_v=64
$$

---

## 2. 第一步：文本变成 Token ID

神经网络不能直接接收字符串，因此先通过 tokenizer 分词并映射到整数 ID。

### 2.1 分词

为了讲解简单，暂时假设按单词分词：

```text
I love you .
↓
[I, love, you, .]
```

真实模型常用子词分词，例如 BPE、WordPiece 或 SentencePiece：

```text
unbelievable
↓
[un, believe, able]
```

子词分词能够控制词表大小，并处理训练中没完整出现过的新词。

### 2.2 特殊 token

常见特殊 token 包括：

| token | 作用 |
|---|---|
| `<BOS>` | 序列开始，告诉 Decoder 开始生成 |
| `<EOS>` | 序列结束，告诉 Decoder 停止生成 |
| `<PAD>` | 将同一 batch 中不同长度的句子补齐 |
| `<UNK>` | 未知 token；现代子词模型较少依赖它 |
| `<MASK>` | BERT 掩码语言模型训练使用，并非所有 Transformer 都有 |

例如目标句子可表示为：

```text
[<BOS>, 我, 爱, 你, 。, <EOS>]
```

### 2.3 Token ID

假设词表映射为：

```text
I     → 15
love  → 87
you   → 42
.     → 9
```

那么输入序列就是：

```text
[15, 87, 42, 9]
```

这些整数只是词表索引，不表示词义大小关系。

---

## 3. 第二步：Token Embedding

模型维护一个可学习的嵌入矩阵：

$$
E\in\mathbb{R}^{V\times d_{model}}
$$

每个 Token ID 从 (E) 中查出一行：

$$
X_{emb}=E[\text{token ids}]
$$

对 `I love you .`：

$$
X_{emb}\in\mathbb{R}^{4\times d_{model}}
$$

可以理解为：

```text
                 d_model 个特征
I       [ · · · · · · · · · · · · ]
love    [ · · · · · · · · · · · · ]
you     [ · · · · · · · · · · · · ]
.       [ · · · · · · · · · · · · ]
```

每一行代表一个 token 的语义起点。Embedding 参数会通过训练被更新。

原始 Transformer 论文将词嵌入乘以：

$$
\sqrt{d_{model}}
$$

目的是调整词嵌入的数值尺度，使其与位置编码相匹配。不同实现不一定都显式保留这一步。

---

## 4. 第三步：加入位置信息

仅靠自注意力，模型本身不能区分 token 的先后顺序。若没有位置相关信息，交换两行输入会使输出相应交换，但模型不知道哪个词在前、哪个词在后。

因此构造：

$$
X_0=X_{emb}+PE
$$

### 4.1 经典正弦余弦位置编码

原论文使用：

$$
PE(pos,2i)=\sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

$$
PE(pos,2i+1)=\cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

其中：

- (pos)：token 在序列中的位置，属于序列长度 (L) 方向。
- (i)：位置向量内部的维度对编号，属于 (d_{model}) 方向。
- 偶数维用 sin，奇数维用 cos。

以 `I love you .` 为例：

| token | (pos) |
|---|---:|
| I | 0 |
| love | 1 |
| you | 2 |
| . | 3 |

位置编码形状与词嵌入相同：

$$
PE\in\mathbb{R}^{L\times d_{model}}
$$

因此可以逐元素相加。

### 4.2 现代位置方法

常见替代方案包括：

- 可学习绝对位置嵌入；
- 相对位置偏置；
- RoPE（旋转位置编码）；
- ALiBi。

它们实现方式不同，但共同目标都是让注意力感知顺序和相对距离。

### 4.3 Dropout

原始 Transformer 在词嵌入与位置编码相加后使用 Dropout：

$$
X_0=\operatorname{Dropout}(X_{emb}+PE)
$$

Dropout 在训练时随机将部分激活置零，降低过拟合；推理时关闭。

---

## 5. Encoder 总体结构

Encoder 由 (N) 个结构相同、参数不同的 Encoder Layer 堆叠而成。

每个 Encoder Layer 有两个主要子层：

1. 多头自注意力 Multi-Head Self-Attention；
2. 逐位置前馈网络 Position-wise FFN。

每个子层外面还有：

- 残差连接 Residual Connection；
- Dropout；
- Layer Normalization。

原论文采用 Post-LN，写成：

$$
Y=\operatorname{LayerNorm}(X+\operatorname{Dropout}(\operatorname{Sublayer}(X)))
$$

许多现代模型采用训练更稳定的 Pre-LN：

$$
Y=X+\operatorname{Dropout}(\operatorname{Sublayer}(\operatorname{LayerNorm}(X)))
$$

两种结构顺序不同，阅读具体模型代码时必须确认。下面先按原论文的 Post-LN 讲解。

---

## 6. Encoder 自注意力：Q、K、V 从哪里来

假设当前 Encoder 层输入为：

$$
X\in\mathbb{R}^{L_s\times d_{model}}
$$

自注意力中，Q、K、V 都来自同一个 (X)，因此叫 **Self-Attention**。

### 6.1 单个注意力头

第 (i) 个头拥有三组可学习参数：

$$
W_i^Q\in\mathbb{R}^{d_{model}\times d_k}
$$

$$
W_i^K\in\mathbb{R}^{d_{model}\times d_k}
$$

$$
W_i^V\in\mathbb{R}^{d_{model}\times d_v}
$$

计算：

$$
Q_i=XW_i^Q
$$

$$
K_i=XW_i^K
$$

$$
V_i=XW_i^V
$$

对应形状：

$$
Q_i,K_i\in\mathbb{R}^{L_s\times d_k}
$$

$$
V_i\in\mathbb{R}^{L_s\times d_v}
$$

直觉：

- Query：当前 token “想找什么信息”；
- Key：每个候选 token “有什么特征可供匹配”；
- Value：候选 token “真正要被取走和汇总的信息”。

> [!note] 实际代码中的一次大投影
> 实现中通常不会真的分别调用 (h) 次小矩阵乘法，而是一次得到形状 (L\times(hd_k)) 的大 Q、K、V，再 reshape 成多个头。数学含义仍可理解为每个头有自己的一组投影参数。

---

## 7. 计算注意力分数 (QK^T)

第 (i) 个头先计算：

$$
S_i=Q_iK_i^T
$$

形状为：

$$
(L_s\times d_k)(d_k\times L_s)=L_s\times L_s
$$

对于 `I love you .`，(L_s=4)，所以 (S_i) 是 (4\times4)：

| Query 所在行 ↓ / Key 所在列 → | I | love | you | . |
|---|---:|---:|---:|---:|
| I | (q_I\cdot k_I) | (q_I\cdot k_{love}) | (q_I\cdot k_{you}) | (q_I\cdot k_.) |
| love | (q_{love}\cdot k_I) | (q_{love}\cdot k_{love}) | (q_{love}\cdot k_{you}) | (q_{love}\cdot k_.) |
| you | (q_{you}\cdot k_I) | (q_{you}\cdot k_{love}) | (q_{you}\cdot k_{you}) | (q_{you}\cdot k_.) |
| . | (q_.\cdot k_I) | (q_.\cdot k_{love}) | (q_.\cdot k_{you}) | (q_.\cdot k_.) |

每个格子表示：某个 Query 与某个 Key 的匹配程度。

---

## 8. 为什么除以 \(\sqrt{d_k}\)

缩放点积注意力定义为：

$$
\frac{Q_iK_i^T}{\sqrt{d_k}}
$$

当 (d_k) 较大时，点积是很多维乘积之和，数值幅度容易随维度增大。过大的数值进入 Softmax 后，会让概率过早接近 0 或 1，梯度可能变得很小。

除以 \(\sqrt{d_k}\) 可以控制分数尺度，让训练更稳定。

---

## 9. Mask：哪些位置不允许被看见

在 Softmax 之前，可以把不允许关注的位置加上一个极大的负数，理论上写作 \(-\infty\)：

$$
\widetilde S_i=\frac{Q_iK_i^T}{\sqrt{d_k}}+M
$$

然后：

$$
A_i=\operatorname{Softmax}(\widetilde S_i)
$$

被 Mask 的位置经过 Softmax 后权重约为 0。

### 9.1 Padding Mask

同一 batch 中序列长度不同，短序列常补 `<PAD>`：

```text
句子1：[I, love, you, .]
句子2：[Hello, ., <PAD>, <PAD>]
```

`<PAD>` 只是为了对齐形状，不包含有效语义，所以其他 token 不应关注它。Padding Mask 通常遮住 Key 方向上的 `<PAD>` 列；损失计算时也要忽略 `<PAD>` 标签。

### 9.2 Causal Mask

Causal Mask 主要用于 Decoder 自注意力，禁止当前位置偷看未来 token。后文单独展开。

### 9.3 两种 Mask 可以同时存在

Decoder 训练时通常同时需要：

- Padding Mask：别看 `<PAD>`；
- Causal Mask：别看未来 token。

实现中会把它们合并或分别广播到注意力分数张量。

---

## 10. Softmax 得到注意力权重矩阵

Softmax 沿 Key 所在的最后一维进行，使每个 Query 对所有 Key 的权重之和等于 1：

$$
A_i=\operatorname{Softmax}\left(\frac{Q_iK_i^T}{\sqrt{d_k}}+M\right)
$$

假设某个头得到：

$$
A_i=
\begin{bmatrix}
0.55&0.30&0.10&0.05\\
0.30&0.10&0.55&0.05\\
0.10&0.60&0.25&0.05\\
0.10&0.10&0.10&0.70
\end{bmatrix}
$$

第二行表示模型正在更新 `love` 的表示：

```text
love 看向 I       ：0.30
love 看向 love    ：0.10
love 看向 you     ：0.55
love 看向 .       ：0.05
```

注意力权重是模型训练学到的结果，并不是人工指定“某个头必须负责语法”。

---

## 11. 为什么还要计算 \(AV\)

到目前为止，(A_i) 只说明“应该按什么比例查看其他 token”，还没有真正把内容取回来。

因此必须计算：

$$
Z_i=A_iV_i
$$

形状验证：

$$
(L_s\times L_s)(L_s\times d_v)=L_s\times d_v
$$

对于 `love` 对应的第二行：

$$
z_{love}
=0.30v_I+0.10v_{love}+0.55v_{you}+0.05v_.
$$

因此：

- (A) 是“权重”；
- (V) 是“内容”；
- (AV) 才是按权重混合后的上下文表示。

> [!warning] 常见混淆
> (V) 不是权重矩阵。真正的注意力权重是 (A=\operatorname{Softmax}(QK^T/\sqrt{d_k}+M))。

---

## 12. 多头注意力

第 (i) 个头为：

$$
head_i=\operatorname{Attention}(XW_i^Q,XW_i^K,XW_i^V)
$$

多个头可以在不同投影子空间中学习不同的关系，例如：

- 某个头更关注主语与谓语；
- 某个头更关注谓语与宾语；
- 某个头更关注相邻 token；
- 某个头更关注远距离依赖；
- 某个头更关注标点或边界。

这些只是训练后可能出现的行为，不代表每个头一定有清晰、固定、可解释的工作分工。

### 12.1 拼接各个头

每个头输出：

$$
head_i\in\mathbb{R}^{L_s\times d_v}
$$

将 (h) 个头沿特征维拼接：

$$
H=\operatorname{Concat}(head_1,\ldots,head_h)
$$

$$
H\in\mathbb{R}^{L_s\times(hd_v)}
$$

### 12.2 输出投影 \(W^O\)

拼接后乘：

$$
W^O\in\mathbb{R}^{hd_v\times d_{model}}
$$

得到：

$$
O=HW^O\in\mathbb{R}^{L_s\times d_{model}}
$$

(W^O) 有两个作用：

1. 融合不同头的信息；
2. 把维度投影回 (d_{model})，从而能够与输入 (X) 做残差相加。

### 12.3 一个完整的尺寸例子

为了便于观察，假设：

$$
L_s=4,\quad d_{model}=8,\quad h=2,\quad d_k=d_v=4
$$

| 中间量 | 形状 |
|---|---|
| (X) | (4\times8) |
| 每个 (W_i^Q,W_i^K,W_i^V) | (8\times4) |
| 每个头的 (Q_i,K_i,V_i) | (4\times4) |
| 每个头的 (Q_iK_i^T) | (4\times4) |
| 每个头的 (A_i) | (4\times4) |
| 每个头的 (A_iV_i) | (4\times4) |
| 两头拼接 | (4\times8) |
| (W^O) | (8\times8) |
| 多头注意力输出 | (4\times8) |

---

## 13. 残差连接、Dropout 与 LayerNorm

### 13.1 残差连接

原论文 Encoder 自注意力子层之后：

$$
X'=\operatorname{LayerNorm}(X+\operatorname{Dropout}(O))
$$

残差连接让模型保留原始信息，并为梯度传播提供更直接的路径，有助于训练深层网络。

相加要求两者形状完全相同：

$$
X,O\in\mathbb{R}^{L_s\times d_{model}}
$$

这也是为什么多头拼接后要通过 (W^O) 回到 (d_{model})。

### 13.2 LayerNorm

LayerNorm 对每个 token 自己的特征维进行归一化。简化表示：

$$
\operatorname{LN}(x)=\gamma\odot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta
$$

其中 \(\gamma,\beta\) 是可学习参数。

它与 BatchNorm 的主要区别是：LayerNorm 不依赖 batch 中其他样本，更适合变长序列和自回归生成。

### 13.3 Dropout 所在位置

原始 Transformer 中常见 Dropout 位置包括：

- Embedding 与位置编码相加后；
- Softmax 后的注意力权重或注意力输出附近；
- FFN 输出后；
- 残差相加前。

具体代码实现可能略有不同。

---

## 14. Position-wise Feed-Forward Network

注意力完成 token 之间的信息交换后，每个 token 独立通过同一个两层前馈网络：

$$
\operatorname{FFN}(x)=\sigma(xW_1+b_1)W_2+b_2
$$

原论文使用 ReLU：

$$
\operatorname{FFN}(x)=\max(0,xW_1+b_1)W_2+b_2
$$

形状通常为：

$$
d_{model}\rightarrow d_{ff}\rightarrow d_{model}
$$

原论文：

$$
512\rightarrow2048\rightarrow512
$$

现代模型常用 GELU、SwiGLU、GeGLU 等激活或门控结构。

“Position-wise” 表示：

- 每个位置分别计算；
- 所有位置共享同一套 FFN 参数；
- 这一步不会直接在不同 token 之间传递信息，跨 token 信息交换已经由注意力完成。

原论文第二个子层：

$$
Y=\operatorname{LayerNorm}(X'+\operatorname{Dropout}(\operatorname{FFN}(X')))
$$

---

## 15. Encoder 为什么要堆叠多层

一层 Encoder 只能完成一次：

```text
跨 token 收集信息 → 每个 token 内部变换
```

堆叠多层以后：

- 低层可能更多处理局部、词法和位置关系；
- 中层可能组合短语和句法信息；
- 高层可能形成更抽象、与任务相关的语义表示。

这些层次行为并非人工硬编码，只是训练后常见的观察。

Encoder 最终输出：

$$
H_{enc}\in\mathbb{R}^{L_s\times d_{model}}
$$

它常被称为 Encoder output、memory 或 hidden states。每一行仍对应一个源 token，但已经融合了整句上下文。

---

## 16. Decoder 的输入为什么要右移

训练目标为：

```text
我 爱 你 。 <EOS>
```

Decoder 输入则是：

```text
<BOS> 我 爱 你 。
```

两者错开一位：

| Decoder 当前看到的输入 | 应预测的标签 |
|---|---|
| `<BOS>` | 我 |
| `<BOS> 我` | 爱 |
| `<BOS> 我 爱` | 你 |
| `<BOS> 我 爱 你` | 。 |
| `<BOS> 我 爱 你 。` | `<EOS>` |

这称为 **shift right**。训练时将正确的历史 token 提供给 Decoder，称为 **Teacher Forcing**。

目标 token 同样经过：

```text
Token ID → Embedding → 加位置编码 → Dropout
```

得到：

$$
Y_0\in\mathbb{R}^{L_t\times d_{model}}
$$

---

## 17. Decoder Layer 的三个子层

原始 Decoder 每一层包含：

1. 带因果遮挡的多头自注意力；
2. Encoder–Decoder 交叉注意力；
3. Position-wise FFN。

每个子层同样配有残差、Dropout 和 LayerNorm。

---

## 18. Decoder 的 Masked Self-Attention

Decoder 的第一个注意力也是 Self-Attention，因为 Q、K、V 都来自 Decoder 当前层输入。

但它必须防止当前位置看到未来答案。

### 18.1 因果遮挡矩阵

对长度 4 的序列，可以写成：

$$
M_{causal}=
\begin{bmatrix}
0&-\infty&-\infty&-\infty\\
0&0&-\infty&-\infty\\
0&0&0&-\infty\\
0&0&0&0
\end{bmatrix}
$$

它是一个下三角可见结构：

```text
当前位置 ↓   可以看见的位置 →

第1个位置    ✓  ×  ×  ×
第2个位置    ✓  ✓  ×  ×
第3个位置    ✓  ✓  ✓  ×
第4个位置    ✓  ✓  ✓  ✓
```

因此，训练时虽然整条正确目标句都放进矩阵并行计算，第 2 个位置仍然看不到第 3、4 个位置，不会泄漏未来答案。

### 18.2 为什么训练能并行、推理却不能完全并行

训练时正确目标序列已知，因果 Mask 可以一次构建整张下三角注意力矩阵。

推理时未来 token 尚不存在，必须先生成当前 token，才能把它作为下一步输入，因此沿时间方向通常是串行的。

---

## 19. Decoder–Encoder Cross-Attention

Decoder 的第二个注意力子层连接 Decoder 和 Encoder，因此叫交叉注意力。

设 Decoder 当前隐藏状态为：

$$
H_{dec}\in\mathbb{R}^{L_t\times d_{model}}
$$

Encoder 输出为：

$$
H_{enc}\in\mathbb{R}^{L_s\times d_{model}}
$$

交叉注意力中：

$$
Q=H_{dec}W^Q
$$

$$
K=H_{enc}W^K
$$

$$
V=H_{enc}W^V
$$

也就是说：

- Query 来自 Decoder：我现在为了生成下一个目标 token，需要找什么？
- Key、Value 来自 Encoder：源句子各位置能够提供什么线索和内容？

形状：

$$
Q\in\mathbb{R}^{L_t\times d_k}
$$

$$
K\in\mathbb{R}^{L_s\times d_k},\qquad V\in\mathbb{R}^{L_s\times d_v}
$$

所以注意力权重矩阵为：

$$
\operatorname{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}+M_{src}\right)
\in\mathbb{R}^{L_t\times L_s}
$$

这里每一行对应一个目标位置，每一列对应一个源位置。例如生成“爱”时，可能更关注源句的 `love`。

交叉注意力通常使用源序列 Padding Mask，防止 Decoder 读取 Encoder 端的 `<PAD>`；它不需要对源句使用因果 Mask，因为整个源句允许被查看。

---

## 20. Decoder 的 FFN 与多层堆叠

完成：

```text
目标内部的因果自注意力
        ↓
读取源句的交叉注意力
        ↓
每个位置独立的 FFN
```

以后，一个 Decoder Layer 完成。

该过程堆叠 (N) 次，得到最终 Decoder 隐藏状态：

$$
H_{dec}^{final}\in\mathbb{R}^{L_t\times d_{model}}
$$

---

## 21. 从隐藏状态映射到词表

Decoder 的输出仍是 (d_{model}) 维向量，不能直接表示具体单词，因此使用输出投影：

$$
W_{vocab}\in\mathbb{R}^{d_{model}\times V}
$$

$$
logits=H_{dec}^{final}W_{vocab}+b
$$

形状：

$$
logits\in\mathbb{R}^{L_t\times V}
$$

每个位置都有长度为词表大小 (V) 的分数向量。

对词表维做 Softmax：

$$
p(y_t\mid y_{<t},x)=\operatorname{Softmax}(logits_t)
$$

得到每个候选 token 的概率。

> [!note] 权重共享
> 很多模型会让输出投影矩阵与输入 Embedding 矩阵共享参数，称为 weight tying。是否共享取决于具体架构。

---

## 22. 训练时如何计算 Loss

假设某位置正确答案是“爱”，模型给整个词表产生概率分布。交叉熵损失关注正确 token 的概率：

$$
\mathcal{L}_t=-\log p(y_t^{true})
$$

整条目标序列的损失通常对有效位置求和或平均：

$$
\mathcal{L}=\frac{1}{N_{valid}}\sum_{t\in valid}-\log p(y_t^{true})
$$

其中 `<PAD>` 位置必须忽略，不应计入损失。

原始 Transformer 还使用 Label Smoothing：不给正确类别绝对概率 1，而是保留少量概率给其他类别，以减轻过度自信并改善泛化。

---

## 23. 反向传播到底更新哪些参数

一次训练迭代为：

```text
1. 前向传播：得到 logits
2. 计算交叉熵 Loss
3. loss.backward()：通过链式法则计算梯度
4. optimizer.step()：根据梯度更新参数
5. optimizer.zero_grad()：清除旧梯度，准备下一次迭代
```

不同框架可能将清零梯度放在迭代开头，但核心含义不变。

梯度会从 Loss 向前传播经过：

```text
词表输出层
  ↓
Decoder 各层 FFN
  ↓
交叉注意力与 Masked Self-Attention
  ↓
Encoder 各层
  ↓
Embedding、WQ、WK、WV、WO、FFN、LayerNorm 等参数
```

需要明确区分：

- `loss.backward()`：计算每个参数的梯度；
- `optimizer.step()`：按照 SGD、Adam、AdamW 等规则真正修改参数。

原始 Transformer 使用 Adam，并配合 warmup 学习率策略：训练初期逐渐增大学习率，之后按步数衰减。

工程中还可能使用：

- 梯度裁剪；
- 混合精度训练；
- 梯度累积；
- 分布式数据并行；
- Checkpoint；
- 早停与验证集评估。

它们不是注意力公式本身，但属于完整训练流程的重要部分。

---

## 24. 训练阶段的一次完整数据对齐

源句：

```text
I love you .
```

目标完整序列：

```text
<BOS> 我 爱 你 。 <EOS>
```

送入 Decoder：

```text
<BOS> 我 爱 你 。
```

监督标签：

```text
我 爱 你 。 <EOS>
```

虽然训练时这些位置一起并行计算，但 Causal Mask 保证：

```text
预测“我”时只能看 <BOS>
预测“爱”时只能看 <BOS> 我
预测“你”时只能看 <BOS> 我 爱
预测“。”时只能看 <BOS> 我 爱 你
预测 <EOS> 时只能看 <BOS> 我 爱 你 。
```

---

## 25. 推理阶段：自回归生成

训练完成后不再有正确目标句供 Teacher Forcing 使用。

### 25.1 Encoder 只运行一次

```text
I love you . → Encoder → H_enc
```

只要源句没有变化，(H_{enc}) 就可以重复使用。

### 25.2 Decoder 逐步生成

```text
第1步：[<BOS>]                 → 预测“我”
第2步：[<BOS>, 我]             → 预测“爱”
第3步：[<BOS>, 我, 爱]         → 预测“你”
第4步：[<BOS>, 我, 爱, 你]     → 预测“。”
第5步：[<BOS>, 我, 爱, 你, 。] → 预测 <EOS>
```

遇到 `<EOS>` 后停止；也会设置最大生成长度，防止模型一直不输出 `<EOS>`。

### 25.3 如何从概率中选择 token

常见解码策略：

| 策略 | 做法 | 特点 |
|---|---|---|
| Greedy Search | 每步选择概率最大的 token | 快，但容易错过整体更优序列 |
| Beam Search | 每步保留若干条高分候选序列 | 翻译常用，计算更大 |
| Temperature | 用温度调节概率分布尖锐程度 | 温度低更确定，温度高更多样 |
| Top-k | 只在概率最高的 (k) 个 token 中采样 | 限制低概率噪声 |
| Top-p | 在累计概率达到 (p) 的最小候选集合中采样 | 候选数量可动态变化 |

翻译通常强调准确性，常用 Greedy 或 Beam Search；开放式文本生成更常使用 Temperature、Top-k、Top-p。

---

## 26. KV Cache：为什么生成时不用重复计算全部历史

在自回归生成中，过去 token 的 Key 和 Value 一旦计算完成，在后续步骤不会变化。

如果每生成一个 token 都重新计算全部历史，会产生大量重复计算。因此可缓存各 Decoder 层过去位置的 K、V：

```text
第1步：计算 token 1 的 K、V，放入缓存
第2步：只计算 token 2 的 Q、K、V，并读取 token 1 的缓存
第3步：只计算 token 3 的 Q、K、V，并读取 token 1～2 的缓存
……
```

KV Cache 的效果：

- 显著减少重复计算；
- 提高逐 token 解码速度；
- 但缓存会随层数、序列长度、头数和 (d_k/d_v) 增长，占用显存。

在 Encoder–Decoder 模型中，交叉注意力的 Encoder K、V 也可预先计算并复用，因为源句 memory 不变。

KV Cache 主要优化推理，不改变模型训练得到的数学函数。

---

## 27. Encoder 自注意力、Decoder 自注意力、交叉注意力对比

| 类型 | Q 来自哪里 | K、V 来自哪里 | 是否因果遮挡 | 权重矩阵形状 |
|---|---|---|---|---|
| Encoder Self-Attention | Encoder 当前层输入 | Encoder 当前层输入 | 通常否 | (L_s\times L_s) |
| Decoder Masked Self-Attention | Decoder 当前层输入 | Decoder 当前层输入 | 是 | (L_t\times L_t) |
| Cross-Attention | Decoder 隐藏状态 | Encoder 输出 | 源句不做因果遮挡 | (L_t\times L_s) |

三者使用的是同一套缩放点积注意力思想，差别主要在 Q、K、V 的来源以及 Mask。

---

## 28. 原始 Transformer、BERT、GPT 的关系

### 28.1 原始 Transformer

```text
Encoder + Decoder
```

适合机器翻译等“输入一个序列，输出另一个序列”的任务。

### 28.2 BERT：Encoder-only

BERT 主要保留 Encoder：

```text
双向 Self-Attention + FFN
```

它通常允许一个 token 同时查看左边和右边上下文，经典预训练任务是 Masked Language Modeling。适合理解、分类、抽取等任务。

### 28.3 GPT：Decoder-only

GPT 主要使用带因果 Mask 的 Transformer Block：

```text
Causal Self-Attention + FFN
```

它没有经典 Encoder–Decoder 架构中的交叉注意力层；提示词和已经生成的文本放在同一序列中，通过因果注意力逐 token 预测后续内容。

### 28.4 Encoder–Decoder 模型

T5、BART 等模型仍使用 Encoder–Decoder 思路，适合翻译、摘要和文本到文本任务。

> [!important] “Decoder-only”不等于直接照搬原始 Decoder
> 原始 Decoder 包含 Cross-Attention；典型 GPT Block 没有独立 Encoder，因此通常只有 Causal Self-Attention 与 FFN。

---

## 29. 计算复杂度与长序列问题

标准全注意力需要构造：

$$
A\in\mathbb{R}^{L\times L}
$$

因此注意力分数在序列长度方向的时间和显存成本通常呈二次增长：

$$
O(L^2)
$$

当上下文很长时，(L\times L) 注意力矩阵会成为主要瓶颈之一。

常见优化方向包括：

- FlashAttention：减少显存访问和中间张量开销，数学结果仍是精确注意力；
- 稀疏注意力或滑动窗口注意力；
- 分块注意力；
- 线性注意力近似；
- Grouped-Query Attention（GQA）和 Multi-Query Attention（MQA），减少推理时 KV Cache；
- 上下文压缩、检索增强等系统方案。

---

## 30. 常见实现细节

### 30.1 Batch 维与 Head 维

真实代码中常见形状变化：

```text
输入：        [B, L, d_model]
Q/K/V 投影： [B, L, h × d_k]
拆成多头：   [B, h, L, d_k]
注意力分数： [B, h, L, L]
每头输出：   [B, h, L, d_v]
合并多头：   [B, L, h × d_v]
输出投影：   [B, L, d_model]
```

不同库可能把维度排列成 `[L, B, d_model]` 等形式，阅读代码时应先检查接口约定。

### 30.2 Mask 的广播

Mask 可能以如下形状出现：

```text
[B, 1, 1, L]       Padding Mask
[1, 1, L, L]       Causal Mask
[B, 1, L, L]       合并后的 Mask
```

它们通过广播应用到 `[B, h, L_q, L_k]` 的注意力分数上。

### 30.3 数值稳定性

Softmax 实现通常会先减去每行最大值：

$$
\operatorname{Softmax}(x_j)=
\frac{e^{x_j-\max(x)}}{\sum_k e^{x_k-\max(x)}}
$$

这样不会改变结果，却可以降低指数溢出的风险。

### 30.4 参数与激活不是一回事

- 参数：(W^Q,W^K,W^V,W^O)、FFN 权重、Embedding、LayerNorm 参数等，会被优化器更新。
- 激活：Q、K、V、注意力权重、隐藏状态等，是一次前向传播根据当前输入计算出的中间结果。

### 30.5 序列长度可以变化

权重矩阵主要作用在特征维，例如 (d_{model}\rightarrow d_k)，并不要求固定 (L)。所以同一模型通常可以处理不同长度的序列，只要不超过模型、位置编码和显存允许的最大上下文长度。

---

## 31. 一段接近代码的完整伪流程

```python
# ---------- Encoder ----------
src_ids = tokenizer(source_text)
src_mask = make_padding_mask(src_ids)

x = token_embedding(src_ids) + position_encoding(src_ids)
x = dropout(x)

for encoder_layer in encoder_layers:
    # 下面写的是原论文 Post-LN 形式
    attn_out = multi_head_self_attention(
        query=x,
        key=x,
        value=x,
        mask=src_mask,
    )
    x = layer_norm(x + dropout(attn_out))

    ffn_out = ffn(x)
    x = layer_norm(x + dropout(ffn_out))

encoder_memory = x

# ---------- Decoder ----------
decoder_input = shift_right(target_ids)  # 开头加入 BOS，移除最后一个标签
tgt_padding_mask = make_padding_mask(decoder_input)
causal_mask = make_causal_mask(decoder_input)

y = target_embedding(decoder_input) + position_encoding(decoder_input)
y = dropout(y)

for decoder_layer in decoder_layers:
    masked_attn_out = multi_head_self_attention(
        query=y,
        key=y,
        value=y,
        mask=tgt_padding_mask + causal_mask,
    )
    y = layer_norm(y + dropout(masked_attn_out))

    cross_attn_out = multi_head_attention(
        query=y,
        key=encoder_memory,
        value=encoder_memory,
        mask=src_mask,
    )
    y = layer_norm(y + dropout(cross_attn_out))

    ffn_out = ffn(y)
    y = layer_norm(y + dropout(ffn_out))

# ---------- 预测与训练 ----------
logits = output_projection(y)  # [B, L_t, vocab_size]
loss = cross_entropy(logits, target_labels, ignore_pad=True)

optimizer.zero_grad()
loss.backward()
clip_grad_norm_if_needed()
optimizer.step()
```

---

## 32. 最容易混淆的十个问题

### 32.1 Q、K、V 是三份不同的输入吗？

Self-Attention 中它们来自同一个输入 (X)，但经过三组不同的可学习投影，所以数值和作用不同。

### 32.2 (W^Q,W^K,W^V) 是注意力权重吗？

不是。它们是训练得到的参数矩阵。注意力权重是每次根据当前输入动态计算的 (A)。

### 32.3 为什么 (QK^T) 后还要乘 (V)？

(QK^T) 只计算“看谁以及看多少”；乘 (V) 才真正把相应内容汇总回来。

### 32.4 每行 Softmax 还是每列 Softmax？

通常对每个 Query 所对应的 Key 维做 Softmax，也就是让每一行对所有 Key 的权重和为 1。

### 32.5 多头是把句子分成多段吗？

不是。通常每个头都能看到整个允许访问的序列，只是使用不同投影子空间分析关系。

### 32.6 多头是否把原始向量机械切成几段？

更准确地说，是先通过可学习线性投影产生多个头的 Q、K、V，再 reshape 为多头。它不等于不经学习地切片原始 Embedding。

### 32.7 为什么注意力输出的序列长度不变？

一个 Query 产生一个输出。输入有 (L_q) 个 Query，输出就有 (L_q) 行。

### 32.8 为什么残差能够相加？

因为 (W^O) 和 FFN 的第二层会把输出维度恢复为 (d_{model})，与子层输入形状一致。

### 32.9 Encoder 能看后面的词，Decoder 为什么不能？

Encoder 的任务通常是理解完整输入，完整源句已经存在；自回归 Decoder 训练时若看到未来目标 token，就会提前知道答案，导致训练和推理条件不一致。

### 32.10 Attention 中的“权重”是模型参数吗？

注意力矩阵 (A) 是当前输入动态产生的激活，不是固定参数；产生它所用的 (W^Q,W^K) 才是被训练更新的参数。

---

## 33. 最终总复习

### 33.1 Encoder 一层

```text
输入 X
  ↓
多头双向自注意力
  ↓
残差 + Dropout + LayerNorm
  ↓
FFN
  ↓
残差 + Dropout + LayerNorm
  ↓
下一层 Encoder
```

### 33.2 Decoder 一层

```text
目标前缀表示
  ↓
带 Causal Mask 的多头自注意力
  ↓
残差 + Dropout + LayerNorm
  ↓
读取 Encoder 输出的交叉注意力
  ↓
残差 + Dropout + LayerNorm
  ↓
FFN
  ↓
残差 + Dropout + LayerNorm
  ↓
下一层 Decoder
```

### 33.3 注意力核心

$$
\boxed{
\operatorname{Attention}(Q,K,V)
=
\operatorname{Softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}+M
\right)V
}
$$

可以拆成四句话：

1. (QK^T)：计算每个 Query 与所有 Key 的匹配程度；
2. 除以 \(\sqrt{d_k}\)：稳定数值尺度；
3. Mask + Softmax：屏蔽非法位置，并把分数变成权重；
4. 乘 (V)：按权重把真正的内容信息汇总回来。

### 33.4 Transformer 的本质

```text
Embedding：把符号变成向量
Position：告诉模型顺序
Attention：让 token 之间交换信息
FFN：在每个 token 内部进一步加工
Residual + LayerNorm：让深层训练更稳定
Output Projection：把隐藏向量变成词表分数
Loss + Backprop：让所有可学习参数逐渐变好
Autoregressive Decoding：推理时逐 token 生成
```

如果只记一句话，可以记为：

> Transformer 反复执行“让每个 token 有选择地读取其他 token，再加工自己的表示”，最后把加工后的表示用于理解或预测下一个 token。
