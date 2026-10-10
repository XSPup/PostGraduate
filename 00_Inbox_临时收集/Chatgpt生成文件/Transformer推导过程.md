下面我们真的让一个很小的 Transformer **从输入走到输出**。建议你第一遍只看每一步的“输入是什么、输出是什么”，第二遍再跟着数字计算。

我们要演示：

\[ \text{输入：}[\text{红},\text{灯}] \quad\longrightarrow\quad \text{输出：}[\text{停},\texttt{<EOS>}] \]

`<EOS>` 表示“生成结束”，最终给人看的文字是“停”。

> **这个模型是教学用的缩小版**：一层 Encoder、一层 Decoder、每种注意力一个头；向量宽度为 3。参数是为了演示而事先选定的，并非真实训练所得。矩阵并非单位矩阵。完整的数据路径与原始 Encoder–Decoder Transformer 一致。

## 0. 先认清整条路线

#chatgpt-mermaid-_r_1m9_{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;font-size:14px;fill:rgb(255, 255, 255);}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#chatgpt-mermaid-_r_1m9_ .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#chatgpt-mermaid-_r_1m9_ .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#chatgpt-mermaid-_r_1m9_ .error-icon{fill:rgba(54, 54, 54, 0.96);}#chatgpt-mermaid-_r_1m9_ .error-text{fill:rgb(255, 255, 255);stroke:rgb(255, 255, 255);}#chatgpt-mermaid-_r_1m9_ .edge-thickness-normal{stroke-width:1px;}#chatgpt-mermaid-_r_1m9_ .edge-thickness-thick{stroke-width:3.5px;}#chatgpt-mermaid-_r_1m9_ .edge-pattern-solid{stroke-dasharray:0;}#chatgpt-mermaid-_r_1m9_ .edge-thickness-invisible{stroke-width:0;fill:none;}#chatgpt-mermaid-_r_1m9_ .edge-pattern-dashed{stroke-dasharray:3;}#chatgpt-mermaid-_r_1m9_ .edge-pattern-dotted{stroke-dasharray:2;}#chatgpt-mermaid-_r_1m9_ .marker{fill:rgba(255, 255, 255, 0.498);stroke:rgba(255, 255, 255, 0.498);}#chatgpt-mermaid-_r_1m9_ .marker.cross{stroke:rgba(255, 255, 255, 0.498);}#chatgpt-mermaid-_r_1m9_ svg{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;font-size:14px;}#chatgpt-mermaid-_r_1m9_ p{margin:0;}#chatgpt-mermaid-_r_1m9_ .label{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:rgb(255, 255, 255);}#chatgpt-mermaid-_r_1m9_ .cluster-label text{fill:rgb(255, 255, 255);}#chatgpt-mermaid-_r_1m9_ .cluster-label span{color:rgb(255, 255, 255);}#chatgpt-mermaid-_r_1m9_ .cluster-label span p{background-color:transparent;}#chatgpt-mermaid-_r_1m9_ .label text,#chatgpt-mermaid-_r_1m9_ span{fill:rgb(255, 255, 255);color:rgb(255, 255, 255);}#chatgpt-mermaid-_r_1m9_ .node rect,#chatgpt-mermaid-_r_1m9_ .node circle,#chatgpt-mermaid-_r_1m9_ .node ellipse,#chatgpt-mermaid-_r_1m9_ .node polygon,#chatgpt-mermaid-_r_1m9_ .node path{fill:rgb(26, 40, 61);stroke:rgb(31, 78, 148);stroke-width:1px;}#chatgpt-mermaid-_r_1m9_ .rough-node .label text,#chatgpt-mermaid-_r_1m9_ .node .label text,#chatgpt-mermaid-_r_1m9_ .image-shape .label,#chatgpt-mermaid-_r_1m9_ .icon-shape .label{text-anchor:middle;}#chatgpt-mermaid-_r_1m9_ .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#chatgpt-mermaid-_r_1m9_ .rough-node .label,#chatgpt-mermaid-_r_1m9_ .node .label,#chatgpt-mermaid-_r_1m9_ .image-shape .label,#chatgpt-mermaid-_r_1m9_ .icon-shape .label{text-align:center;}#chatgpt-mermaid-_r_1m9_ .node.clickable{cursor:pointer;}#chatgpt-mermaid-_r_1m9_ .root .anchor path{fill:rgba(255, 255, 255, 0.498)!important;stroke-width:0;stroke:rgba(255, 255, 255, 0.498);}#chatgpt-mermaid-_r_1m9_ .arrowheadPath{fill:rgba(255, 255, 255, 0.498);}#chatgpt-mermaid-_r_1m9_ .edgePath .path{stroke:rgba(255, 255, 255, 0.498);stroke-width:1px;}#chatgpt-mermaid-_r_1m9_ .flowchart-link{stroke:rgba(255, 255, 255, 0.498);fill:none;}#chatgpt-mermaid-_r_1m9_ .edgeLabel{background-color:rgb(24, 24, 24);text-align:center;}#chatgpt-mermaid-_r_1m9_ .edgeLabel p{background-color:rgb(24, 24, 24);}#chatgpt-mermaid-_r_1m9_ .edgeLabel rect{opacity:0.5;background-color:rgb(24, 24, 24);fill:rgb(24, 24, 24);}#chatgpt-mermaid-_r_1m9_ .labelBkg{background-color:rgba(24, 24, 24, 0.5);}#chatgpt-mermaid-_r_1m9_ .cluster rect{fill:rgba(54, 54, 54, 0.96);stroke:rgba(255, 255, 255, 0.082);stroke-width:1px;}#chatgpt-mermaid-_r_1m9_ .cluster text{fill:rgb(255, 255, 255);}#chatgpt-mermaid-_r_1m9_ .cluster span{color:rgb(255, 255, 255);}#chatgpt-mermaid-_r_1m9_ div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;font-size:12px;background:rgba(54, 54, 54, 0.96);border:1px solid rgba(255, 255, 255, 0.082);border-radius:2px;pointer-events:none;z-index:100;}#chatgpt-mermaid-_r_1m9_ .flowchartTitleText{text-anchor:middle;font-size:18px;fill:rgb(255, 255, 255);}#chatgpt-mermaid-_r_1m9_ rect.text{fill:none;stroke-width:0;}#chatgpt-mermaid-_r_1m9_ .icon-shape,#chatgpt-mermaid-_r_1m9_ .image-shape{background-color:rgb(24, 24, 24);text-align:center;}#chatgpt-mermaid-_r_1m9_ .icon-shape p,#chatgpt-mermaid-_r_1m9_ .image-shape p{background-color:rgb(24, 24, 24);padding:2px;}#chatgpt-mermaid-_r_1m9_ .icon-shape .label rect,#chatgpt-mermaid-_r_1m9_ .image-shape .label rect{opacity:0.5;background-color:rgb(24, 24, 24);fill:rgb(24, 24, 24);}#chatgpt-mermaid-_r_1m9_ .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#chatgpt-mermaid-_r_1m9_ .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#chatgpt-mermaid-_r_1m9_ .node .neo-node{stroke:rgb(31, 78, 148);}#chatgpt-mermaid-_r_1m9_ [data-look="neo"].node rect,#chatgpt-mermaid-_r_1m9_ [data-look="neo"].cluster rect,#chatgpt-mermaid-_r_1m9_ [data-look="neo"].node polygon{stroke:url(#chatgpt-mermaid-_r_1m9_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_1m9_ [data-look="neo"].swimlane.cluster rect{filter:none;}#chatgpt-mermaid-_r_1m9_ [data-look="neo"].node path{stroke:url(#chatgpt-mermaid-_r_1m9_-gradient);stroke-width:1px;}#chatgpt-mermaid-_r_1m9_ [data-look="neo"].node .outer-path{filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_1m9_ [data-look="neo"].node .neo-line path{stroke:rgb(31, 78, 148);filter:none;}#chatgpt-mermaid-_r_1m9_ [data-look="neo"].node circle{stroke:url(#chatgpt-mermaid-_r_1m9_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_1m9_ [data-look="neo"].node circle .state-start{fill:#000000;}#chatgpt-mermaid-_r_1m9_ [data-look="neo"].icon-shape .icon{fill:url(#chatgpt-mermaid-_r_1m9_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_1m9_ [data-look="neo"].icon-shape .icon-neo path{stroke:url(#chatgpt-mermaid-_r_1m9_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_1m9_ .node text{font-size:14px;font-weight:600;letter-spacing:normal;fill:rgb(153, 206, 255);}#chatgpt-mermaid-_r_1m9_ .edgeLabels text{font-size:13px;font-weight:600;letter-spacing:-0.08px;fill:rgb(153, 206, 255);}#chatgpt-mermaid-_r_1m9_ .node tspan[font-weight="normal"],#chatgpt-mermaid-_r_1m9_ .edgeLabels tspan[font-weight="normal"]{font-weight:600;}#chatgpt-mermaid-_r_1m9_ .edgeLabel .label rect{opacity:1;rx:13px;ry:13px;fill:rgb(0, 14, 26);stroke:rgb(26, 62, 95);stroke-width:1px;}#chatgpt-mermaid-_r_1m9_ .node rect,#chatgpt-mermaid-_r_1m9_ .node circle,#chatgpt-mermaid-_r_1m9_ .node ellipse,#chatgpt-mermaid-_r_1m9_ .node polygon,#chatgpt-mermaid-_r_1m9_ .node path{fill:rgb(0, 40, 77);stroke:rgba(255, 255, 255, 0.1);stroke-width:1px;}#chatgpt-mermaid-_r_1m9_ .node rect{rx:16px;ry:16px;}#chatgpt-mermaid-_r_1m9_ .node.mermaid-decision .label-container{fill:rgb(0, 14, 26);stroke:rgb(26, 62, 95);stroke-dasharray:2,2;}#chatgpt-mermaid-_r_1m9_ .edgePaths .flowchart-link{stroke:rgba(255, 255, 255, 0.498);stroke-width:1px;stroke-linecap:round;stroke-linejoin:round;}#chatgpt-mermaid-_r_1m9_ .marker{fill:rgba(255, 255, 255, 0.498);stroke:rgba(255, 255, 255, 0.498);}#chatgpt-mermaid-_r_1m9_ :root{--mermaid-font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;}输入：红 灯Embedding + 位置Encoder 自注意力 + FFN输入的表示 H已生成的输出：BOS、停Decoder 带遮挡的自注意力交叉注意力：读取 HFFN + 词表预测下一词：停、EOS

这里的 **FFN** 是对每个位置分别运行的小型前馈网络。注意，Decoder 每预测一个词，都可以通过交叉注意力查看 Encoder 处理过的整句输入。

所有向量写成**行向量**，例如 `[1.1, 0.2, -0.1]`。保留三位小数，所以手算与表格末位可能有微小差别。

---

## 1. 把输入词变成数字：Embedding 和位置

模型不能直接拿汉字做矩阵乘法。我们给每个词查一个长度为 3 的 Embedding，再加上它所在位置的向量：

|输入位置|词|Embedding \(E\)|位置向量 \(P\)|相加后的 \(X=E+P\)|
|---|---|---|---|---|
|1|红|\([1,\ 0.2,\ 0]\)|\([0.1,\ 0,\ -0.1]\)|\([1.1,\ 0.2,\ -0.1]\)|
|2|灯|\([0.1,\ 1,\ 0.3]\)|\([-0.1,\ 0.1,\ 0.05]\)|\([0,\ 1.1,\ 0.35]\)|

把两个词上下叠起来：

\[ X= \begin{bmatrix} 1.1&0.2&-0.1\\ 0&1.1&0.35 \end{bmatrix} \qquad \text{形状为 }2\times3 \]

**为什么加位置？** 如果没有位置，注意力能看到“红”和“灯”，却缺少可靠的信息来区分“红灯”和“灯红”的顺序。

## 2. Encoder 自注意力：让“红”和“灯”互相看

先从 \(X\) 算出三组东西：

\[ Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V \]

这三个字母可以先这样记：

- **Q（Query）**：我正在找什么？
- **K（Key）**：我有哪些特征可供别人匹配？
- **V（Value）**：别人关注我后，实际拿走什么信息？

例如，这次用的三个权重矩阵是：

\[ W_Q=\begin{bmatrix}.8&.1\\.2&.9\\.1&.2\end{bmatrix}, \quad W_K=\begin{bmatrix}.7&.2\\.1&.8\\.3&-.1\end{bmatrix}, \quad W_V=\begin{bmatrix}.9&.1\\.2&.7\\.1&.4\end{bmatrix} \]

乘完得到：

\[ Q=\begin{bmatrix}.910&.270\\.255&1.060\end{bmatrix},\quad K=\begin{bmatrix}.760&.390\\.215&.845\end{bmatrix},\quad V=\begin{bmatrix}1.020&.210\\.255&.910\end{bmatrix} \]

每个矩阵的**第一行属于“红”，第二行属于“灯”**。

### 2.1 算 \(S\)：谁和谁有多匹配

\[ S=\frac{QK^\mathsf T}{\sqrt{2}} = \begin{bmatrix} .563&.300\\ .429&.672 \end{bmatrix} \]

\(S\) 是 **score，匹配分数**。行表示“谁在发问”，列表示“看谁”。例如左上角是“红看红”：

\[ S_{\text{红,红}} =\frac{[.910,.270]\cdot[.760,.390]}{\sqrt2} =\frac{.910(.760)+.270(.390)}{\sqrt2} \approx .563 \]

而“红看灯”的分数约为 `.300`。除以 \(\sqrt2\) 是为了控制点积分数的大小，让后面的 Softmax 不容易变得过于极端。

### 2.2 算 \(A\)：把分数变成注意力比例

对 \(S\) 的**每一行分别做 Softmax**：

\[ A=\operatorname{softmax}_{\text{逐行}}(S) = \begin{bmatrix} .566&.434\\ .440&.560 \end{bmatrix} \]

第一行的意思是：“红”这次取信息，约 **56.6% 来自红自身，43.4% 来自灯**。第二行是“灯”取信息的比例。每一行加起来都是 1。

### 2.3 算上下文，再投影回长度 3

\[ C=AV = \begin{bmatrix} .688&.514\\ .591&.602 \end{bmatrix} \]

例如第一行就是：

\[ C_{\text{红}} =.566[1.020,.210]+.434[.255,.910] \approx[.688,.514] \]

所以 **\(S\) 是分数，\(A\) 是比例，\(C\) 是按比例取回的信息**。有些讲义把最后的注意力结果叫 \(O\)；这里为了分清“加权取回”和“输出投影”，写成：

\[ O=CW_O,\qquad W_O=\begin{bmatrix}.4&.2&.1\\.1&.3&.5\end{bmatrix} \]

得到：

\[ O= \begin{bmatrix} .326&.292&.326\\ .297&.299&.360 \end{bmatrix} \]

注意：\(A\) 的形状是 \(2\times2\)，\(C\) 是 \(2\times2\)，而 \(O\) 又回到 \(2\times3\)，这样才能与原输入 \(X\) 相加。

## 3. Encoder 的残差、归一化、FFN

原始 Transformer 的一层还没结束。注意力输出要先做**残差连接与 LayerNorm**：

\[ U=\operatorname{LN}(X+O) = \begin{bmatrix} 1.382&-.433&-.949\\ -1.111&1.313&-.202 \end{bmatrix} \]

以“红”为例，进入归一化前是：

\[ X_{\text{红}}+O_{\text{红}} =[1.1,.2,-.1]+[.326,.292,.326] =[1.426,.492,.226] \]

LayerNorm 对**这一行的三个数**求均值和标准差，再把它们标准化；因此得到约 `[1.382, -.433, -.949]`。这里把 LayerNorm 自身可学习的缩放、平移参数设为 1 和 0，方便计算。

接着是逐位置 FFN：

\[ F=\operatorname{ReLU}(UW_1)W_2 = \begin{bmatrix} .129&.049&.079\\ -.152&.292&.205 \end{bmatrix} \]

例如“红”经过 FFN 第一层的四个数和 ReLU 后为 `[.363, 0, .165, .198]`；负数被 ReLU 变成 0，然后再投影回长度 3。最后再做一次残差和归一化：

\[ H=\operatorname{LN}(U+F) = \begin{bmatrix} 1.388&-.457&-.931\\ -1.174&1.270&-.095 \end{bmatrix} \]

**Encoder 到这里完成了。** \(H\) 的两行仍分别对应“红”和“灯”，但每一行已经结合了另一词的信息。Decoder 稍后要读取这个 \(H\)。

---

## 4. Decoder：先理解为什么出现 `<BOS>` 和“停”

要预测的输出是：

\[ [\text{停},\texttt{<EOS>}] \]

在展示一次完整的**训练时并行前向计算**时，Decoder 的输入要向右挪一位：

|Decoder 输入位置|已给 Decoder 的词|此位置要预测|
|---|---|---|
|1|`<BOS>`|停|
|2|停|`<EOS>`|

`<BOS>` 是“开始生成”的标记。这里第二行出现“停”是因为我们在展示训练时的并行计算；**实际生成时，模型起初只拿到 `<BOS>`，预测出“停”之后才会把“停”送回去**。我们会在最后单独走一遍实际生成过程。

给 Decoder 输入也加上位置向量：

\[ Y= \begin{bmatrix} .5&.1&.7\\ .7&.3&.45 \end{bmatrix} \]

第一行来自 `<BOS>`，第二行来自“停”。

## 5. Decoder 的第一次注意力：带遮挡的自注意力

Decoder 从 \(Y\) 计算**自己的** \(Q,K,V\)。它们使用另一组权重，与 Encoder 的权重不同。这次结果是：

\[ Q_d=\begin{bmatrix}.390&.380\\.525&.400\end{bmatrix}, \quad K_d=\begin{bmatrix}.210&.300\\.395&.410\end{bmatrix}, \quad V_d=\begin{bmatrix}.580&.270\\.685&.400\end{bmatrix} \]

计算分数后，必须给第一行的“未来位置”加遮挡：

\[ S_d= \begin{bmatrix} .139&-\infty\\ .163&.263 \end{bmatrix} \quad\xrightarrow{\text{逐行 Softmax}}\quad A_d= \begin{bmatrix} 1&0\\ .475&.525 \end{bmatrix} \]

**这里是整个例子最关键的防泄漏设计：**

- 第一个位置要预测“停”，它目前只有 `<BOS>`，**不能偷看第二位置输入的“停”**。所以第一行第二列是 \(-\infty\)，Softmax 后权重为 0。
- 第二个位置要预测 `<EOS>`，此时“停”已经生成，可以看 `<BOS>` 和“停”。

将 \(A_dV_d\) 投影回长度 3，做残差和 LayerNorm 后得到：

\[ U_d= \begin{bmatrix} .471&-1.390&.919\\ 1.244&-1.205&-.039 \end{bmatrix} \]

目前 Decoder **只整理了自己已经生成的内容**，还需要去读取输入“红 灯”。

## 6. Decoder 的第二次注意力：交叉注意力

这一步要特别分清三组向量**来自哪里**：

\[ Q_c=U_dW_Q^{(c)},\qquad K_c=HW_K^{(c)},\qquad V_c=HW_V^{(c)} \]

- **Q 来自 Decoder**：当前生成位置想从输入里找什么。
- **K、V 来自 Encoder 的 \(H\)**：输入词提供可匹配的特征和可取走的信息。

在本例中：

\[ Q_c= \begin{bmatrix} .233&-.971\\ .369&-.591 \end{bmatrix}, \quad K_c= \begin{bmatrix} .509&-.415\\ -.343&.616 \end{bmatrix}, \quad V_c= \begin{bmatrix} .601&-.135\\ -.597&.644 \end{bmatrix} \]

继续按同一套 \(S\rightarrow A\rightarrow AV\) 计算：

\[ S_c=\frac{Q_cK_c^\mathsf T}{\sqrt2} = \begin{bmatrix} .369&-.479\\ .306&-.347 \end{bmatrix} \]\[ A_c= \begin{bmatrix} .700&.300\\ .658&.342 \end{bmatrix} \]

第一行表示：在**预测“停”的位置**，Decoder 此次从 Encoder 两个位置取信息的比例约为“红”70%、“灯”30%。这只是我们这组教学参数的结果，不能据此认定真实模型总按这个比例理解“红灯”。

第一行实际取回的二维信息约为：

\[ .700[.601,-.135]+.300[-.597,.644] \approx[.242,.099] \]

投影回三维、加残差、做 LayerNorm 后：

\[ Z_d= \begin{bmatrix} .472&-1.391&.918\\ 1.235&-1.214&-.021 \end{bmatrix} \]

至此，Decoder 的每个位置都同时具有**自己此前生成的信息**与**输入“红 灯”的信息**。

## 7. Decoder 的 FFN 与词表预测

Decoder 也要经过自己的 FFN、残差和 LayerNorm：

\[ D= \begin{bmatrix} .566&-1.405&.839\\ 1.274&-1.169&-.105 \end{bmatrix} \]

最后将每行乘以输出矩阵，得到对词表中各个词的 **logit（尚未归一化的分数）**，再做 Softmax。为突出重点，下表列出“停”和 `<EOS>`；这个玩具词表里其余三个符号的分数均为 \(-8\)。

|位置|输入 Decoder 的当前词|“停”的 logit|`<EOS>` 的 logit|Softmax 后最可能的词|
|---|---|---|---|---|
|1|`<BOS>`|\(9.000\)|\(4.654\)|**停，约 98.7%**|
|2|停|\(4.654\)|\(9.000\)|**`<EOS>`，约 98.7%**|

例如输出层对“停”和 `<EOS>` 使用下面的权重（两列分别对应这两个候选词）：

\[ W_{\text{out,两列}}= \begin{bmatrix} 3.398&7.644\\ -8.432&-7.012\\ 5.035&-.632 \end{bmatrix}, \qquad b=[-9,-9] \]

第一个位置预测“停”的分数来自一次普通的矩阵乘法：

\[ [.566,-1.405,.839]\! \begin{bmatrix}3.398\\-8.432\\5.035\end{bmatrix} -9 \approx 9.000 \]

这一步**没有直接输出文字**：模型先产生各词的分数，选出词表中的“停”，我们再按词表把它显示成汉字。

## 8. 真正生成时，时间顺序是什么？

上面的两行可以在训练时一起算，因为遮挡确保第一行看不到第二行。实际推理则是一轮一轮进行：

|轮次|Encoder 输入|当前 Decoder 输入|只读取最后位置的预测|接下来|
|---|---|---|---|---|
|1|红、灯|`<BOS>`|**停**|把“停”接到 Decoder 输入后|
|2|红、灯|`<BOS>`、停|**`<EOS>`**|停止生成|

因此最终输出词串是 `[停, <EOS>]`，显示给用户的是 **“停”**。第一轮的数值与上面并行计算的第一行一致：因果遮挡让它根本用不到第二行的“停”。

---

### 最后把容易混的符号放在一张表里

|符号|它是什么|本例形状|
|---|---|---|
|\(X\)|输入词的 Embedding 加位置|\(2\times3\)|
|\(Q\)|发起查询的向量|\(2\times2\)|
|\(K\)|供查询匹配的向量|\(2\times2\)|
|\(V\)|匹配后实际取回的向量|\(2\times2\)|
|\(S=QK^\mathsf T/\sqrt{d_k}\)|匹配**分数**|\(2\times2\)|
|\(A=\operatorname{softmax}(S+\text{Mask})\)|注意力**比例**|\(2\times2\)|
|\(AV\)|按比例混合得到的内容|\(2\times2\)|
|\(O=(AV)W_O\)|注意力模块的输出，投影回模型宽度|\(2\times3\)|
|\(H\)|Encoder 最终交给 Decoder 的输入表示|\(2\times3\)|

**你可以用一句话检验自己是否真的理解：** Encoder 把“红 灯”变成可供读取的 \(H\)；Decoder 从 `<BOS>` 开始，每轮先看自己已经生成的词，再用交叉注意力读取 \(H\)，最后对词表打分，依次选出“停”和 `<EOS>`。