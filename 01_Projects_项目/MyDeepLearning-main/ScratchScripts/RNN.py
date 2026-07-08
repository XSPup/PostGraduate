"""
最简 RNN 入门示例（不依赖任何第三方库）。

这份代码目标是：
1. 用最少代码看懂 RNN 的时间展开。
2. 把“方块图”对应到可运行的实现。
3. 每一步都打印中间结果，便于手算核对。

ASCII 结构图（时间展开）：

	x1 ----> [RNN cell] ----> h1 ----> y1
			   ^
			   |
	h0 --------+

	x2 ----> [RNN cell] ----> h2 ----> y2
			   ^
			   |
	h1 --------+

	x3 ----> [RNN cell] ----> h3 ----> y3
			   ^
			   |
	h2 --------+

每个时间步 t 的公式：
	a_t = Wxh * x_t + Whh * h_(t-1) + b_h
	h_t = tanh(a_t)
	o_t = Why * h_t + b_y
	y_t = sigmoid(o_t)

符号含义：
	x_t: 第 t 步输入
	h_t: 第 t 步隐藏状态（记忆）
	y_t: 第 t 步输出
"""

from math import exp, tanh


def sigmoid(x):
	# 把任意实数压到 (0, 1)，常用于二分类概率输出
	return 1.0 / (1.0 + exp(-x))


def matvec_mul(matrix, vector):
	"""矩阵(m x n)乘向量(n)，返回向量(m)。"""
	out = []
	for row in matrix:
		s = 0.0
		for a, b in zip(row, vector):
			s += a * b
		out.append(s)
	return out


def vec_add(a, b):
	# 向量逐元素相加
	return [x + y for x, y in zip(a, b)]


def vec_apply_tanh(v):
	# 向量逐元素应用 tanh
	return [tanh(x) for x in v]


def rnn_forward(xs, params, h0):
	"""
	对完整序列执行一次 vanilla RNN 前向传播。

	xs: 输入序列，形状近似 (T, input_size)
	h0: 初始隐藏状态，形状近似 (hidden_size)
	"""
	# 读取参数（权重和偏置）
	Wxh = params["Wxh"]
	Whh = params["Whh"]
	bh = params["bh"]
	Why = params["Why"]
	by = params["by"]

	# h_prev 表示上一个时间步的隐藏状态（t=1 时是 h0）
	h_prev = h0[:]
	all_h = []
	all_y = []

	# 沿时间维度循环：一次循环就是一个时间步
	for t, x_t in enumerate(xs, start=1):
		# a_t = Wxh*x_t + Whh*h_prev + b_h
		part_x = matvec_mul(Wxh, x_t)
		part_h = matvec_mul(Whh, h_prev)
		a_t = vec_add(vec_add(part_x, part_h), bh)

		# h_t = tanh(a_t)
		h_t = vec_apply_tanh(a_t)

		# o_t = Why*h_t + b_y, then y_t = sigmoid(o_t)
		o_t = vec_add(matvec_mul(Why, h_t), by)
		y_t = [sigmoid(v) for v in o_t]

		# 保存每个时间步的结果，便于后续分析
		all_h.append(h_t)
		all_y.append(y_t)
		h_prev = h_t

		print(f"Step {t}")
		print(f"  x_t = {x_t}")
		print(f"  h_t = {[round(v, 4) for v in h_t]}")
		print(f"  y_t = {[round(v, 4) for v in y_t]}")
		print("-" * 45)

	return all_h, all_y


if __name__ == "__main__":
	# 一个极小示例：
	# input_size = 2, hidden_size = 2, output_size = 1
	# 权重是手动写死的，目的是“可读、可手算”，不是训练得到的
	params = {
		"Wxh": [
			[0.5, -0.1],
			[0.3, 0.8],
		],
		"Whh": [
			[0.2, 0.0],
			[-0.3, 0.4],
		],
		"bh": [0.0, 0.0],
		"Why": [
			[1.0, -1.0],
		],
		"by": [0.0],
	}

	# 共有 3 个时间步，每步输入 x_t 有 2 个特征
	xs = [
		[1.0, 0.0],
		[0.0, 1.0],
		[1.0, 1.0],
	]

	h0 = [0.0, 0.0]

	# 运行前向传播并打印每一步细节
	print("RNN sequence demo (no external library)")
	print("=" * 45)
	hs, ys = rnn_forward(xs, params, h0)

	print("Final hidden state h_T:", [round(v, 4) for v in hs[-1]])
	print("Final output y_T:", [round(v, 4) for v in ys[-1]])

