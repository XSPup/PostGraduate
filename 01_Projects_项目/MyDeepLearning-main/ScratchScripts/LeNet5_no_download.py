"""
LeNet-5 全流程离线复现（不下载数据集）

说明：
1) 这份代码复现的是 LeNet-5 的经典结构思想：
   Conv -> Tanh -> AvgPool -> Conv -> Tanh -> AvgPool -> Conv -> FC -> FC
2) 原始 LeNet-5 训练使用的是手写数字数据（历史上是支票/邮编等场景）。
3) 为了满足“完全离线、不下载数据”的要求，这里用程序合成数字图像来训练。

你可以直接运行：
    python LeNet5_no_download.py
"""

import random
from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:
    raise SystemExit(
        "未检测到 PyTorch。\n"
        "本脚本不下载数据，但需要本地已有 torch。\n"
        "可先安装：pip install torch\n"
        f"原始错误: {exc}"
    )


# 7 段数码管风格：
#   ---0---
#  |       |
#  1       2
#  |       |
#   ---3---
#  |       |
#  4       5
#  |       |
#   ---6---
SEGMENTS = {
    0: (0, 1, 2, 4, 5, 6),
    1: (2, 5),
    2: (0, 2, 3, 4, 6),
    3: (0, 2, 3, 5, 6),
    4: (1, 2, 3, 5),
    5: (0, 1, 3, 5, 6),
    6: (0, 1, 3, 4, 5, 6),
    7: (0, 2, 5),
    8: (0, 1, 2, 3, 4, 5, 6),
    9: (0, 1, 2, 3, 5, 6),
}


def _draw_h(img, row, col1, col2, thickness, value):
    img[row : row + thickness, col1:col2] = value


def _draw_v(img, row1, row2, col, thickness, value):
    img[row1:row2, col : col + thickness] = value


def _shift_2d(img, dy, dx):
    """把图像平移 (dy, dx)，越界区域补 0。"""
    h, w = img.shape
    out = torch.zeros_like(img)

    src_y1 = max(0, -dy)
    src_y2 = min(h, h - dy)
    src_x1 = max(0, -dx)
    src_x2 = min(w, w - dx)

    dst_y1 = max(0, dy)
    dst_y2 = dst_y1 + (src_y2 - src_y1)
    dst_x1 = max(0, dx)
    dst_x2 = dst_x1 + (src_x2 - src_x1)

    if src_y1 < src_y2 and src_x1 < src_x2:
        out[dst_y1:dst_y2, dst_x1:dst_x2] = img[src_y1:src_y2, src_x1:src_x2]
    return out


def render_digit_32x32(label, rng, training=True):
    """生成一个 32x32 的单通道数字图像（值域 0~1）。"""
    img = torch.zeros((32, 32), dtype=torch.float32)

    # 定义 7 段的几何位置
    top = 4
    mid = 14
    bottom = 24
    left = 6
    right = 22
    thickness = 3
    height_up = 12
    height_down = 12

    # 亮度加一点随机性，防止模型过拟合固定强度
    bright = rng.uniform(0.8, 1.0) if training else 0.9

    on = set(SEGMENTS[label])
    if 0 in on:
        _draw_h(img, top, left, right, thickness, bright)
    if 1 in on:
        _draw_v(img, top, top + height_up, left, thickness, bright)
    if 2 in on:
        _draw_v(img, top, top + height_up, right, thickness, bright)
    if 3 in on:
        _draw_h(img, mid, left, right, thickness, bright)
    if 4 in on:
        _draw_v(img, mid, mid + height_down, left, thickness, bright)
    if 5 in on:
        _draw_v(img, mid, mid + height_down, right, thickness, bright)
    if 6 in on:
        _draw_h(img, bottom, left, right, thickness, bright)

    # 训练集加入轻微随机平移和噪声，提高泛化能力
    if training:
        dx = rng.randint(-2, 2)
        dy = rng.randint(-2, 2)
        img = _shift_2d(img, dy, dx)

        noise = torch.randn_like(img) * 0.08
        img = img + noise

    img = img.clamp(0.0, 1.0)
    return img.unsqueeze(0)  # (1, 32, 32)


class SyntheticDigitsDataset(Dataset):
    """离线合成数字数据集，不依赖网络下载。"""

    def __init__(self, size, seed, training=True):
        self.size = size
        self.training = training
        self.rng = random.Random(seed)

        # 预先生成标签，便于可复现
        self.labels = [self.rng.randint(0, 9) for _ in range(size)]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # 为每个样本构造独立随机源，确保同一 idx 可复现
        local_rng = random.Random(idx * 10007 + self.labels[idx] * 97 + (1 if self.training else 0))
        y = self.labels[idx]
        x = render_digit_32x32(y, local_rng, training=self.training)
        return x, y


class LeNet5(nn.Module):
    """
    经典 LeNet-5 风格（简化到常见 10 类数字分类）：

    输入:  (N, 1, 32, 32)
    C1:    Conv(1->6, 5x5)      -> (N, 6, 28, 28)
    S2:    AvgPool(2x2, s=2)    -> (N, 6, 14, 14)
    C3:    Conv(6->16, 5x5)     -> (N,16, 10, 10)
    S4:    AvgPool(2x2, s=2)    -> (N,16,  5,  5)
    C5:    Conv(16->120, 5x5)   -> (N,120, 1,  1)
    F6:    Linear(120->84)
    Out:   Linear(84->10)
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 120, kernel_size=5),
            nn.Tanh(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


@dataclass
class TrainConfig:
    epochs: int = 8
    batch_size: int = 64
    lr: float = 0.03
    momentum: float = 0.9
    train_size: int = 12000
    val_size: int = 2000
    seed: int = 42


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)

    return total_loss / total, correct / total


def train():
    cfg = TrainConfig()

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print("数据来源: 本地程序合成（不下载）")

    train_set = SyntheticDigitsDataset(size=cfg.train_size, seed=cfg.seed, training=True)
    val_set = SyntheticDigitsDataset(size=cfg.val_size, seed=cfg.seed + 1, training=False)

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False)

    model = LeNet5().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)

    print("\n开始训练 LeNet-5...\n")
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            seen += x.size(0)

        train_loss = running_loss / seen
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc * 100:.2f}%"
        )

    # 展示几个样本预测
    model.eval()
    sample_x, sample_y = next(iter(val_loader))
    sample_x = sample_x[:10].to(device)
    sample_y = sample_y[:10]
    with torch.no_grad():
        pred = model(sample_x).argmax(dim=1).cpu()

    print("\n样本预测（前 10 个）：")
    for i in range(10):
        print(f"  idx={i:02d} | gt={int(sample_y[i])} | pred={int(pred[i])}")

    torch.save(model.state_dict(), "lenet5_offline_synth.pt")
    print("\n模型已保存: lenet5_offline_synth.pt")


if __name__ == "__main__":
    train()
