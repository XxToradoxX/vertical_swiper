# -*- coding: utf-8 -*-
"""检测网络:输入一张 448x800 的画布图,输出 56x100 的格子级预测。

为什么是这种结构(而不是把 MAA 的 YOLO 抄过来):
- 目标全是**文字块**(行标题、气泡、按钮),没有明显轮廓,轻量网络 + 足够的
  感受野就够了 —— 这里用三个空洞卷积把感受野撑到 ~90 像素,正好覆盖"一行字
  加上它上下的留白";
- 推理端要用 **cv2.dnn 跑 ONNX**(包里已经有 opencv,不想再塞一个 onnxruntime),
  所以只用 Conv/BN/ReLU/Concat 这些最稳的算子,**不做上采样**:
  直接在第 8 个下采样步长处出结果,靠空洞卷积补感受野;
- 参数量 ~12 万,CPU 上单帧几十毫秒,比现在 12 张模板逐张匹配还快。

输出 4 个头(CenterNet 风格):
  hm   1 x GH x GW   有东西的置信度(logit)
  cls  10 x GH x GW  类别(logit)
  wh   2 x GH x GW   框的宽高(单位:格)
  off  2 x GH x GW   中心点相对格子左上角的偏移(单位:格)
"""

import torch
import torch.nn as nn

NUM_CLASSES = 10


def conv_bn(cin, cout, k=3, s=1, d=1):
    """Conv + 归一化 + ReLU。

    归一化用 **GroupNorm 而不是 BatchNorm**:这台机器是 AMD 显卡 + Windows 上的
    ROCm,`torch.batch_norm` 会走 MIOpen 的 hipRTC 现场编译,而它找不到 C++ 标准
    库头(`fatal error: 'type_traits' file not found`)→ 直接抛
    `RuntimeError: miopenStatusUnknownError`。GroupNorm 走原生 aten,不经 MIOpen,
    训练完全正常;ONNX 导出后 cv2.dnn 也认 GroupNormalization 算子。
    """
    p = d * (k - 1) // 2
    groups = 8 if cout % 8 == 0 else 1
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, stride=s, padding=p, dilation=d, bias=False),
        nn.GroupNorm(groups, cout),
        nn.ReLU(inplace=True),
    )


class CoinDet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, width=48):
        super().__init__()
        self.stem = nn.Sequential(                    # /2
            conv_bn(3, 16, s=2),
            conv_bn(16, 16),
        )
        self.s4 = nn.Sequential(                      # /4
            conv_bn(16, 32, s=2),
            conv_bn(32, 32),
        )
        self.s8 = nn.Sequential(                      # /8  <- 输出步长
            conv_bn(32, width, s=2),
            conv_bn(width, width),
        )
        self.ctx = nn.Sequential(
            conv_bn(width, width, d=2),
            conv_bn(width, width, d=4),
            conv_bn(width, width, d=8),
        )
        # 注意:conv_bn 内部已经带 ReLU,这里不能再套一个 nn.ReLU(inplace=True) ——
        # 连着两次原地 ReLU 会把第一次的中间结果覆盖掉,反向传播时报
        # "one of the variables needed for gradient computation has been modified
        #  by an inplace operation"(踩过)。
        self.head = conv_bn(width, 64, k=1)

        self.hm = nn.Conv2d(64, 1, 1)
        self.cls = nn.Conv2d(64, num_classes, 1)
        self.wh = nn.Conv2d(64, 2, 1)
        self.off = nn.Conv2d(64, 2, 1)

        # 初始化:热力图头的偏置取负值,一开始不要满屏"有东西"(CenterNet 的做法)
        nn.init.constant_(self.hm.bias, -4.0)
        for m in (self.hm, self.cls, self.wh, self.off):
            nn.init.normal_(m.weight, std=0.001)

    def forward(self, x):
        x = self.stem(x)
        x = self.s4(x)
        x = self.s8(x)
        x = self.ctx(x)
        x = self.head(x)
        return self.hm(x), self.cls(x), self.wh(x), self.off(x)


def focal_loss(pred_logit, target, alpha=2.0, beta=4.0):
    """CenterNet 的焦点损失:正样本少,负样本多,普通 BCE 会被背景淹没。"""
    pred = torch.sigmoid(pred_logit).clamp(1e-4, 1 - 1e-4)
    pos = target >= 1.0
    neg = ~pos
    pos_loss = -((1 - pred) ** alpha) * torch.log(pred) * pos
    neg_loss = -((1 - target) ** beta) * (pred ** alpha) * torch.log(1 - pred) * neg
    npos = pos.float().sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / npos


def total_loss(out, batch):
    hm, cls, wh, off = out
    t_hm, t_cls, t_wh, t_off, mask = batch
    l_hm = focal_loss(hm, t_hm)

    m = mask.unsqueeze(1)                                   # (B,1,GH,GW)
    ce = (-(t_cls * torch.log_softmax(cls, dim=1)) * m).sum() / m.sum().clamp(min=1.0)
    wp = m.expand_as(t_wh)
    l_wh = (torch.abs(wh - t_wh) * wp).sum() / (wp.sum().clamp(min=1.0) * 2.0)
    l_off = (torch.abs(off - t_off) * wp).sum() / (wp.sum().clamp(min=1.0) * 2.0)
    return l_hm + ce + 0.5 * l_wh + 1.0 * l_off, {
        "hm": float(l_hm.detach()),
        "cls": float(ce.detach()),
        "wh": float(l_wh.detach()),
        "off": float(l_off.detach()),
    }