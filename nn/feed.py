# -*- coding: utf-8 -*-
"""挂机用的轻量判页网络:只回答一个问题 —— 「现在是不是还在多多视频信息流」。

为什么不用现成的 `coindet.onnx`:
- 那个网络是 3 个下采样 + 3 个空洞卷积的 CenterNet,处理 448x800,单帧 12 ms,
  但它回答的是"18 类控件各在哪",判"是不是信息流"属于杀鸡用牛刀;
- 挂机时每 3~4 秒要看一眼界面,**单价必须压到 1 ms 级**,不然省下来的时间
  还不如现在直接跑整帧模板匹配(3.6 s)。

所以这里是一个极小的二分类网络(输入 128x228,约 4 万参数):
    Conv 3x3 /2 → 3 组深度可分离卷积 → 全局平均池化 → 1 个 logit
- 判据是"结构"而不是"内容":信息流页面一定有**底部导航栏**和**右侧竖排按钮**
  (头像/点赞/评论/分享/唱片),任一视频内容都被它们框住。网络只要学到这两条,
  换成什么视频、什么手机、什么夜色都认得;
- 归一化统一用 GroupNorm(和 `nn/model.py` 一样,绕开某些平台 BN 的坑);
- 训练侧用 torch(CUDA 可选),发布侧 `vision/feedcheck.py` 用 cv2.dnn 跑 ONNX,
  两条路的预处理都在本文件里定义,保证一致。
"""

import numpy as np

# 模型输入尺寸(宽, 高),与训练/导出/推理三处共用
INPUT_W, INPUT_H = 128, 228

# 归一化:手机截图整体偏暗,统一减均值除方差,免得网络先学"亮度"
MEAN = 0.42
STD = 0.24

# 判"是信息流"的默认阈值。tools\bench_feed.py 在全部 67 张样本上量的:
#   正样本(信息流)最低 0.601、中位 0.896;负样本(金币页/弹窗/落地页/商品页)
#   最高 0.046、中位 0.001。两侧差了十几倍,阈值放在中间最稳。
# 取 0.40 而不是 0.60:把信息流误判成"不是"(漏报)要多花一次整帧识别(约 1 s),
# 而把别的页面误判成信息流只会多上滑一下 —— 后者便宜得多。
DEFAULT_THR = 0.40


def preprocess(bgr):
    """BGR 图(任意尺寸)-> (1,3,INPUT_H,INPUT_W) float32。

    推理侧(cv2.dnn)和训练侧(torch)都调这一个函数,避免两条路差一个
    BGR/RGB 或者差一个 /255(踩过:同一张图一边 0.97 一边 0.03)。
    """
    import cv2
    small = cv2.resize(bgr, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)
    x = small.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return np.transpose(x, (2, 0, 1))[None, ...].copy()


def build_model(width=32, depth=(32, 64, 128, 192)):
    """返回一个未初始化的 torch 模型。没装 torch 的机器不该 import 这个函数。"""
    import torch.nn as nn

    def gn(c):
        return 8 if c % 8 == 0 else 1

    def conv_gn(cin, cout, k=3, s=1, groups=1):
        return nn.Sequential(
            nn.Conv2d(cin, cout, k, stride=s, padding=k // 2, groups=groups, bias=False),
            nn.GroupNorm(gn(cout), cout),
            nn.ReLU(inplace=True),
        )

    def dw(cin, cout, s):
        """深度可分离:3x3 逐通道 + 1x1 升维,参数量只有普通卷积的 1/8。"""
        return nn.Sequential(
            conv_gn(cin, cin, 3, s=s, groups=cin),
            conv_gn(cin, cout, 1),
        )

    d = depth
    return nn.Sequential(
        conv_gn(3, d[0], 3, s=2),        # /2  64x114
        dw(d[0], d[1], s=2),             # /4  32x57
        dw(d[1], d[2], s=2),             # /8  16x29
        dw(d[2], d[3], s=2),             # /16 8x15
        conv_gn(d[3], d[3], 1),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(d[3], 1),
    )