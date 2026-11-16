# -*- coding: utf-8 -*-
"""把训练好的 .pt 导出成 ONNX,并用 **项目自己的 cv2** 读回来验证。

为什么导出 ONNX 而不是直接带 torch 进发布包:
- 项目 venv 里的 opencv 已经带 `dnn` 模块,`readNetFromONNX` 就能跑,单帧几毫秒;
- torch 光 wheel 就 2GB+,而发布包要发给用户双击运行;
- ONNX 也不锁死推理设备,以后想换 GPU/换机器都不用改代码。

用法(用训练用的那个解释器):
    & .\\venv_train\\Scripts\\python.exe nn\\export.py
    & .\\venv_train\\Scripts\\python.exe nn\\export.py --ckpt nn\\checkpoints\\best.pt
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
sys.path.insert(0, APP)

from nn import data as D          # noqa: E402

DEFAULT_CKPT = os.path.join(HERE, "checkpoints", "best.pt")
DEFAULT_OUT = os.path.join(HERE, "coindet.onnx")


def export(ckpt_path, out_path, opset=17):
    import torch
    from nn.model import CoinDet

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = CoinDet()
    model.load_state_dict(ck["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    dummy = torch.zeros(1, 3, D.INPUT_H, D.INPUT_W, dtype=torch.float32)
    # 用**旧版导出器**(dynamo=False):torch 2.10 默认走 dynamo/torch.export,
    # 那条路要 onnxscript 且对自定义结构容易炸;旧版 TorchScript 路径对
    # Conv/GroupNorm/ReLU 这种纯前馈网络最稳。
    kw = dict(
        input_names=["image"], output_names=["hm", "cls", "wh", "off"],
        opset_version=opset, do_constant_folding=True,
    )
    try:
        torch.onnx.export(model, dummy, out_path, dynamo=False, **kw)
    except TypeError:                     # 老版本 torch 没有 dynamo 这个参数
        torch.onnx.export(model, dummy, out_path, **kw)
    size = os.path.getsize(out_path) / 1024.0
    print("导出 %s (%.0f KB, epoch %s, val_loss %s)"
          % (out_path, size, ck.get("epoch"), ck.get("val_loss")))
    return model


def verify(model, onnx_path):
    """用 cv2.dnn 跑一遍 ONNX,和 torch 的输出逐元素比。"""
    import cv2
    net = cv2.dnn.readNetFromONNX(onnx_path)
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (D.INPUT_H, D.INPUT_W, 3), dtype=np.uint8)

    blob = cv2.dnn.blobFromImage(img, 0.00392156862745098, (D.INPUT_W, D.INPUT_H),
                                 (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())
    print("cv2.dnn 输出:", [(n, o.shape) for n, o in zip(net.getUnconnectedOutLayersNames(), outs)])

    import torch
    with torch.no_grad():
        x = torch.from_numpy(blob).float()
        t = model(x)
    for name, o, tv in zip(net.getUnconnectedOutLayersNames(), outs, t):
        a = o.reshape(-1).astype(np.float32)
        b = tv.numpy().reshape(-1)
        err = float(np.abs(a - b).max())
        print("  %-4s 最大误差 %.2e  %s" % (name, err, "OK" if err < 1e-3 else "差异偏大"))

    # CPU 单帧耗时
    for _ in range(3):
        net.setInput(blob)
        net.forward()
    t0 = time.time()
    n = 20
    for _ in range(n):
        net.setInput(blob)
        net.forward()
    ms = (time.time() - t0) / n * 1000
    print("cv2.dnn CPU 单帧 %.1f ms" % ms)
    return ms


def sanity(samples=None):
    """在留出的验证图上跑一遍,看检测出来的框和标注框对不对得上。"""
    import cv2
    onnx = os.path.join(HERE, "coindet.onnx")
    net = cv2.dnn.readNetFromONNX(onnx)
    import vision.imgio as imgio
    files = sorted(os.listdir(os.path.join(APP, "shoot")))
    if samples is None:
        samples = [f for f in files if f.lower().endswith((".png", ".jpg")) and f in D.VAL_FILES]
    for f in samples:
        img = imgio.imread(os.path.join(APP, "shoot", f))
        if img is None:
            continue
        canvas = cv2.resize(img, (D.CANVAS_W, D.CANVAS_H), interpolation=cv2.INTER_AREA)
        blob = cv2.dnn.blobFromImage(D.preprocess(canvas), 1.0, (D.INPUT_W, D.INPUT_H),
                                     (0, 0, 0), swapRB=False, crop=False)
        net.setInput(blob)
        hm, cls, wh, off = net.forward(net.getUnconnectedOutLayersNames())
        dets = D.decode(hm, np.asarray(cls)[0], wh[0], off[0])
        txt = ", ".join("%s %.2f" % (D.CLASSES[c], s) for c, s, *_ in dets[:6])
        print("  %-40s %s" % (f[:40], txt or "(空)"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--skip-sanity", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.ckpt):
        raise SystemExit("找不到 %s —— 先跑 nn\\train.py" % args.ckpt)
    model = export(args.ckpt, args.out, args.opset)
    verify(model, args.out)
    if not args.skip_sanity:
        print("验证集抽样:")
        sanity()


if __name__ == "__main__":
    main()