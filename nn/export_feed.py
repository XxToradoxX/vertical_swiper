# -*- coding: utf-8 -*-
"""把 nn\\feednet.pt 导出成 nn\\feednet.onnx,并用 cv2.dnn 读回来逐元素校验。

为什么要有 ONNX 这份:
- 软件平时用 Scroller 环境(只有 opencv,没有 torch),`vision\\feedcheck.py`
  走 cv2.dnn 这条 CPU 路;
- 装了 torch+CUDA 的环境(项目里的 `env_gpu`)则可以直接走 GPU,
  两条路必须给出一模一样的分数,所以导出后立刻比对。

用法(用带 torch 的环境):
    env_gpu\\python.exe nn\\export_feed.py
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

from nn.feed import INPUT_H, INPUT_W, MEAN, STD, build_model    # noqa: E402

DEFAULT_CKPT = os.path.join(HERE, "feednet.pt")
DEFAULT_OUT = os.path.join(HERE, "feednet.onnx")


def export(ckpt_path, out_path, opset=13):
    import torch
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model()
    model.load_state_dict(ck["state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    dummy = torch.zeros(1, 3, INPUT_H, INPUT_W, dtype=torch.float32)
    kw = dict(input_names=["image"], output_names=["logit"],
              opset_version=opset, do_constant_folding=True)
    try:
        torch.onnx.export(model, dummy, out_path, dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(model, dummy, out_path, **kw)
    meta = ck.get("meta", {})
    print("导出 %s (%.0f KB,epoch %s,val_acc %s)"
          % (out_path, os.path.getsize(out_path) / 1024.0,
             meta.get("epoch"), meta.get("val_acc")))
    return model


def blob_of(img_bgr):
    """和 nn/feed.py 的 preprocess 等价,但用 blobFromImage 一步到位。"""
    import cv2
    x = cv2.resize(img_bgr, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)
    x = (x.astype(np.float32) / 255.0 - MEAN) / STD
    return cv2.dnn.blobFromImage(x, 1.0, (INPUT_W, INPUT_H), (0, 0, 0),
                                 swapRB=False, crop=False)


def verify(model, onnx_path):
    import cv2
    import torch
    net = cv2.dnn.readNetFromONNX(onnx_path)
    rng = np.random.default_rng(0)
    worst = 0.0
    for i in range(4):
        img = rng.integers(0, 255, (INPUT_H * 2, INPUT_W * 2, 3), dtype=np.uint8)
        blob = blob_of(img)
        net.setInput(blob)
        o = float(np.asarray(net.forward()).reshape(-1)[0])
        with torch.no_grad():
            t = float(model(torch.from_numpy(blob).float()).reshape(-1)[0])
        worst = max(worst, abs(o - t))
    print("cv2.dnn vs torch:最大误差 %.2e %s"
          % (worst, "OK" if worst < 1e-3 else "*** 差异偏大 ***"))

    for _ in range(3):
        net.setInput(blob)
        net.forward()
    t0 = time.time()
    n = 50
    for _ in range(n):
        net.setInput(blob)
        net.forward()
    print("cv2.dnn CPU 单帧 %.2f ms" % ((time.time() - t0) / n * 1000))
    return worst


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args(argv)

    if not os.path.exists(args.ckpt):
        raise SystemExit("找不到 %s —— 先跑 nn\\train_feed.py" % args.ckpt)
    model = export(args.ckpt, args.out, args.opset)
    if not args.skip_verify:
        verify(model, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())