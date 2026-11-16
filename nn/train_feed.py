# -*- coding: utf-8 -*-
"""训练挂机判页网络(「现在还在多多视频信息流吗」)。

用法(用项目里的 GPU 环境,自带 torch+cu121):

    env_gpu\\python.exe nn\\train_feed.py                 # 默认 120 轮
    env_gpu\\python.exe nn\\train_feed.py --epochs 300 --lr 0.001

产物:nn\\feednet.pt(torch 权重,GPU 推理用),再跑 nn\\export_feed.py 导出 ONNX。

数据只有几十张真机截图,**正样本尤其少**,所以这里不追求大网络,而是把
增广做狠:随机仿射(缩放/旋转/平移)+ 亮度对比度抖动 + 噪声 + 随机遮挡。
输入只有 128x228,视频内容早就糊成色块了,网络只能去抓"底部导航栏 +
右侧竖排按钮"这种结构线索 —— 这正是我们要它学的东西,也是它能跨手机的原因。
"""

import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from nn.feed import INPUT_H, INPUT_W, MEAN, STD, build_model   # noqa: E402

OUT = os.path.join(ROOT, "nn", "feednet.pt")
CACHE = os.path.join(ROOT, "nn", "feed_cache.npz")


def load():
    """直接读缓存。这里不 import nn.feed_data,免得把 vision(要 opencv)拖进来 ——
    训练环境只装了 torch,没有 cv2。"""
    if not os.path.exists(CACHE):
        raise SystemExit("缺少 %s,先用 Scroller 环境跑 python nn\\feed_data.py" % CACHE)
    data = np.load(CACHE, allow_pickle=True)
    return data["x"], data["y"], [str(p) for p in data["paths"]]


def to_tensor(x_uint8, device):
    """(N,284,160,3) uint8 BGR -> (N,3,284,160) float,已归一化。"""
    x = torch.from_numpy(np.ascontiguousarray(x_uint8)).to(device).float() / 255.0
    x = x.permute(0, 3, 1, 2)                       # NHWC -> NCHW
    x = (x - MEAN) / STD
    return x


def augment(x, gen):
    """训练增广。x 已是归一化后的 (B,3,H,W)。"""
    b = x.shape[0]
    s = 1.0 + (torch.rand(b, 1, generator=gen, device=x.device) * 0.16 - 0.08)
    ang = (torch.rand(b, 1, generator=gen, device=x.device) * 2 - 1) * math.radians(2.5)
    tx = (torch.rand(b, 1, generator=gen, device=x.device) * 2 - 1) * 0.03
    ty = (torch.rand(b, 1, generator=gen, device=x.device) * 2 - 1) * 0.03
    cos, sin = torch.cos(ang), torch.sin(ang)
    theta = torch.cat([s * cos, -s * sin, tx, s * sin, s * cos, ty], dim=1).view(b, 2, 3)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="border", align_corners=False)
    x = F.interpolate(x, size=(INPUT_H, INPUT_W), mode="area")

    # 亮度/对比度(手机自动亮度、不同壁纸都会变)
    gain = 1.0 + (torch.rand(b, 1, 1, 1, generator=gen, device=x.device) * 0.3 - 0.15)
    bias = torch.rand(b, 1, 1, 1, generator=gen, device=x.device) * 0.12 - 0.06
    x = x * gain + bias

    # 噪声(视频压缩痕迹)
    x = x + torch.randn_like(x) * 0.02

    # 随机遮挡:模拟弹窗、手指、字幕挡住半个屏幕
    for i in range(b):
        if torch.rand(1, generator=gen, device=x.device).item() < 0.3:
            h = int(INPUT_H * (0.15 + 0.35 * torch.rand(1, generator=gen, device=x.device).item()))
            w = int(INPUT_W * (0.15 + 0.45 * torch.rand(1, generator=gen, device=x.device).item()))
            y0 = int(torch.randint(0, max(1, INPUT_H - h), (1,), generator=gen, device=x.device).item())
            x0 = int(torch.randint(0, max(1, INPUT_W - w), (1,), generator=gen, device=x.device).item())
            x[i, :, y0:y0 + h, x0:x0 + w] = 0.0
    return x


def plain(x):
    """验证/推理用:只缩放到网络输入尺寸,不做任何增广。"""
    return F.interpolate(x, size=(INPUT_H, INPUT_W), mode="area")


def split(y, seed=0):
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    nv_pos = max(2, len(pos) // 4)
    nv_neg = max(6, len(neg) // 5)
    val = np.concatenate([pos[:nv_pos], neg[:nv_neg]])
    tr = np.concatenate([pos[nv_pos:], neg[nv_neg:]])
    rng.shuffle(tr)
    rng.shuffle(val)
    return tr, val


def evaluate(model, x, y, device, bs=64):
    model.eval()
    scores = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            s = model(plain(x[i:i + bs])).squeeze(1)
            scores.append(torch.sigmoid(s).float().cpu())
    sc = torch.cat(scores).numpy()
    pred = (sc >= 0.5).astype(np.int64)
    acc = float((pred == y).mean())
    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    return acc, sc, tp, fn, fp


def main(argv=None):
    ap = argparse.ArgumentParser(description="训练挂机判页网络")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)

    x_u8, y, paths = load()
    y = np.asarray(y, dtype=np.int64)
    tr, va = split(y, args.seed)
    print("设备 %s | 样本 %d(信息流 %d)| 训练 %d / 验证 %d"
          % (device, len(y), int(y.sum()), len(tr), len(va)))

    xtr = to_tensor(x_u8[tr], device)
    ytr = torch.from_numpy(y[tr].astype(np.float32)).to(device)
    xva = to_tensor(x_u8[va], device)
    yva = y[va]

    model = build_model().to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print("参数量 %d" % n_par)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = torch.nn.BCEWithLogitsLoss()
    amp = (device == "cuda")

    best = (-1.0, None, 0.0)
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(xtr), device=device)
        tot, cnt = 0.0, 0
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            xb = augment(xtr[idx], gen)
            yb = ytr[idx]
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=amp):
                out = model(xb).squeeze(1)
                loss = lossf(out, yb)
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
            cnt += len(idx)
        sched.step()
        if ep % 10 == 0 or ep == 1:
            acc, sc, tp, fn, fp = evaluate(model, xva, yva, device)
            print("ep %3d  loss %.4f  验证 acc %.3f  (TP %d FN %d FP %d)"
                  % (ep, tot / cnt, acc, tp, fn, fp))
            if acc > best[0]:
                best = (acc, {k: v.detach().clone() for k, v in model.state_dict().items()}, ep)

    acc, sd, ep = best
    model.load_state_dict(sd)
    model.eval()
    torch.save({
        "state": sd,
        "meta": {"input_w": INPUT_W, "input_h": INPUT_H,
                 "mean": MEAN, "std": STD, "epoch": ep, "val_acc": acc,
                 "n_train": len(tr), "n_val": len(va)},
    }, args.out)

    # 把训练集也过一遍,看清"分得开不开"(阈值应该落在一段无人的空隙里)
    tr_acc, tr_sc, _, _, _ = evaluate(model, xtr, ytr.cpu().numpy(), device)
    print("\n最佳验证 acc %.3f(%d 轮)-> %s" % (acc, ep, args.out))
    print("训练集 acc %.3f" % tr_acc)
    print("正样本分数 min %.3f / 负样本分数 max %.3f"
          % (float(tr_sc[ytr.cpu().numpy() == 1].min()) if (ytr == 1).any() else -1,
             float(tr_sc[ytr.cpu().numpy() == 0].max()) if (ytr == 0).any() else -1))
    return 0


if __name__ == "__main__":
    sys.exit(main())