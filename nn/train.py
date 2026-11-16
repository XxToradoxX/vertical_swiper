# -*- coding: utf-8 -*-
"""训练检测器。

    venv_train\\Scripts\\python.exe nn\\train.py --epochs 30 --batch 16

数据只有 25 张真机截图,所以每个 epoch 都是"随机增广后的同一批图"——
这不是过拟合,增广本身就把位置/尺度/颜色/模糊/遮挡都变了,网络学到的是
"这种字块长什么样",而不是"它在这张图的第几行"。

验证集是留出的 5 张(见 nn/data.py 的 VAL_FILES),只用来观察,不参与训练。
"""

import argparse
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nn import data as D                                    # noqa: E402
from nn.model import CoinDet, total_loss                    # noqa: E402

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class CoinDataset(Dataset):
    def __init__(self, samples, pool=None, train=True, per_epoch=3200):
        self.samples = samples
        self.pool = pool or {}
        self.train = train
        self.per_epoch = per_epoch

    def __len__(self):
        return self.per_epoch if self.train else len(self.samples)

    def __getitem__(self, i):
        rng = np.random.default_rng((torch.initial_seed() + i) % (2 ** 32))
        if self.train:
            # 一半用真机图(位置就是真的,但只有 25 张),一半用合成图
            # (位置/尺度/底色全随机,弥补真机图不够)。两种混着训,网络既不会
            # 只记住真图,也不会只学会合成数据的味道。
            if self.pool and rng.random() < 0.5:
                img, boxes = D.synthesize(self.pool, self.samples, rng)
                img, boxes = D.augment(img, boxes, rng)
            else:
                _f, img, boxes = self.samples[rng.integers(0, len(self.samples))]
                img, boxes = D.augment(img, boxes, rng)
        else:
            _f, img, boxes = self.samples[i]
        x = D.preprocess(img)
        hm, cls, wh, off, mask = D.encode(boxes)
        t = [torch.from_numpy(v) for v in (hm, cls, wh, off, mask)]
        return torch.from_numpy(x).permute(2, 0, 1), t


def collate(batch):
    xs = torch.stack([b[0] for b in batch])
    ts = [torch.stack([b[1][k] for b in batch]) for k in range(5)]
    return xs, ts


def evaluate(model, loader, device):
    model.eval()
    total = 0.0
    n = 0
    tp = fp = fn = 0
    with torch.no_grad():
        for x, t in loader:
            x = x.to(device, non_blocking=True)
            t = [v.to(device, non_blocking=True) for v in t]
            with torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else _null():
                out = model(x)
                loss, _ = total_loss(out, t)
            total += float(loss)
            n += 1
            hm = torch.sigmoid(out[0])
            pred_pos = hm[0] > 0.35
            gt_pos = t[0][0] > 0.5
            tp += int((pred_pos & gt_pos).sum())
            fp += int((pred_pos & ~gt_pos).sum())
            fn += int((~pred_pos & gt_pos).sum())
    model.train()
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    return total / max(1, n), prec, rec


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--per-epoch", type=int, default=3200)
    ap.add_argument("--out", default=os.path.join(APP_DIR, "nn", "checkpoints"))
    ap.add_argument("--resume", default="")
    args = ap.parse_args()

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("设备: %s" % device)
    if device.type == "cuda":
        print("  %s, %.1f GB" % (torch.cuda.get_device_name(0),
                                 torch.cuda.get_device_properties(0).total_memory / 1e9))

    print("收集样本…")
    samples = D.collect_samples(verbose=True)
    train = [s for s in samples if s[0] not in D.VAL_FILES]
    val = [s for s in samples if s[0] in D.VAL_FILES]
    print("训练 %d 张 / 验证 %d 张;训练集标注总数 %d" %
          (len(train), len(val), sum(len(s[2]) for s in train)))
    if not val:
        raise SystemExit("验证集为空,检查 data.VAL_FILES 的文件名")

    train_ds = CoinDataset(train, pool=D._crop_pool(samples, D.load_spec()),
                           train=True, per_epoch=args.per_epoch)
    val_ds = CoinDataset(val, train=False)
    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=0, collate_fn=collate, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0,
                        collate_fn=collate)

    model = CoinDet().to(device)
    nparam = sum(p.numel() for p in model.parameters())
    print("参数量: %.1f 万" % (nparam / 1e4))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * len(train_ld), pct_start=0.25)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        print("从 %s 继续(epoch %d)" % (args.resume, ck.get("epoch", -1)))

    os.makedirs(args.out, exist_ok=True)
    best = 1e9
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        run = 0.0
        seen = 0
        for it, (x, t) in enumerate(train_ld):
            x = x.to(device, non_blocking=True)
            t = [v.to(device, non_blocking=True) for v in t]
            with torch.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                out = model(x)
                loss, parts = total_loss(out, t)
            if not torch.isfinite(loss):
                # fp16 偶尔会溢出成 nan/inf:跳过这个 batch,别让权重被污染
                print("  ep%d %3d  跳过(loss 非有限值)" % (ep + 1, it + 1))
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += float(loss)
            seen += 1
            if (it + 1) % 20 == 0:
                els = time.time() - t0
                print("  ep%d %3d/%3d  loss %.3f (hm %.3f cls %.3f wh %.3f off %.3f) %.1fs"
                      % (ep + 1, it + 1, len(train_ld), run / seen, parts["hm"],
                         parts["cls"], parts["wh"], parts["off"], els))
                run = 0.0
                seen = 0
        vloss, prec, rec = evaluate(model, val_ld, device)
        print("[epoch %d] 验证 loss %.4f  hm-P %.3f  hm-R %.3f  用时 %.0fs"
              % (ep + 1, vloss, prec, rec, time.time() - t0))
        ck = {"model": model.state_dict(), "epoch": ep,
              "val_loss": vloss, "val_prec": prec, "val_rec": rec}
        torch.save(ck, os.path.join(args.out, "last.pt"))
        if vloss < best:
            best = vloss
            torch.save(ck, os.path.join(args.out, "best.pt"))
            print("  → 保存 best.pt (val_loss %.4f)" % vloss)

    print("完成,总用时 %.1f 分钟,best val_loss %.4f" % ((time.time() - t0) / 60, best))


if __name__ == "__main__":
    main()