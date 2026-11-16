# -*- coding: utf-8 -*-
"""把真机截图整理成「是不是信息流」的二分类数据集。

用法(用 Scroller 环境跑,需要 opencv):

    python nn\\feed_data.py --list      # 只打印,看看都收了哪些图、标了什么
    python nn\\feed_data.py            # 生成 nn\\feed_cache.npz

标注是**自动**的:直接用现有的 `Recognizer.classify()`(模板那条路,不用 NN,
快一倍),页面判成 feed 的就是正样本,其余(金币页、弹窗、直播、落地页、
商品页、首页……)全是负样本。这样模型学的就是"人肉规则当初判 feed 的那件事",
但只需要 1 ms 而不是 3.6 s —— 挂机时用它做粗筛,拿不准再退回整帧识别。

缓存里存的是 160x284 的缩略图(uint8),训练时在 GPU 上做仿射增广,
所以**训练侧不需要 opencv**,只要 torch。
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cv2                                        # noqa: E402  (只有生成缓存要用)

import vision.imgio as imgio                      # noqa: E402
from vision.canvas import Canvas                  # noqa: E402
from vision.recognizer import Recognizer          # noqa: E402

CACHE = os.path.join(ROOT, "nn", "feed_cache.npz")

# 缓存尺寸:比网络输入略大一圈,训练时靠仿射变换/缩放再裁到 128x228
CACHE_W, CACHE_H = 160, 284

# 自动标注偶尔会翻车(尤其"变暗的信息流 + 白色底部弹层"这种),这里手工纠正。
# 键是文件名,值是 1(信息流)/ 0(不是)/ None(直接丢弃,别污染训练集)。
OVERRIDES = {
    "coin3.png": 0,            # 现金账户弹层,自动标注误判成 feed
    "newphone_now.png": 1,     # 新机信息流,当时 ROI 还没改,识别不出入口
    "coin_new_clean.png": 0,   # 拼小圈页
    "feed_again.png": 0,       # 商品详情页
    # 这张上面被画了紫红色圈注("看00:12 得金币"),留着会让网络学"紫红=信息流"
    "紫色部分圈出来的是需要识别的部分.png": None,
}


# 调试渲染图(带识别框)和裁出来的小图都不能当训练样本
SKIP_PREFIX = ("band_", "bubble_", "row_", "entry_", "cmp_", "digits_", "diag_",
               "templates_", "gui_", "live_widget", "tap_pill", "after_tap2")


def candidates():
    """列出候选截图。只收"整屏竖图",裁出来的小图 / GUI 截图不要。"""
    out = []
    for sub in ("shoot", "debug"):
        d = os.path.join(ROOT, sub)
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if not name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                continue
            if sub == "debug" and (name[:1].isdigit() or name.startswith(SKIP_PREFIX)):
                continue                       # 040_unknown.png、diag_xxx 之类
            path = os.path.join(d, name)
            img = imgio.imread(path)
            if img is None:
                continue
            h, w = img.shape[:2]
            # 竖屏、且够大:1280x720 那种 GUI 截图、几百像素的小裁图都排除
            if w < 800 or h < 1400 or h <= w:
                continue
            out.append(path)
    return out


def label_one(rec, path):
    img = imgio.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    canvas = Canvas(w, h)
    can = canvas.normalize(img)
    view = rec.classify(can)
    small = cv2.resize(can, (CACHE_W, CACHE_H), interpolation=cv2.INTER_AREA)
    label = 1 if view.page == "feed" else 0
    if os.path.basename(path) in OVERRIDES:
        label = OVERRIDES[os.path.basename(path)]
        if label is None:
            return None
    return {
        "path": os.path.relpath(path, ROOT),
        "label": label,
        "page": view.page,
        "rows": [m for m in ("shop_video", "live", "water") if view.row(m) is not None],
        "img": small,
    }


def build(force=False):
    if os.path.exists(CACHE) and not force:
        data = np.load(CACHE, allow_pickle=True)
        return list(data["paths"]), data["y"]
    rec = Recognizer(use_nn=False)
    items = []
    for path in candidates():
        try:
            it = label_one(rec, path)
        except Exception as exc:                  # 单张坏图不该拖垮整个数据集
            print("[skip] %s: %s" % (path, exc))
            continue
        if it:
            items.append(it)
    if not items:
        raise SystemExit("没找到可用截图")
    y = np.array([it["label"] for it in items], dtype=np.int64)
    x = np.stack([it["img"] for it in items]).astype(np.uint8)
    paths = np.array([it["path"] for it in items])
    np.savez_compressed(CACHE, x=x, y=y, paths=paths)
    return list(paths), y


def load():
    """训练侧入口:返回 (x uint8 (N,284,160,3), y (N,), paths)。"""
    if not os.path.exists(CACHE):
        raise SystemExit("缺少 %s,先跑 python nn\\feed_data.py" % CACHE)
    data = np.load(CACHE, allow_pickle=True)
    return data["x"], data["y"], [str(p) for p in data["paths"]]


def main(argv=None):
    ap = argparse.ArgumentParser(description="生成挂机判页用的数据集")
    ap.add_argument("--list", action="store_true", help="只打印清单,不写缓存")
    ap.add_argument("--force", action="store_true", help="忽略缓存重新生成")
    args = ap.parse_args(argv)

    if args.list:
        rec = Recognizer(use_nn=False)
        total = pos = 0
        for path in candidates():
            it = label_one(rec, path)
            if not it:
                continue
            total += 1
            pos += it["label"]
            print("%-52s %-12s %s" % (it["path"], it["page"],
                                      "信息流" if it["label"] else ""))
        print("\n合计 %d 张:信息流 %d,其他 %d" % (total, pos, total - pos))
        return 0

    paths, y = build(force=args.force)
    pos = int(y.sum())
    print("已写入 %s" % CACHE)
    print("共 %d 张:信息流 %d,其他 %d" % (len(paths), pos, len(paths) - pos))
    return 0


if __name__ == "__main__":
    sys.exit(main())