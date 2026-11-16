#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从真机截图裁出模板图(标定工具)。

用法:
    python tools/make_templates.py                # 按 template_spec.json 重新生成全部模板
    python tools/make_templates.py --montage      # 顺便拼一张审阅图 debug/templates_review.png
    python tools/make_templates.py --only row_water

所有 box 都是 1080x1920 标准画布坐标。截图会先经过 Canvas.normalize(),
所以裁出来的模板天然就是"16:9 训练"的产物,换手机只要重跑本脚本。
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP_DIR)

from vision.canvas import Canvas            # noqa: E402
from vision.imgio import imread, imwrite    # noqa: E402

SHOOT_DIR = os.path.join(APP_DIR, "shoot")
SPEC_PATH = os.path.join(APP_DIR, "tools", "template_spec.json")
TPL_DIR = os.path.join(APP_DIR, "vision", "templates")
REVIEW_PATH = os.path.join(APP_DIR, "debug", "templates_review.png")


def load_spec():
    with open(SPEC_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def crop_one(spec, name, item, out_dir=TPL_DIR):
    src = os.path.join(SHOOT_DIR, item["source"])
    frame = imread(src)
    if frame is None:
        print("  [跳过] %-20s 找不到截图: %s" % (name, item["source"]))
        return None
    h, w = frame.shape[:2]
    canvas = Canvas(w, h)
    img = canvas.normalize(frame)
    x0, y0, x1, y1 = [int(v) for v in item["box"]]
    x0 = max(0, min(x0, canvas.width - 2))
    y0 = max(0, min(y0, canvas.height - 2))
    x1 = max(x0 + 2, min(x1, canvas.width))
    y1 = max(y0 + 2, min(y1, canvas.height))
    crop = img[y0:y1, x0:x1].copy()
    out = os.path.join(out_dir, name + ".png")
    if not imwrite(out, crop):
        print("  [失败] %-20s 写不了 %s" % (name, out))
        return None
    print("  [完成] %-20s %3dx%-3d  <- %s" % (name, crop.shape[1], crop.shape[0], item["source"]))
    return crop


def build_montage(crops, review_path=REVIEW_PATH):
    """把模板拼成一张带标签的审阅图,方便一眼看出裁歪了没有。"""
    if not crops:
        return
    pad, label_h, cols = 12, 22, 4
    items = list(crops.items())
    cell_w = max(c.shape[1] for _, c in items) + pad
    cell_h = max(c.shape[0] for _, c in items) + pad + label_h
    rows = (len(items) + cols - 1) // cols
    sheet = np.full((rows * cell_h + pad, cols * cell_w + pad, 3), 40, np.uint8)
    for idx, (name, crop) in enumerate(items):
        r, c = divmod(idx, cols)
        ox = pad + c * cell_w
        oy = pad + r * cell_h
        sheet[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
        cv2.putText(sheet, name, (ox, oy + crop.shape[0] + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 128), 1, cv2.LINE_AA)
    imwrite(review_path, sheet)
    print("审阅图: %s" % review_path)


def build_context_sheet(spec, out_path):
    """把每个模板放回它周围的画面里看:外框=周围区域,绿框=实际裁剪范围。

    比"只看裁片"更容易判断裁歪了没有。
    """
    cells = []
    for name, item in spec["templates"].items():
        src_path = os.path.join(SHOOT_DIR, item["source"])
        frame = imread(src_path)
        if frame is None:
            continue
        h, w = frame.shape[:2]
        canvas = Canvas(w, h)
        img = canvas.normalize(frame)
        x0, y0, x1, y1 = [int(v) for v in item["box"]]
        bw, bh = x1 - x0, y1 - y0
        px, py = int(bw * 0.28) + 6, int(bh * 0.55) + 6
        cx0 = max(0, x0 - px)
        cy0 = max(0, y0 - py)
        cx1 = min(canvas.width, x1 + px)
        cy1 = min(canvas.height, y1 + py)
        cell = img[cy0:cy1, cx0:cx1].copy()
        cv2.rectangle(cell, (x0 - cx0, y0 - cy0), (x1 - cx0 - 1, y1 - cy0 - 1),
                      (0, 255, 0), 1)
        cells.append((name, cell))

    if not cells:
        return
    cols = 3
    pad, label_h = 10, 20
    cell_w = max(c.shape[1] for _, c in cells) + pad
    cell_h = max(c.shape[0] for _, c in cells) + pad + label_h
    rows = (len(cells) + cols - 1) // cols
    sheet = np.full((rows * cell_h + pad, cols * cell_w + pad, 3), 30, np.uint8)
    for idx, (name, cell) in enumerate(cells):
        r, c = divmod(idx, cols)
        ox = pad + c * cell_w
        oy = pad + r * cell_h
        sheet[oy:oy + cell.shape[0], ox:ox + cell.shape[1]] = cell
        cv2.putText(sheet, name, (ox, oy + cell.shape[0] + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 128), 1, cv2.LINE_AA)
    imwrite(out_path, sheet)
    print("上下文审阅图: %s" % out_path)


def write_manifest(spec, out_dir):
    """把阈值/缩放档连同模板一起落成 manifest.json,运行时只读它。

    阈值单独放在这里而不是写死在代码里,是学 MAA 的做法:调参数不用改代码。
    """
    manifest = {"canvas": spec.get("canvas", [1080, 1920]), "templates": {}}
    for name, item in spec["templates"].items():
        png = os.path.join(out_dir, name + ".png")
        if not os.path.isfile(png):
            continue
        manifest["templates"][name] = {
            "file": name + ".png",
            "desc": item.get("desc", ""),
            "threshold": float(item.get("threshold", 0.72)),
            "scales": [float(s) for s in item.get("scales", [1.0])],
            "mode": item.get("mode", "color"),
        }
    path = os.path.join(out_dir, "manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print("清单: %s (共 %d 项)" % (path, len(manifest["templates"])))


def main():
    parser = argparse.ArgumentParser(description="从截图裁模板图")
    parser.add_argument("--only", help="只生成指定模板")
    parser.add_argument("--montage", action="store_true", help="额外输出审阅拼图")
    parser.add_argument("--out", default=TPL_DIR, help="模板输出目录")
    parser.add_argument("--review", default=REVIEW_PATH, help="审阅拼图输出路径")
    parser.add_argument("--context", action="store_true",
                        help="额外输出「模板放回上下文」的审阅图")
    args = parser.parse_args()

    spec = load_spec()
    os.makedirs(args.out, exist_ok=True)
    print("模板目录: %s" % args.out)
    crops = {}
    for name, item in spec["templates"].items():
        if args.only and args.only != name:
            continue
        crop = crop_one(spec, name, item, args.out)
        if crop is not None:
            crops[name] = crop
    print("共 %d 个模板" % len(crops))
    if crops and not args.only:
        write_manifest(spec, args.out)
    if args.montage:
        build_montage(crops, args.review)
    if args.context:
        build_context_sheet(spec, os.path.join(os.path.dirname(args.review), "templates_context.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())