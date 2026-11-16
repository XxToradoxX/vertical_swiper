"""从真机截图里把倒计时数字抠出来,生成数字模板(一次性标定工具)。

模板只做 5 和 7(其它数字用几何/孔洞规则判断,见 vision/digits.py)。
产物落 venv/_stage/digits 之后再用 pwsh 搬进 vision/templates/digits。

    python tools\\harvest_digits.py --out venv\\_stage\\digits --montage venv\\_stage\\digits.png
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from vision.canvas import Canvas
from vision.digits import lc_plane, segment
from vision.imgio import imread, imwrite
from vision.matcher import TemplateSet, match
from vision.recognizer import ROI

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHOOT = os.path.join(APP_DIR, "shoot")
TPL = os.path.join(APP_DIR, "vision", "templates")

# (文件名, 气泡上显示的倒计时文本)
SOURCES = [
    ("看商品视频得现金_.png", "00:55"),
    ("看直播赚现金.jpg", "00:57"),
    ("喝水打卡.png", "00:54"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--montage", default="")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    tset = TemplateSet(TPL)
    canvas = Canvas(1264, 2780)
    tiles = []
    saved = {}
    for fname, expect in SOURCES:
        path = os.path.join(SHOOT, fname)
        if not os.path.exists(path):
            print("缺文件:", fname)
            continue
        img = canvas.normalize(imread(path))
        hit = match(img, tset, "bubble_watch", roi=canvas.roi_px(ROI["bubble"]),
                    threshold=0.4)
        if hit is None:
            print(fname, "没找到「再看」")
            continue
        band = lc_plane(img)[hit.bottom - 4:hit.bottom + 30, max(0, hit.x - 12):hit.x + 118]
        items = segment(band)
        got = "".join(str(i) for i, _ in enumerate(items))
        print(f"\n=== {fname} 期望 {expect},切出 {len(items)} 个字形")
        want = [c for c in expect if c != ":"]
        if len(items) != len(want):
            print("   数量对不上,跳过这个源图")
            continue
        for (box, mask), ch in zip(items, want):
            tiles.append((ch, box, mask))
            if ch in ("5", "7") and ch not in saved:
                dst = os.path.join(args.out, "d%s.png" % ch)
                imwrite(dst, np.where(mask > 0, 255, 0).astype(np.uint8))
                saved[ch] = dst
                print("   保存模板 d%s -> %s  (%dx%d)" % (ch, dst, box[2], box[3]))

    print("\n收集到的数字模板:", sorted(saved) or "(无)")
    if args.montage and tiles:
        wmax = max(m.shape[1] for _, _, m in tiles) * 4
        hmax = max(m.shape[0] for _, _, m in tiles) * 4
        cols = 8
        rows = (len(tiles) + cols - 1) // cols
        sheet = np.full((rows * (hmax + 26), cols * (wmax + 10), 3), 255, np.uint8)
        for i, (ch, box, m) in enumerate(tiles):
            r, c = divmod(i, cols)
            t = cv2.resize(m, (box[2] * 4, box[3] * 4), interpolation=cv2.INTER_NEAREST)
            t = cv2.copyMakeBorder(t, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=40)
            yy, xx = r * (hmax + 26), c * (wmax + 10)
            sheet[yy:yy + t.shape[0], xx:xx + t.shape[1]] = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
            cv2.putText(sheet, ch, (xx + 2, yy + t.shape[0] + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 1, cv2.LINE_AA)
        imwrite(args.montage, sheet)
        print("montage:", args.montage, sheet.shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())