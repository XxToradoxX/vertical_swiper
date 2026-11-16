# -*- coding: utf-8 -*-
"""对比"模板匹配"和"神经网络"两条识别路线(离线,用 shoot 里的真机图)。

三种测法:
1. 原图:两条路各自怎么判,和人工期望对不对得上(期望表复用 tools\\selftest.py);
2. 位置/尺度扰动(--stress):把画面整体缩放平移一点,模拟"控件没对齐" ——
   模板匹配靠像素对齐,一旦错位就掉分;神经网络见过各种偏移,应该更稳;
3. 耗时:两条路各跑一遍,看单帧多少毫秒。

用法:
    & .\\venv\\Scripts\\python.exe tools\\bench_nn.py
    & .\\venv\\Scripts\\python.exe tools\\bench_nn.py --stress 12
"""

import argparse
import os
import random
import sys
import time

import cv2
import numpy as np

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
sys.path.insert(0, os.path.join(APP, "tools"))

import vision.imgio as imgio                      # noqa: E402
from vision.canvas import Canvas                  # noqa: E402
from vision.recognizer import Recognizer          # noqa: E402
import selftest                                   # noqa: E402  (复用它的期望表)


def load_frames():
    canvas = Canvas(1264, 2780)
    out = []
    for name in sorted(os.listdir(os.path.join(APP, "shoot"))):
        if not name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        img = imgio.imread(os.path.join(APP, "shoot", name))
        if img is None:
            continue
        out.append((name, canvas.normalize(img)))
    return out


def expect_of(name):
    e = selftest.EXPECT.get(name)
    if not e:
        return None
    page, rows, bubble = e[0], e[1], e[2]
    cools = e[3] if len(e) > 3 and e[3] is not None else None
    return page, set(rows), bubble, (set(cools) if cools is not None else None)


def match_expected(view, exp):
    page, rows, bubble, cools = exp
    if view.page != page:
        return False, "页面 %s != %s" % (view.page, page)
    got = set(view.rows_visible())
    if page == "coin_page" and rows and not (rows & got):
        return False, "行 %s 里一个都没认到" % ",".join(sorted(rows))
    if bubble and view.bubble != bubble:
        return False, "气泡 %s != %s" % (view.bubble, bubble)
    if cools is not None and set(view.cooldowns) != cools:
        return False, "冷却 %s != %s" % (sorted(view.cooldowns), sorted(cools))
    return True, ""


def warp(img, scale, dx, dy):
    """整体缩放 + 平移(模拟换机/控件错位)。空出来的边用黑边填。"""
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 0.0, scale)
    m[0, 2] += dx
    m[1, 2] += dy
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stress", type=float, default=0.0,
                    help="扰动的最大缩放偏移(百分比,例如 12 = ±12%%)")
    ap.add_argument("--trials", type=int, default=6, help="每张图扰动几次")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--thr", type=float, default=0.35, help="神经网络置信度阈值")
    args = ap.parse_args()

    frames = load_frames()
    tpl = Recognizer(use_nn=False)
    nn = Recognizer(use_nn=True)
    nn.net.thr = args.thr
    if not nn.nn_available():
        raise SystemExit("找不到模型 %s —— 先跑 nn\\train.py 再跑 nn\\export.py" % nn.net.path)

    print("模板匹配 vs 神经网络 | 素材 %d 张 | NN 阈值 %.2f\n" % (len(frames), args.thr))
    print("%-42s %-10s %-10s %-10s" % ("文件", "期望", "模板", "NN"))

    ok_t = ok_n = total = 0
    t_t = t_n = 0.0
    for name, img in frames:
        exp = expect_of(name)
        t0 = time.time()
        vt = tpl.classify(img)
        t_t += time.time() - t0
        t0 = time.time()
        vn = nn.classify(img)
        t_n += time.time() - t0
        if exp is None:
            print("%-42s %-10s %-10s %-10s" % (name[:40], "-", vt.page, vn.page))
            continue
        total += 1
        good_t, why_t = match_expected(vt, exp)
        good_n, why_n = match_expected(vn, exp)
        ok_t += good_t
        ok_n += good_n
        print("%-42s %-10s %-10s %-10s %s" % (
            name[:40], exp[0],
            "OK" if good_t else "x",
            "OK" if good_n else "x",
            ("" if good_t else "模板:" + why_t) + ("" if good_n else "  NN:" + why_n)))

    print("\n原图准确率:模板 %d/%d  NN %d/%d" % (ok_t, total, ok_n, total))
    print("平均耗时:模板 %.0f ms  NN %.0f ms(cv2.dnn,CPU)"
          % (t_t / max(1, len(frames)) * 1000, t_n / max(1, len(frames)) * 1000))

    if args.stress > 0:
        pct = args.stress
        rnd = random.Random(args.seed)
        print("\n扰动测试:缩放 ±%.0f%%、平移 ±%.0f%% 画布,每张图 %d 次"
              % (pct, pct * 0.6, args.trials))
        print("%-10s %-14s %-14s" % ("扰动", "模板 命中率", "NN 命中率"))
        for step in (2.0, 4.0, 6.0, 8.0, 10.0, float(pct)):
            if step > pct:
                continue
            st = sn = 0
            for name, img in frames:
                exp = expect_of(name)
                if exp is None:
                    continue
                for _ in range(args.trials):
                    s = 1.0 + rnd.uniform(-step, step) / 100.0
                    dx = rnd.uniform(-step, step) / 100.0 * img.shape[1] * 0.6
                    dy = rnd.uniform(-step, step) / 100.0 * img.shape[0] * 0.6
                    w = warp(img, s, dx, dy)
                    st += match_expected(tpl.classify(w), exp)[0]
                    sn += match_expected(nn.classify(w), exp)[0]
            n = total * args.trials
            print("%-10s %-14s %-14s" % (
                "±%.0f%%" % step,
                "%.1f%%" % (st * 100.0 / max(1, n)),
                "%.1f%%" % (sn * 100.0 / max(1, n))))


if __name__ == "__main__":
    main()