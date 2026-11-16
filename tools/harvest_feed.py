# -*- coding: utf-8 -*-
"""从真机采一批「多多视频信息流」画面,给 `nn/feed_data.py` 当正样本。

负样本(金币页、弹窗、落地页)已经有几十张了,正样本反而只有十来张,
而且几乎全来自同一台手机的同一天 —— 网络容易学成"这台手机的信息流"。
这个脚本就是补正样本的:连上手机、停在多多视频上,它每采一张就上滑一条。

用法:
    python tools\\harvest_feed.py --count 24
    python tools\\harvest_feed.py --count 40 --gap 2.0 --out shoot\\信息流采样

采到的图会进训练集(该目录下所有图都按"信息流"标注),文件本身不进版本库。
"""

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import vision.imgio as imgio                       # noqa: E402
from vision.screen import Screen                   # noqa: E402
from vertical_swiper import load_config            # noqa: E402

FEED_DIR = os.path.join(ROOT, "shoot", "信息流采样")


def main(argv=None):
    ap = argparse.ArgumentParser(description="采信息流截图(正样本)")
    ap.add_argument("--count", type=int, default=24, help="采多少张")
    ap.add_argument("--gap", type=float, default=2.2, help="每张之间等几秒")
    ap.add_argument("--out", default=FEED_DIR)
    ap.add_argument("--serial", default=None)
    args = ap.parse_args(argv)

    cfg = load_config()
    adb = cfg.adb_path or "adb"
    screen = Screen(adb=adb, serial=args.serial or cfg.serial or None,
                    log=lambda s: print(s))
    w, h = screen.size()
    print("设备 %dx%d,输出到 %s" % (w, h, args.out))
    os.makedirs(args.out, exist_ok=True)

    saved = 0
    for i in range(args.count):
        img = screen.capture()
        if img is None:
            print("  第 %d 张截图失败,重试" % (i + 1))
            time.sleep(0.6)
            img = screen.capture()
        if img is not None:
            path = os.path.join(args.out, "feed_%02d.png" % (i + 1))
            if imgio.imwrite(path, img):
                saved += 1
                print("  [%02d/%d] %s" % (i + 1, args.count, os.path.basename(path)))
        # 上滑一条,换下一个视频
        screen.swipe(int(w * 0.5), int(h * 0.78), int(w * 0.5), int(h * 0.22), ms=320)
        time.sleep(args.gap)
    print("完成:采到 %d 张" % saved)
    return 0


if __name__ == "__main__":
    sys.exit(main())