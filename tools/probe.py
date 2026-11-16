# -*- coding: utf-8 -*-
r"""连机后第一件事:截一张当前屏幕,看看识别成什么。

    python tools\probe.py                 # 截一帧,识别,存调试图
    python tools\probe.py --wait 3        # 等 3 秒再截(方便切页面)
    python tools\probe.py --repeat 10     # 每 2 秒截一次,连着看 10 次

输出:
    - 控制台:识别结论(页面 / 看到的模块行 / 气泡状态)
    - debug\probe_<时间>.png:同一帧上画好的命中框和分数

调参就看这张图:框画歪了就是 template_spec.json 里的 box 要挪,
框对但没命中就是 threshold 要降。
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.canvas import Canvas          # noqa: E402
from vision.imgio import imwrite          # noqa: E402
from vision.recognizer import Recognizer  # noqa: E402
from vision.screen import Screen          # noqa: E402

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from vertical_swiper import load_config   # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="识别一张实时截图")
    parser.add_argument("--wait", type=float, default=0.0, help="截图前等几秒")
    parser.add_argument("--repeat", type=int, default=1, help="连续截几次")
    parser.add_argument("--gap", type=float, default=2.0, help="连截的间隔秒数")
    parser.add_argument("--out", default=os.path.join(APP_DIR, "debug"))
    args = parser.parse_args()

    cfg = load_config()
    screen = Screen(adb=cfg.adb_path or "adb", serial=cfg.serial or None, log=print)
    print("adb: %s" % screen.adb)
    w, h = screen.size()
    print("设备: %s  %dx%d" % (screen.serial or "自动", w, h))
    canvas = Canvas(w, h)
    print(canvas.describe())

    rec = Recognizer()
    print("模板: %d 个" % len(rec.templates.names()))
    os.makedirs(args.out, exist_ok=True)

    for i in range(max(1, args.repeat)):
        if args.wait:
            time.sleep(args.wait)
        t0 = time.time()
        raw = screen.capture()
        if raw is None:
            print("截不到图(检查 adb devices / 是否授权)")
            return 2
        cost = (time.time() - t0) * 1000
        view = rec.classify(canvas.normalize(raw))
        path = os.path.join(args.out, "probe_%s.png" % time.strftime("%H%M%S"))
        rec.draw(canvas.normalize(raw), view, path)
        print("[%d] %s   (截图+识别 %.0f ms)  →  %s"
              % (i + 1, time.strftime("%H:%M:%S"), cost, view))
        for name, hit in view.hits.items():
            print("      %-20s score=%.2f box=(%d,%d %dx%d)"
                  % (name, hit.score, hit.x, hit.y, hit.w, hit.h))
        print("      调试图: %s" % path)
        if i + 1 < args.repeat:
            time.sleep(args.gap)
    return 0


if __name__ == "__main__":
    sys.exit(main())