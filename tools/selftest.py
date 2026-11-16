# -*- coding: utf-8 -*-
"""离线回归:拿 shoot 里的真机截图跑一遍识别,不进手机就能验证。

两种用法:
  python tools/selftest.py              # 只跑断言(快,FAIL 才输出细节)
  python tools/selftest.py -v           # 额外打印"模板 × 截图"的打分矩阵

打分矩阵是调阈值的唯一依据:每张模板在它自己的 ROI 里对每张截图的最佳分数
都列出来,命中的格子标 ⇒,低于阈值的标 ·。想调阈值/挪框,先看这张表。
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.canvas import Canvas                     # noqa: E402
from vision.imgio import imread, imwrite            # noqa: E402
from vision.matcher import match                    # noqa: E402
from vision.recognizer import ROI, Recognizer       # noqa: E402

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHOOT_DIR = os.path.join(APP_DIR, "shoot")

# 文件名 -> (期望页面, 必须看到的模块行, 期望气泡状态)
EXPECT = {
    "紫色部分圈出来的是需要识别的部分.png": ("feed", set(), None),
    "实时推荐页.png": ("feed", set(), None),
    "点击金币按钮进入的二级界面 红色部分可以滑动.png": ("coin_page", set(), None),
    "二级界面滑动一下.jpg": ("coin_page", {"shop_video", "water", "live"}, None),
    "二级界面滑动两下.jpg": ("coin_page", set(), None),
    "看商品视频得现金_.png": ("task", set(), "watch"),
    "喝水打卡.png": ("task", set(), "watch"),
    "周末喝水翻倍赚钱.jpg": ("water_popup", set(), None),
    "二级金币_当前2.png": ("coin_page", {"water"}, None, {"water"}),
    "喝水倒计时冷却.png": ("coin_page", {"water"}, None),
    "倒计时任务完成.png": ("task", set(), "done"),
    "看直播赚现金.jpg": ("task", set(), "watch"),
    "二级金币_冷却.png": ("coin_page", {"shop_video", "water"}, None),
    # 只有「再逛逛」浅粉按钮、没有红字的冷却态(用户 2026-11-14 提供的真机帧)
    "二级金币_粉色冷却.png": ("coin_page", {"shop_video"}, None, {"shop_video"}),
    # 「看直播赚现金」改了文案,变成「去直播间下单」(同一模块的第二个变体)
    "二级金币_去直播间下单.png": ("coin_page", {"live"}, None, set()),
    # 直播间右上角「任务完成」= 这个板块今天已经领过(冷却),不是"还要再等"
    "直播_任务完成冷却.png": ("task", set(), "done"),
    # 同一页里 water 冷却(浅粉)、live 可用(红)—— 两种色调必须在同一帧里分开
    "喝水倒计时冷却.png": ("coin_page", {"water", "live"}, None, {"water"}),
    "直播_后领冷却.png": ("task", set(), "cooldown"),
    "商品视频_领取成功.png": ("task", set(), "cooldown"),
}

# 每张模板该在哪个 ROI 里找(打分矩阵用)
TPL_ROI = {
    "page_coin": "coin_header",
    "row_shop_video": "content",
    "row_water": "content",
    "row_live": "content",
    "row_live2": "content",
    "bubble_watch": "bubble",
    "bubble_done": "bubble",
    "coin_entry": "coin_entry",
    "water_popup_title": "water_title",
    "water_checkin_btn": "water_btn",
    "bubble_cooldown": "bubble",
    "bubble_cooldown2": "bubble",
}


def canvas_roi(roi_name, w, h):
    x0, y0, x1, y1 = ROI[roi_name]
    return (int(x0 * w), int(y0 * h),
            max(2, int((x1 - x0) * w)), max(2, int((y1 - y0) * h)))


def score_matrix(rec, frames):
    """模板 × 截图 的最佳分数矩阵。"""
    names = [n for n in TPL_ROI if rec.templates.has(n)]
    rows = []
    for name in names:
        row = []
        for fname, img in frames:
            h, w = img.shape[:2]
            hit = match(img, rec.templates, name,
                        roi=canvas_roi(TPL_ROI[name], w, h))
            row.append((name, fname, hit.score if hit else 0.0,
                        rec.templates.threshold(name)))
        rows.append(row)
    header = ["模板".ljust(20)] + [os.path.splitext(f)[0][:12].ljust(13)
                                  for f, _ in frames]
    print("")
    print("=" * 100)
    print("  模板 × 截图 打分矩阵(⇒ 过阈值 / · 未过阈值)")
    print("=" * 100)
    print("".join(header))
    for row in rows:
        line = [row[0][0].ljust(20)]
        for name, fname, score, thr in row:
            mark = "⇒" if score >= thr else "·"
            line.append(("%s%.2f" % (mark, score)).ljust(13))
        print("".join(line))
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="离线回归测试")
    parser.add_argument("-v", "--verbose", action="store_true", help="打印打分矩阵")
    parser.add_argument("--save", action="store_true", help="保存每张图的识别调试图")
    parser.add_argument("--out", default=os.path.join(APP_DIR, "debug"))
    args = parser.parse_args()

    files = [f for f in EXPECT if os.path.isfile(os.path.join(SHOOT_DIR, f))]
    missing = [f for f in EXPECT if f not in files]
    if missing:
        print("缺少截图: %s" % ", ".join(missing))

    frames = []
    for fname in files:
        raw = imread(os.path.join(SHOOT_DIR, fname))
        if raw is None:
            print("读不出来: %s" % fname)
            continue
        canvas = Canvas(raw.shape[1], raw.shape[0])
        frames.append((fname, canvas.normalize(raw)))

    rec = Recognizer()
    print("模板: %d 项(%s)" % (len(rec.templates.names()), rec.templates.dir))

    failed = []
    for fname, img in frames:
        view = rec.classify(img)
        page, want_rows, want_bubble = EXPECT[fname][:3]
        want_cool = EXPECT[fname][3] if len(EXPECT[fname]) > 3 else None
        errs = []
        if view.page != page:
            errs.append("页面 %s != %s" % (view.page, page))
        got_rows = set(view.rows_visible())
        if want_rows and not want_rows <= got_rows:
            errs.append("缺行 %s(只看到 %s)" % (sorted(want_rows - got_rows), sorted(got_rows)))
        if want_bubble and view.bubble != want_bubble:
            errs.append("气泡 %s != %s" % (view.bubble, want_bubble))
        if want_cool is not None and set(want_cool) != set(view.cooldowns):
            errs.append("冷却 %s != %s" % (sorted(view.cooldowns), sorted(want_cool)))
        mark = "OK  " if not errs else "FAIL"
        print("[%s] %-38s -> %s%s" % (mark, fname[:38], view,
                                      ("   << " + "; ".join(errs)) if errs else ""))
        if errs:
            failed.append((fname, errs))
        if args.save:
            os.makedirs(args.out, exist_ok=True)
            rec.draw(img, view, os.path.join(args.out, "selftest_" + os.path.splitext(fname)[0][:12] + ".png"))

    if args.verbose:
        score_matrix(rec, frames)

    print("")
    print("%d/%d 通过" % (len(frames) - len(failed), len(frames)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())