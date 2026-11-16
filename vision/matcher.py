# -*- coding: utf-8 -*-
"""多尺度模板匹配。

为什么是多尺度
--------------
参考 MAA 的思路:匹配前把画面归一化到固定画布,模板也按画布裁。
但只要换一台手机,同一个控件在画布里的像素尺寸就会变(1080 宽的手机
和 1264 宽的手机,缩放系数不同),所以还得在 ±15% 范围里各试一档缩放,
取分最高的那档。这一步是"换机不用重新标定"的关键。

相似度用 TM_CCOEFF_NORMED:它对整体亮度/对比度变化不敏感,视频背景
一直在变也能稳定命中文字/按钮。
"""

import json
import os

import cv2
import numpy as np

from .imgio import imread

DEFAULT_THRESHOLD = 0.72
DEFAULT_SCALES = (1.0,)
DEFAULT_MODE = "color"


def _prep(img, mode):
    """把图变成待匹配的形式。

    mode="edge" 是给"白字压在任何背景上"的气泡用的:商品页气泡是粉底白字,
    直播间气泡是深底白字,颜色完全不同但字形边缘一样 —— 在 Canny 边缘图上
    匹配就都能命中。这是像素级识别里很实用的一招。

    mode="white" 更极端一点:只保留"亮且不彩"的像素,也就是白字本身,
    背景无论粉的、黑的、还是视频画面都被抹成 0。气泡里的倒计时数字会变
    (00:55 / 00:54 / 00:57),所以模板只裁「再看」两个字,不裁数字。
    """
    if mode == "edge":
        return cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 50, 150)
    if mode == "gray":
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if mode == "white":
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = ((hsv[:, :, 2] >= 190) & (hsv[:, :, 1] <= 95)).astype("uint8") * 255
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    if mode == "text":
        # 局部对比度文本掩码:比周围亮 45 以上的像素留下。
        # 气泡里的字永远是"比底色亮"的,粉底粉字、深底白字都能抠出来,
        # 抠完只剩字形,背景颜色就彻底不影响了。
        v = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2]
        bg = cv2.blur(v, (25, 25))
        mask = (v.astype("int16") - bg.astype("int16") > 45).astype("uint8") * 255
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    if mode == "lc":
        # 局部对比度归一化:只保留"比周围亮/暗多少",抹掉绝对亮度。
        # 这是气泡识别的正解:商品页是粉底粉字,直播间是深底白字,
        # 绝对值天差地别,但"字比底亮"这件事不变。归一化后
        # TM_CCOEFF_NORMED 能把两种情况都打到 0.84~0.98,
        # 而错误的那个状态只有 0.35~0.43,分得很开。
        v = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2].astype("float32")
        bg = cv2.blur(v, (25, 25))
        return np.clip(128.0 + (v - bg), 0, 255).astype("uint8")
    return img


def _method(mode):
    """二值掩码用余弦相似度,彩色/灰度用归一化相关系数。"""
    return cv2.TM_CCORR_NORMED if mode in ("white", "text") else cv2.TM_CCOEFF_NORMED


class Hit:
    """一次命中的结果,坐标都是标准画布像素。"""

    __slots__ = ("name", "score", "x", "y", "w", "h", "scale")

    def __init__(self, name, score, x, y, w, h, scale=1.0):
        self.name = name
        self.score = float(score)
        self.x, self.y = int(x), int(y)
        self.w, self.h = int(w), int(h)
        self.scale = float(scale)

    @property
    def cx(self):
        return self.x + self.w // 2

    @property
    def cy(self):
        return self.y + self.h // 2

    @property
    def right(self):
        return self.x + self.w

    @property
    def bottom(self):
        return self.y + self.h

    @property
    def rect(self):
        return self.x, self.y, self.w, self.h

    def __repr__(self):
        return "<Hit %s %.3f @(%d,%d %dx%d) x%.2f>" % (
            self.name, self.score, self.x, self.y, self.w, self.h, self.scale)


class TemplateSet:
    """一套模板 + 每张模板自己的阈值和缩放档。"""

    def __init__(self, tpl_dir):
        self.dir = tpl_dir
        self.items = {}      # name -> {file, threshold, scales, desc}
        self._cache = {}     # (name, scale) -> ndarray
        self.load_manifest()

    def load_manifest(self):
        path = os.path.join(self.dir, "manifest.json")
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.items = dict(data.get("templates") or {})
                return
            except (OSError, ValueError):
                self.items = {}
        # 没有清单就退化成"目录里有什么就用什么"
        if os.path.isdir(self.dir):
            for fn in os.listdir(self.dir):
                if fn.lower().endswith(".png"):
                    name = os.path.splitext(fn)[0]
                    self.items[name] = {"file": fn, "threshold": DEFAULT_THRESHOLD,
                                        "scales": list(DEFAULT_SCALES), "desc": ""}

    def names(self):
        return sorted(self.items)

    def has(self, name):
        return name in self.items

    def threshold(self, name):
        return float(self.items.get(name, {}).get("threshold", DEFAULT_THRESHOLD))

    def scales(self, name):
        sc = self.items.get(name, {}).get("scales") or DEFAULT_SCALES
        return [float(s) for s in sc]

    def mode(self, name):
        return self.items.get(name, {}).get("mode") or DEFAULT_MODE

    def image(self, name, scale=1.0, mode=None):
        """按缩放档取模板图,结果缓存(模板是常量,不需要每帧重算)。"""
        mode = mode or self.mode(name)
        key = (name, round(scale, 3), mode)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        item = self.items.get(name)
        if not item:
            return None
        path = os.path.join(self.dir, item.get("file") or (name + ".png"))
        img = imread(path)
        if img is None:
            return None
        if abs(scale - 1.0) > 1e-3:
            nw = max(4, int(round(img.shape[1] * scale)))
            nh = max(4, int(round(img.shape[0] * scale)))
            img = cv2.resize(img, (nw, nh),
                             interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
        img = _prep(img, mode)
        self._cache[key] = img
        return img


def match(img, tpl_set, name, roi=None, scales=None, threshold=None, mode=None,
          prepped=False):
    """在 img(标准画布 BGR)的 roi 里找模板 name,返回最佳 Hit 或 None。

    roi 是画布像素 (x, y, w, h);None 表示全画面。
    prepped=True 表示 img 已经过 _prep 处理过(同一帧要匹配多个模板时,
    预处理一次就够,别每个模板都重算一遍)。
    """
    if img is None or name not in tpl_set.items:
        return None
    mode = mode or tpl_set.mode(name)
    if not prepped:
        img = _prep(img, mode)
    H, W = img.shape[:2]
    if roi is None:
        rx, ry, rw, rh = 0, 0, W, H
    else:
        rx, ry, rw, rh = [int(v) for v in roi]
        rx = max(0, min(rx, W - 2))
        ry = max(0, min(ry, H - 2))
        rw = max(2, min(rw, W - rx))
        rh = max(2, min(rh, H - ry))
    region = img[ry:ry + rh, rx:rx + rw]
    if region.size == 0:
        return None

    thr = tpl_set.threshold(name) if threshold is None else float(threshold)
    best = None
    for scale in (scales or tpl_set.scales(name)):
        tpl = tpl_set.image(name, scale, mode)
        if tpl is None:
            continue
        th, tw = tpl.shape[:2]
        if tw > rw or th > rh:
            continue
        res = cv2.matchTemplate(region, tpl, _method(mode))
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val != max_val:      # NaN:模板是纯色块,相关系数无定义
            continue
        if best is None or max_val > best.score:
            best = Hit(name, max_val, rx + max_loc[0], ry + max_loc[1], tw, th, scale)
    if best is not None and best.score >= thr:
        return best
    return None


def best_of(img, tpl_set, names, roi=None, **kw):
    """在多个候选模板里取分最高的那个(用于"到底是哪种状态"的判定)。"""
    winner = None
    for name in names:
        if name not in tpl_set.items:
            continue
        hit = match(img, tpl_set, name, roi=roi, **kw)
        if hit is not None and (winner is None or hit.score > winner.score):
            winner = hit
    return winner


def roi_around(hit, width, height, pad_x=0.0, pad_y=0.0):
    """以某个命中框为中心,扩出一个画布像素 ROI(用于"行内再看按钮"这种相对搜索)。"""
    hw = int(hit.w * (1 + pad_x))
    hh = int(hit.h * (1 + pad_y))
    x = max(0, hit.cx - hw // 2)
    y = max(0, hit.cy - hh // 2)
    return x, y, min(hw, width - x), min(hh, height - y)