"""倒计时数字读取(不用深度学习,纯几何 + 5/7 两个模板)。

设计动机:日志里要能写「再看 00:55(还剩 55 秒)」,这需要把气泡上的数字读出来。
数字是亮笔画、字体固定、尺寸固定(画布上约 17×24),所以不需要 OCR 引擎:

    0 / 4 / 6 / 8 / 9  →  看「孔洞」的个数和位置(8 有两个洞,6 的洞偏下,9 偏上,0 的洞很长,4 的洞很小)
    1              →  看「宽度只有一个笔画」
    5 / 7          →  用真机截图里抠出来的两张模板做形状匹配
    2 / 3          →  看左下角有没有笔画(2 的斜笔收在左下,3 没有)

任何一步拿不准就返回 "?",绝不猜 —— 日志里写 "00:5?" 比写错一个数字好。
"""
from __future__ import annotations

import os

import cv2
import numpy as np

GLYPH_W = 20
GLYPH_H = 26

# 5 和 7 的形状靠模板比对(其它数字都有更强的结构性特征,不需要模板)
TEMPLATE_DIGITS = ("5", "7")


def lc_plane(img: np.ndarray) -> np.ndarray:
    """局部对比度归一化:把亮字从任意底色里剥离出来(粉胶囊/深色胶囊通用)。"""
    v = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2].astype("float32")
    return np.clip(128.0 + (v - cv2.blur(v, (25, 25))), 0, 255).astype("uint8")


def _binarize(band: np.ndarray) -> np.ndarray:
    thr, out = cv2.threshold(band, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return out


def segment(band: np.ndarray):
    """把一条数字带切成单个字形,按 x 排序,丢掉胶囊边框之类的杂块。"""
    binimg = _binarize(band)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binimg, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 30 or not (14 <= h <= 30) or not (5 <= w <= 24):
            continue
        out.append(((int(x), int(y), int(w), int(h)), binimg[y:y + h, x:x + w]))
    out.sort(key=lambda t: t[0][0])
    return out


def _holes(mask: np.ndarray):
    """字形内部的孔洞:[(cx, cy, area)] ,坐标相对字形左上角。"""
    m = (mask > 0).astype(np.uint8)
    m = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    n, _, stats, _ = cv2.connectedComponentsWithStats(1 - m, 8)
    hgt, wid = m.shape
    holes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if x <= 0 or y <= 0 or x + w >= wid or y + h >= hgt:   # 碰到外框 = 外部背景
            continue
        if area < 5:
            continue
        holes.append((x + w / 2.0, y + h / 2.0, int(area), int(h)))
    return holes


def _norm(mask: np.ndarray) -> np.ndarray:
    return cv2.resize(mask, (GLYPH_W, GLYPH_H), interpolation=cv2.INTER_AREA)


def _dens(mask: np.ndarray, y0, y1, x0, x1) -> float:
    part = mask[y0:y1, x0:x1]
    return float(part.mean()) / 255.0 if part.size else 0.0


class DigitReader:
    """把倒计时气泡上的「00:55」读成整数秒。模板从 vision/templates/digits 加载。"""

    def __init__(self, template_dir: str):
        self.dir = template_dir
        self.templates = {}
        for d in TEMPLATE_DIGITS:
            path = os.path.join(template_dir, "d%s.png" % d)
            img = None
            if os.path.isfile(path):
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            self.templates[d] = _norm((img > 127).astype(np.uint8) * 255)

    # ---------- 单个字形 ----------
    def read_glyph(self, mask: np.ndarray) -> str:
        h, w = mask.shape[:2]
        holes = _holes(mask)
        g = _norm(mask)

        if len(holes) >= 2:
            return "8"
        if len(holes) == 1:
            cx, cy, area, hh = holes[0]
            if area / float(w * h) < 0.11:
                return "4"                       # 4 的那个小三角
            if hh / float(h) > 0.55:
                return "0"                       # 0 的洞几乎占满整个高度
            if cy / float(h) < 0.50:
                return "9"
            return "6"

        # 没有孔洞:1 最窄
        if w / float(h) < 0.52:
            return "1"

        best, best_score = "?", 0.0
        for d, tpl in self.templates.items():
            res = cv2.matchTemplate(g, tpl, cv2.TM_CCOEFF_NORMED)
            score = float(res.max())
            if score > best_score:
                best, best_score = d, score
        if best_score >= 0.72:
            return best

        # 2 的斜笔收在左下角,3 的左下是空的
        left_low = _dens(g, int(GLYPH_H * 0.62), GLYPH_H, 0, int(GLYPH_W * 0.45))
        bottom = _dens(g, int(GLYPH_H * 0.82), GLYPH_H, 0, GLYPH_W)
        if left_low >= 0.30 and bottom >= 0.25:
            return "2"
        if bottom >= 0.35 or left_low < 0.30:
            return "3"
        return "?"

    # ---------- 整条倒计时 ----------
    def read_band(self, band: np.ndarray):
        """返回 (文本, 秒数)。文本形如 "00:55";读不出来就是 "?"。"""
        items = segment(band)
        if not items:
            return "", None
        # 只保留高度接近最大值的字形(排除下面的「得2500」残片和边框)
        hmax = max(b[3] for b, _ in items)
        items = [it for it in items if it[0][3] >= hmax * 0.8]
        if not items:
            return "", None

        digits = []
        for box, mask in items:
            ch = self.read_glyph(mask)
            digits.append(ch)
            if len(digits) >= 5:
                break
        if len(digits) < 4:
            return "".join(digits), None
        text = "".join(digits[:4])
        if "?" in text:
            return text, None
        secs = int(text[:2]) * 60 + int(text[2:])
        return text, secs

    def read_bubble(self, canvas_img: np.ndarray, hit, band_h: int = 34):
        """给定「再看」的命中框,读它下面那行数字。返回 (文本, 秒数)。"""
        h, w = canvas_img.shape[:2]
        x0 = max(0, int(hit.x) - 12)
        y0 = max(0, int(hit.bottom) - 4)
        x1 = min(w, x0 + 130)
        y1 = min(h, y0 + band_h)
        if x1 - x0 < 20 or y1 - y0 < 12:
            return "", None
        band = lc_plane(canvas_img)[y0:y1, x0:x1]
        return self.read_band(band)