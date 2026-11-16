# -*- coding: utf-8 -*-
"""16:9 标准画布。

为什么要它
----------
模板匹配对**尺度**敏感:同一张模板图,在 1080p 和 1264 宽的手机上
像素尺寸不一样,不归一化就一定对不上。MAA 的做法是强制设备分辨率
(虚拟屏锁 16:9),我们做不到改真机分辨率,只能在两边各做一次几何变换:

    设备截图 --normalize--> 1080x1920 标准画布 --match--> 命中框(画布坐标)
                                                        |
    设备坐标 <--to_device---------------------------------+

模板全部在标准画布上裁剪、模板框和点击点全部用画布坐标表达,于是
"手机分辨率"这件事只出现在这一个文件里。

policy
------
stretch(默认)  直接把画面缩放到 1080x1920。长宽比不是 16:9 的手机会有
               轻微的纵向压缩(1264x2780 约 19%)。好处是画布恒定、
               两边变换严格互逆,点击位置精确。
fit            等比缩放后居中黑边。图像不变形,但非 16:9 的机器会留下
               黑边,画布内容区变小。适合和别的 16:9 设备混用模板时。
"""

import cv2
import numpy as np

CANON_W = 1080
CANON_H = 1920


class Canvas:
    """设备像素 <-> 1080x1920 标准画布 的双向映射。"""

    def __init__(self, dev_w, dev_h, policy="stretch", size=(CANON_W, CANON_H)):
        if dev_w <= 0 or dev_h <= 0:
            raise ValueError("设备分辨率无效: %sx%s" % (dev_w, dev_h))
        self.dev_w = int(dev_w)
        self.dev_h = int(dev_h)
        self.width, self.height = int(size[0]), int(size[1])
        self.policy = policy if policy in ("stretch", "fit") else "stretch"

        if self.policy == "stretch":
            # 逐轴独立缩放:画布永远铺满,内容可能轻微变形
            self._sx = self.width / float(self.dev_w)
            self._sy = self.height / float(self.dev_h)
            self._ox = self._oy = 0
            self.cw, self.ch = self.width, self.height
        else:
            # 等比缩放 + 居中留黑边:内容不变形,但四周可能有 padding
            s = min(self.width / float(self.dev_w), self.height / float(self.dev_h))
            self._sx = self._sy = s
            self.cw = max(1, int(round(self.dev_w * s)))
            self.ch = max(1, int(round(self.dev_h * s)))
            self._ox = (self.width - self.cw) // 2
            self._oy = (self.height - self.ch) // 2

    # -- 归一化 --------------------------------------------------------
    def normalize(self, frame):
        """设备原始帧(BGR) -> 标准画布(BGR)。"""
        if frame is None:
            return None
        h, w = frame.shape[:2]
        if (w, h) == (self.width, self.height):
            return frame
        if self.policy == "stretch":
            interp = cv2.INTER_AREA if (w > self.width) else cv2.INTER_LINEAR
            return cv2.resize(frame, (self.width, self.height), interpolation=interp)
        resized = cv2.resize(frame, (self.cw, self.ch), interpolation=cv2.INTER_AREA)
        if (self.cw, self.ch) == (self.width, self.height):
            return resized
        out = np.zeros((self.height, self.width, 3), np.uint8)
        out[self._oy:self._oy + self.ch, self._ox:self._ox + self.cw] = resized
        return out

    # -- 坐标换算 ------------------------------------------------------
    def to_device(self, cx, cy):
        """画布坐标 -> 设备坐标(点按/滑动前必须过这一步)。"""
        dx = self._ox_plus(cx) / self._sx
        dy = self._oy_plus(cy) / self._sy
        dx = min(max(int(round(dx)), 0), self.dev_w - 1)
        dy = min(max(int(round(dy)), 0), self.dev_h - 1)
        return dx, dy

    def _ox_plus(self, x):
        return (x - self._ox) if self.policy == "fit" else x

    def _oy_plus(self, y):
        return (y - self._oy) if self.policy == "fit" else y

    def to_canvas(self, dx, dy):
        """设备坐标 -> 画布坐标(调试用)。"""
        cx = dx * self._sx + self._ox
        cy = dy * self._sy + self._oy
        return int(round(cx)), int(round(cy))

    def roi_px(self, spec):
        """归一化 ROI [x0,y0,x1,y1](0~1)-> 画布像素 (x, y, w, h)。

        越界自动裁剪;退化成一个点以下时返回 None。
        """
        if not spec:
            return (0, 0, self.width, self.height)
        x0, y0, x1, y1 = [float(v) for v in spec]
        x0 = min(max(x0, 0.0), 1.0)
        x1 = min(max(x1, 0.0), 1.0)
        y0 = min(max(y0, 0.0), 1.0)
        y1 = min(max(y1, 0.0), 1.0)
        if x1 <= x0 or y1 <= y0:
            return None
        px = int(round(x0 * self.width))
        py = int(round(y0 * self.height))
        pw = max(1, int(round((x1 - x0) * self.width)))
        ph = max(1, int(round((y1 - y0) * self.height)))
        pw = min(pw, self.width - px)
        ph = min(ph, self.height - py)
        if pw < 2 or ph < 2:
            return None
        return px, py, pw, ph

    def describe(self):
        return ("画布 %dx%d / 设备 %dx%d / %s (缩放 %.3f x %.3f)"
                % (self.width, self.height, self.dev_w, self.dev_h,
                   self.policy, self._sx, self._sy))