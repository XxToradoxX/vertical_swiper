# -*- coding: utf-8 -*-
"""图片读写小工具。

Windows 上 cv2.imread / cv2.imwrite 不支持非 ASCII 路径,项目路径和
截图文件名都可能有中文,所以统一走 np.fromfile + imdecode。
"""

import os

import cv2
import numpy as np


def imread(path, flags=cv2.IMREAD_COLOR):
    """读图,支持中文路径。失败返回 None。"""
    try:
        buf = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    img = cv2.imdecode(buf, flags)
    return img


def imwrite(path, img):
    """写图,支持中文路径。成功返回 True。"""
    if img is None:
        return False
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    try:
        parent = os.path.dirname(os.path.abspath(path))
        if parent and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)
        buf.tofile(path)
    except OSError:
        return False
    return True