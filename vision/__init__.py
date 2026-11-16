# -*- coding: utf-8 -*-
"""多多视频金币任务的视觉层。

模块边界(从上到下,下层的不知道上层的存在):
    screen.py     adb 截图 / 点按 / 滑动 —— 只认坐标,不认内容
    canvas.py     设备分辨率 <-> 1080x1920 标准画布 —— 只做几何变换
    matcher.py    多尺度模板匹配 —— 只回答"这块区域里有没有它,分数多少"
    recognizer.py 模板 + ROI + 阈值 -> "当前是什么页面" —— 业务语义在这里
    coin_task.py  状态机 —— 只关心页面对不对、下一步该点哪

依赖 numpy + opencv(pip install numpy opencv-python-headless)。
"""

from .canvas import CANON_H, CANON_W, Canvas          # noqa: F401
from .matcher import Hit, TemplateSet, match          # noqa: F401
from .recognizer import Recognizer, View              # noqa: F401