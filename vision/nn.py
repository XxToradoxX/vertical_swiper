# -*- coding: utf-8 -*-
"""神经网络识别(推理侧)。

和 `vision/matcher.py`(模板匹配)并列的一条识别路径:
- 模板匹配:拿 12 张裁好的小图去整屏找,像素级对齐才对得上,**位置偏一点就掉分**;
- 神经网络:训练时就把"同一个控件出现在不同位置、不同尺度、不同底色"当成常态,
  所以框位置有偏移、缩放、被弹窗挡一角也能认出来。

这个模块**只用 cv2 + numpy**(不 import torch),发布包里跑的就是它。
模型文件默认在 `<程序目录>\\nn\\coindet.onnx`,由 `nn\\export.py` 导出。

用法:
    net = CoinNet()                 # 文件不存在时 available() 为 False
    dets = net.detect(canvas_img)   # [(class_name, score, x0, y0, x1, y1), ...]
"""

import os

import cv2
import numpy as np

# 与 nn/data.py 保持一致(那边是训练侧,不能 import,免得把 torch 拖进来)
INPUT_W, INPUT_H = 448, 800
CANVAS_W, CANVAS_H = 1080, 1920
GRID_W, GRID_H = INPUT_W // 8, INPUT_H // 8
STRIDE = 8
CLASSES = [
    "row_shop_video", "row_water", "row_live", "coin_entry",
    "bubble_watch", "bubble_done", "bubble_cooldown",
    "water_checkin_btn", "page_coin", "water_popup_title",
]

_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PATH = os.path.join(_APP_DIR, "nn", "coindet.onnx")


class CoinNet(object):
    """加载一次,反复用。第一次 forward 会慢一点(MIOpen/onnx 内部初始化)。"""

    def __init__(self, path=None, thr=0.35, topk=40):
        self.path = path or DEFAULT_PATH
        self.thr = float(thr)
        self.topk = int(topk)
        self._net = None
        self._tried = False
        self._ms = 0.0
        self.backend = None            # "CPU(cv2.dnn)" / "GPU(cv2.dnn CUDA)"

    def available(self):
        if not self._tried:
            self._tried = True
            if os.path.exists(self.path):
                try:
                    self._net = cv2.dnn.readNetFromONNX(self.path)
                    self.backend = _pick_backend(self._net)
                except Exception:
                    self._net = None
        return self._net is not None

    def describe(self):
        """日志用:这条识别路径跑在 CPU 还是 GPU 上。"""
        if not self.available():
            return "不可用(没有 %s)" % os.path.basename(self.path)
        return "ONNX / cv2.dnn %s" % self.backend

    # ---------------------------------------------------------------- 推理
    def _blob(self, canvas_img):
        small = cv2.resize(canvas_img, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return cv2.dnn.blobFromImage(rgb, 1.0, (INPUT_W, INPUT_H), (0, 0, 0),
                                     swapRB=False, crop=False)

    def detect(self, canvas_img, thr=None, topk=None):
        if not self.available():
            return []
        thr = self.thr if thr is None else float(thr)
        topk = self.topk if topk is None else int(topk)

        self._net.setInput(self._blob(canvas_img))
        import time
        t0 = time.time()
        outs = self._net.forward(self._net.getUnconnectedOutLayersNames())
        self._ms = (time.time() - t0) * 1000.0

        hm = np.asarray(outs[0]).reshape(GRID_H, GRID_W)
        cls = np.asarray(outs[1]).reshape(len(CLASSES), GRID_H, GRID_W)
        wh = np.asarray(outs[2]).reshape(2, GRID_H, GRID_W)
        off = np.asarray(outs[3]).reshape(2, GRID_H, GRID_W)
        return decode(hm, cls, wh, off, thr=thr, topk=topk)

    @property
    def last_ms(self):
        return self._ms


def cuda_devices():
    """本机 cv2 能看到几块 CUDA 显卡(普通 headless wheel 是 0)。"""
    try:
        return int(cv2.cuda.getCudaEnabledDeviceCount())
    except Exception:
        return 0


def _pick_backend(net):
    """有 CUDA 版 cv2 就上 GPU,否则老老实实 CPU;返回日志用的名字。"""
    if cuda_devices() <= 0:
        return "CPU(无 CUDA 版 OpenCV)"
    try:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        return "GPU(CUDA)"
    except Exception:
        try:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        except Exception:
            pass
        return "CPU(CUDA 初始化失败)"


def decode(hm, cls, wh, off, thr=0.35, topk=40):
    """网络输出 -> [(class_name, score, x0, y0, x1, y1)](画布坐标)。

    和 nn/data.py 的 decode 同一套规则:hm 过 sigmoid 取 topk,分数乘类别概率,
    再用 wh/off 把格子还原成框。
    """
    prob = 1.0 / (1.0 + np.exp(-hm.reshape(-1)))
    cls_prob = 1.0 / (1.0 + np.exp(-cls))          # 类别头也是 logit,要过 sigmoid
    idx = np.argsort(-prob)[:topk]
    sx = CANVAS_W / float(INPUT_W)
    sy = CANVAS_H / float(INPUT_H)
    out = []
    for i in idx:
        score = float(prob[i])
        if score < thr:
            break
        gy, gx = divmod(int(i), GRID_W)
        c = int(np.argmax(cls_prob[:, gy, gx]))
        cscore = float(cls_prob[c, gy, gx])
        if cscore < 0.25:
            continue
        bw = float(wh[0, gy, gx]) * STRIDE
        bh = float(wh[1, gy, gx]) * STRIDE
        if bw <= 1 or bh <= 1:
            continue
        cx = (gx + float(off[0, gy, gx])) * STRIDE
        cy = (gy + float(off[1, gy, gx])) * STRIDE
        out.append((
            CLASSES[c], score * cscore,
            (cx - bw / 2) * sx, (cy - bh / 2) * sy,
            (cx + bw / 2) * sx, (cy + bh / 2) * sy,
        ))
    return out


def overlay(canvas_img, dets, path=None):
    """把检测结果画在图上(调参/排查用)。"""
    img = canvas_img.copy()
    for name, score, x0, y0, x1, y1 in dets:
        c = (0, 200, 0) if score >= 0.5 else (0, 165, 255)
        cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), c, 2)
        cv2.putText(img, "%s %.2f" % (name, score), (int(x0), max(14, int(y0) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
    if path:
        import vision.imgio as imgio
        imgio.imwrite(path, img)
    return img
