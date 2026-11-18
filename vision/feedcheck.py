# -*- coding: utf-8 -*-
"""挂机时的「还在信息流吗」快速判断(推理侧)。

比 `Recognizer.classify()` 便宜两个数量级:
- classify:22 张模板 × 多尺度 + 可选神经网络,一帧 0.8~3.6 s;
- 这里:一张 128x228 的小图过一次 7.6 万参数的 CNN。纯推理
  CPU(cv2.dnn)2~3 ms、GPU(torch+CUDA)3 ms 左右;算上把整张截屏
  缩到 128x228 的 resize,一帧总共 8~14 ms,是截图(1.4 s)的零头。

挂机循环因此可以"每滑一条只看一眼":判成信息流就直接滑,判成别的
才退回完整识别,去分辨"是金币页(待机)还是落地页(上滑+返回)"。

两条后端,分数必须一致(导出时 `nn\\export_feed.py` 会校验):
- `onnx`:cv2.dnn 跑 `nn\\feednet.onnx`,只要 opencv,Scroller 环境就是这条;
- `torch`:torch 跑 `nn\\feednet.pt`,有 CUDA 就上 GPU(项目里的 `env_gpu`)。

用法:
    chk = FeedCheck()                    # auto:有 torch+CUDA 走 GPU,否则 ONNX
    ok, s = chk.is_feed(canvas_bgr)      # (True/False, 0~1 的分数)
"""

import os
import time

import cv2
import numpy as np

from nn.feed import DEFAULT_THR, INPUT_H, INPUT_W, MEAN, STD   # 与训练侧同一份常量

_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ONNX = os.path.join(_APP_DIR, "nn", "feednet.onnx")
DEFAULT_PT = os.path.join(_APP_DIR, "nn", "feednet.pt")


def _blob(img_bgr):
    """和 nn/feed.py 的 preprocess 完全等价(BN 顺序/归一化都不能差)。"""
    small = cv2.resize(img_bgr, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)
    x = small.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return np.transpose(x, (2, 0, 1))[None, ...].copy()


class FeedCheck(object):
    """加载一次反复用。第一次调用会多花几十毫秒(初始化后端)。"""

    def __init__(self, onnx_path=None, pt_path=None, backend="auto",
                 thr=DEFAULT_THR, device=None):
        self.onnx_path = onnx_path or DEFAULT_ONNX
        self.pt_path = pt_path or DEFAULT_PT
        self.thr = float(thr)
        self.want = backend
        self.backend = None
        self._net = None
        self._model = None
        self._torch = None
        self._device = device
        self._ms = 0.0
        self._tried = False

    # ------------------------------------------------------------ 后端
    def available(self):
        if not self._tried:
            self._tried = True
            if self.want in ("auto", "torch"):
                # auto 的规矩:有能用的显卡才走 torch(GPU);没显卡时 torch-CPU
                # 比 cv2.dnn 慢,所以退回 ONNX。显式 backend="torch" 时照旧强上。
                if self.want == "torch" or self._cuda_ready():
                    self._try_torch()
            if self._net is None and self._model is None and self.want in ("auto", "onnx"):
                self._try_onnx()
        return self._net is not None or self._model is not None

    @staticmethod
    def _cuda_ready():
        """本机到底能不能用显卡(torch 装没装 + 驱动认不认)。"""
        try:
            import torch
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def _try_torch(self):
        try:
            import torch                     # noqa: F401  (装没装决定走哪条路)
            from nn.feed import build_model
        except Exception:
            return
        if not os.path.exists(self.pt_path):
            return
        try:
            dev = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            ck = torch.load(self.pt_path, map_location=dev, weights_only=False)
            model = build_model()
            model.load_state_dict(ck["state"])
            model.eval().to(dev)
            for p in model.parameters():
                p.requires_grad = False
            self._torch = torch
            self._model = model
            self._device = dev
            self.backend = "torch:%s" % dev
        except Exception:
            self._model = None

    def _try_onnx(self):
        if not os.path.exists(self.onnx_path):
            return
        try:
            self._net = cv2.dnn.readNetFromONNX(self.onnx_path)
            self.backend = self.backend or "onnx"
            # 试着把 ONNX 也塞进 GPU(cv2 必须是 CUDA 版才有效)。
            # cv2 5.x 的新图引擎还不支持选后端,直接跳过,免得刷两条 WARN。
            if int(cv2.__version__.split(".")[0]) < 5:
                try:
                    self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                    if not self._cuda_ok():
                        self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                        self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                except Exception:
                    pass
        except Exception:
            self._net = None

    def _cuda_ok(self):
        """cv2.dnn 的 CUDA 后端在 headless wheel 里并不存在,试跑一次就知道。"""
        try:
            self._net.setInput(np.zeros((1, 3, INPUT_H, INPUT_W), dtype=np.float32))
            self._net.forward()
            self.backend = "onnx:cuda"
            return True
        except Exception:
            return False

    # ------------------------------------------------------------ 推理
    def score(self, canvas_bgr):
        """返回 0~1(越大越像信息流);后端不可用时返回 None。"""
        if not self.available() or canvas_bgr is None:
            return None
        t0 = time.time()
        blob = _blob(canvas_bgr)
        if self._model is not None:
            torch = self._torch
            with torch.no_grad():
                x = torch.from_numpy(blob).to(self._device)
                logit = self._model(x)
                s = float(torch.sigmoid(logit).reshape(-1)[0])
        else:
            self._net.setInput(blob)
            logit = float(np.asarray(self._net.forward()).reshape(-1)[0])
            s = 1.0 / (1.0 + float(np.exp(-logit)))
        self._ms = (time.time() - t0) * 1000.0
        return s

    def is_feed(self, canvas_bgr, thr=None):
        thr = self.thr if thr is None else float(thr)
        s = self.score(canvas_bgr)
        if s is None:
            return None, None
        return s >= thr, s

    @property
    def last_ms(self):
        return self._ms

    def describe(self):
        """给日志用的一句话:跑在哪儿、什么后端。"""
        if not self.available():
            return "不可用"
        if self.backend == "onnx":
            return "ONNX / cv2.dnn(CPU,%s)" % cv2.__version__
        if self.backend == "onnx:cuda":
            return "ONNX / cv2.dnn(CUDA)"
        if self.backend and self.backend.startswith("torch:"):
            dev = self.backend.split(":", 1)[1]
            where = "GPU" if dev.startswith("cuda") else "CPU"
            return "torch %s(%s)" % (self._torch.__version__, where + " " + dev)
        return str(self.backend or "未知")