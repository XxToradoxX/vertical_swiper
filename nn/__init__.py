# -*- coding: utf-8 -*-
"""图像识别的训练侧(只在这个文件夹里用 torch,推理端不依赖 torch)。

    data.py    真机截图 -> 带标注的训练样本(精确标 + 模板匹配补标)
    model.py   检测网络(CenterNet 式的格子级预测)
    train.py   GPU 训练
    export.py  导出 ONNX,给 cv2.dnn 用

训练用独立环境 venv_train(从 conda 的 mytorch 继承 torch,互不影响):
    venv_train\\Scripts\\python.exe nn\\train.py --epochs 30
    venv_train\\Scripts\\python.exe nn\\export.py
"""