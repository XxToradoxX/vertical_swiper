# -*- coding: utf-8 -*-
r"""一次性把运行环境准备好(金币任务模式需要 numpy + opencv)。

    python tools\setup_env.py

做的事情:
1. 在项目目录下建 venv(已经有了就跳过)
2. 往里装 numpy + opencv-python-headless
3. 跑一遍离线自检,确认识别层能用

只用标准库,不依赖任何第三方包,所以能用系统 Python 直接跑。
opencv-python-headless 是不带 GUI 的版本:我们只用它做矩阵运算和模板匹配,
不需要它弹窗显示图片,体积小一半、也不会和 tkinter 抢窗口。
"""

import os
import subprocess
import sys

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV_DIR = os.path.join(APP_DIR, "venv")
VENV_PY = os.path.join(VENV_DIR, "Scripts" if os.name == "nt" else "bin",
                       "python.exe" if os.name == "nt" else "python")
PKGS = ["numpy", "opencv-python-headless"]


def run(cmd, **kw):
    print("+ %s" % " ".join(cmd))
    return subprocess.call(cmd, **kw)


def main():
    if not os.path.isfile(VENV_PY):
        print("== 建虚拟环境: %s" % VENV_DIR)
        if run([sys.executable, "-m", "venv", VENV_DIR]) != 0:
            print("建 venv 失败。")
            return 1
    else:
        print("== 已存在虚拟环境,复用: %s" % VENV_DIR)

    print("== 安装依赖: %s" % ", ".join(PKGS))
    if run([VENV_PY, "-m", "pip", "install", "--upgrade", "pip"]) != 0:
        print("升级 pip 失败(不影响使用,继续)。")
    if run([VENV_PY, "-m", "pip", "install"] + PKGS) != 0:
        print("装依赖失败。国内网络可以换镜像:")
        print('  %s -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple %s'
              % (VENV_PY, " ".join(PKGS)))
        return 1

    print("== 自检")
    rc = run([VENV_PY, os.path.join(APP_DIR, "tools", "selftest.py")])
    if rc != 0:
        print("自检没全过,看上面的 FAIL 行;识别框可能要按你的机型重标一遍。")
    print("\n完成。以后双击 `启动助手.bat` 就会用这个环境启动。")
    return rc


if __name__ == "__main__":
    sys.exit(main())