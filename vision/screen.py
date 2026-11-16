# -*- coding: utf-8 -*-
"""adb 屏幕读写:截图 + 点按 + 滑动。

只做"拿到一帧像素"和"发一次输入"这两件事,不做任何识别判断,
这样换设备/换 adb 版本时只需要动这个文件。

截图走 `adb exec-out screencap -p`,拿到的是 PNG 流。为什么不用
`screencap /sdcard/x.png` + pull:那样要写手机存储再读回来,一来慢
(多一次文件传输),二来某些机型会因为没有存储权限失败。
"""

import os
import re
import subprocess
import time

import cv2
import numpy as np

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

# "mResumedActivity: ActivityRecord{... u0 com.xunmeng.pinduoduo/.ui.MainFrameActivity t123}"
# 里取出 "com.xunmeng.pinduoduo"
_FG_RE = re.compile(r"([A-Za-z][\w.]*)/[A-Za-z$][\w.$]*")
_FG_PROBES = (
    "dumpsys activity activities | grep -E 'mResumedActivity|topResumedActivity'",
    "dumpsys window | grep -E 'mCurrentFocus|mFocusedApp'",
)

# pythonw 下没有控制台;不传这个标志的话,每跑一次 adb 都会闪一个黑色
# cmd 窗口出来(截一次图就闪一次,基本没法用电脑)。
CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


def _run(cmd, timeout, text=False):
    """统一的 adb 调用入口:不弹窗、不吃控制台输入。"""
    kwargs = dict(capture_output=True, timeout=timeout, stdin=subprocess.DEVNULL)
    if CREATE_NO_WINDOW:
        kwargs["creationflags"] = CREATE_NO_WINDOW
        # 双保险:有些系统上 adb 会忽略 creationflags,再显式要求隐藏窗口
        info = subprocess.STARTUPINFO()
        info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        info.wShowWindow = subprocess.SW_HIDE
        kwargs["startupinfo"] = info
    return subprocess.run(cmd, **kwargs)


class ScreenError(Exception):
    pass


def list_devices(adb="adb", timeout=15):
    """`adb devices` 里状态为 device 的序列号列表。"""
    try:
        proc = _run([adb, "devices"], timeout)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    text = proc.stdout.decode("utf-8", "replace")
    out = []
    for line in text.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "device":
            out.append(parts[0])
    return out


def pick_serial(serials):
    """从多个设备里挑一个。

    无线调试时同一台手机常会同时出现两条:`192.168.x.x:port` 和
    `adb-xxxx-xxxx._adb-tls-connect._tcp`(mDNS 自动发现)。它们是同一台机,
    随便挑一条都能用,但 mDNS 那条的端口会变,优先挑带 IP:port 的。
    """
    if not serials:
        return ""
    for s in serials:
        if "_adb-tls-connect" not in s:
            return s
    return serials[0]


class Screen:
    def __init__(self, adb="adb", serial=None, timeout=20, log=None):
        self.adb = adb or "adb"
        self.serial = serial or ""
        self.timeout = timeout
        self.log = log
        self._size = None
        self.last_error = ""

    # -- 底层 ----------------------------------------------------------
    def _resolve_serial(self):
        """没指定设备时自动挑一个。避免 `more than one device/emulator`。"""
        if self.serial:
            return self.serial
        serials = list_devices(self.adb, self.timeout)
        self.serial = pick_serial(serials)
        if self.serial and self.log:
            self.log("自动选择设备: %s" % self.serial)
        return self.serial

    def _cmd(self, args, binary=False):
        cmd = [self.adb]
        serial = self._resolve_serial()
        if serial:
            cmd += ["-s", serial]
        cmd += list(args)
        try:
            proc = _run(cmd, self.timeout)
        except FileNotFoundError:
            raise ScreenError("找不到 adb: %s" % self.adb)
        except subprocess.TimeoutExpired:
            raise ScreenError("adb 超时: %s" % " ".join(args))
        if proc.returncode != 0:
            msg = proc.stderr.decode("utf-8", "replace").strip() or proc.stdout[:200].decode("utf-8", "replace")
            self.last_error = msg
            raise ScreenError("adb 失败(%d): %s" % (proc.returncode, msg))
        return proc.stdout if binary else proc.stdout.decode("utf-8", "replace")

    def shell(self, *args):
        return self._cmd(["shell"] + [str(a) for a in args])

    # -- 设备 ----------------------------------------------------------
    def size(self):
        """设备像素宽高。优先 wm size,失败退回截图尺寸。"""
        if self._size:
            return self._size
        try:
            out = self.shell("wm", "size")
            # 例: "Physical size: 1264x2780" / "Override size: 1080x2400"
            for line in out.splitlines():
                if ":" in line and "x" in line:
                    val = line.split(":", 1)[1].strip()
                    w, h = val.split("x")
                    self._size = (int(w), int(h))
                    return self._size
        except (ScreenError, ValueError):
            pass
        img = self.capture()
        if img is None:
            raise ScreenError("取不到屏幕尺寸(%s)" % (self.last_error or "截图失败"))
        self._size = (img.shape[1], img.shape[0])
        return self._size

    # -- 前台 / 启动 ---------------------------------------------------
    def foreground_pkg(self):
        """当前前台应用的包名,取不到返回 ""。

        先问 activity(最准),再退回 window 焦点。注意下拉通知栏时
        mCurrentFocus 会变成 NotificationShade 这种没有包名的窗口,所以两条
        都要试,取第一条能解析出包名的。
        """
        for probe in _FG_PROBES:
            try:
                out = self.shell(probe)
            except ScreenError:
                continue
            match = _FG_RE.search(out)
            if match:
                return match.group(1)
        return ""

    def launch_app(self, pkg):
        """把 App 拉到前台(只保证启动,不保证进到哪个二级页面)。"""
        try:
            self.shell("monkey -p %s -c android.intent.category.LAUNCHER 1" % pkg)
            return True
        except ScreenError:
            return False

    # -- 截图 ----------------------------------------------------------
    def capture(self):
        """截一帧,返回 BGR ndarray。失败返回 None。"""
        try:
            data = self._cmd(["exec-out", "screencap", "-p"], binary=True)
        except ScreenError:
            return None
        # 有的 adb/机型会混进 CRLF,把 PNG 头切坏
        if not data.startswith(PNG_MAGIC):
            idx = data.find(PNG_MAGIC)
            if idx < 0:
                return None
            data = data[idx:]
        buf = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return img

    def capture_settled(self, tries=4, delay=0.25, diff_thr=1.5):
        """连续两帧基本一致才算"画面稳定了",避免在动画中间做判断。

        相似度用平均绝对差:整屏平均差 < diff_thr 灰度级就算稳定。
        """
        prev = self.capture()
        for _ in range(max(1, tries)):
            if prev is None:
                return None
            time.sleep(delay)
            cur = self.capture()
            if cur is None:
                return prev
            if cur.shape == prev.shape:
                diff = float(np.mean(cv2.absdiff(cur, prev)))
                if diff < diff_thr:
                    return cur
            prev = cur
        return prev

    # -- 输入(坐标都是设备像素)----------------------------------------
    def tap(self, x, y):
        self.shell("input", "tap", int(x), int(y))

    def swipe(self, x1, y1, x2, y2, ms=260):
        self.shell("input", "swipe", int(x1), int(y1), int(x2), int(y2), int(ms))

    def back(self):
        self.shell("input", "keyevent", 4)

    def key(self, code):
        self.shell("input", "keyevent", int(code))