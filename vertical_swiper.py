#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
竖屏视频自动上滑助手 (Auto Swiper)

通过 ADB 向手机发送"上滑"手势,模拟一条一条往下刷竖屏视频。
不绑定某个 App:抖音、拼多多(多多视频)、哔哩哔哩竖屏、快手、小红书……
凡是"上滑换下一条"的界面都适用,默认就刷当前手机屏幕上的内容。
默认冷却时间 3–6 秒随机(每次上滑后重新抽一个值),滑动时长可调。

用法:
    双击 "启动助手.bat"                     # 图形界面
    python vertical_swiper.py --cli --interval 5        # 命令行(无界面,刷当前界面)
    python vertical_swiper.py --cli --app pdd --guard   # 只刷拼多多,且限前台
    python vertical_swiper.py --dry-run                 # 只打印坐标,不动手机

依赖: 已安装的 adb(Platform-Tools)、手机已开启 USB 调试或无线调试。
"""

import argparse
import glob
import json
import os
import queue
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, asdict

APP_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(APP_DIR, "config.json")
CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0
APP_TITLE = "竖屏视频自动上滑助手"
ERROR_LOG = os.path.join(APP_DIR, "swiper_error.log")
LOG_DIR = os.path.join(APP_DIR, "logs")
KEEP_LOGS = 60          # 只保留最近这么多次运行的日志


class RunLog:
    """运行日志:界面里看到的每一条,同时落到 logs\\run_<时间>_<模式>.log。

    出问题时让用户把那个文件发过来就行,不用凭记忆描述"当时屏幕上是啥"。
    写失败(只读目录、磁盘满)一律静默 —— 日志不能让程序挂掉。
    """

    def __init__(self, tag="run"):
        self.path = ""
        self._lock = threading.Lock()
        self._fh = None
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            self.path = os.path.join(LOG_DIR, "run_%s_%s.log" % (stamp, tag))
            self._fh = open(self.path, "a", encoding="utf-8", buffering=1)
            self._write_raw("=== %s 启动 (模式:%s) ===" % (
                time.strftime("%Y-%m-%d %H:%M:%S"), tag))
            self._write_raw("Python %s / %s / %s" % (
                sys.version.split()[0], os.name, APP_DIR))
            self._prune()
        except Exception:
            self._fh = None

    def _write_raw(self, line):
        try:
            self._fh.write(line + "\n")
        except Exception:
            pass

    def write(self, msg):
        """写一条带时间戳的日志(与界面上的那条一致)。"""
        text = str(msg)
        with self._lock:
            if self._fh is None:
                return
            stamp = time.strftime("%H:%M:%S")
            for line in text.splitlines() or [""]:
                self._write_raw("[%s] %s" % (stamp, line))

    def close(self, summary=""):
        with self._lock:
            if self._fh is None:
                return
            if summary:
                self._write_raw(summary)
            self._write_raw("=== %s 结束 ===" % time.strftime("%Y-%m-%d %H:%M:%S"))
            try:
                self._fh.close()
            except Exception:
                pass
            self._fh = None

    @staticmethod
    def _prune():
        try:
            files = sorted(glob.glob(os.path.join(LOG_DIR, "run_*.log")))
            for path in files[:-KEEP_LOGS]:
                try:
                    os.remove(path)
                except Exception:
                    pass
        except Exception:
            pass


def report_fatal(exc_text):
    """pythonw 启动时没有控制台:把堆栈写进日志并弹窗,避免"双击了没反应"。"""
    try:
        with open(ERROR_LOG, "w", encoding="utf-8") as f:
            f.write(exc_text)
    except Exception:
        pass
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(
                0, exc_text[-1200:] + "\n\n完整信息已写入:\n" + ERROR_LOG,
                APP_TITLE + " - 启动失败", 0x10)
        except Exception:
            pass


# 常见竖屏视频入口。包名不写死:<app>/<activity> 启动,只保证把 App 拉到前台。
# 金币任务模式的模板、坐标、页面结构全是拼多多家的,所以只有这个包才有那个模式。
COIN_APP_PKG = "com.xunmeng.pinduoduo"
APP_PRESETS = [
    ("保持当前界面", ""),
    ("抖音", "com.ss.android.ugc.aweme"),
    ("拼多多·多多视频", "com.xunmeng.pinduoduo"),
    ("哔哩哔哩·竖屏", "tv.danmaku.bili"),
    ("快手", "com.smile.gifmaker"),
    ("小红书", "com.xingin.xhs"),
    ("微信·视频号", "com.tencent.mm"),
    ("自定义…", "__custom__"),
]

APP_ALIASES = {
    "douyin": "com.ss.android.ugc.aweme",
    "pdd": "com.xunmeng.pinduoduo",
    "pinduoduo": "com.xunmeng.pinduoduo",
    "bili": "tv.danmaku.bili",
    "bilibili": "tv.danmaku.bili",
    "kuaishou": "com.smile.gifmaker",
    "xhs": "com.xingin.xhs",
    "weixin": "com.tencent.mm",
    "none": "",
}


# ---------------------------------------------------------------- 配置

@dataclass
class Config:
    adb_path: str = ""                 # adb.exe 路径,空则自动查找
    serial: str = ""                   # 设备序列号,空则自动选择
    interval_min: float = 3.0          # 上滑冷却时间下限(秒)
    interval_max: float = 6.0          # 上滑冷却时间上限(秒),每次在这个区间内随机
    duration_ms: int = 260             # 单次滑动时长(毫秒),太慢会变成拖拽
    keep_awake: bool = True            # 运行时保持屏幕常亮
    target_pkg: str = ""               # 目标应用包名;空 = 不限制,刷当前界面
    launch_app: bool = False           # 开始时把目标应用拉到前台
    guard_app: bool = False            # 仅当目标应用在前台时才上滑
    # ---- 识别模式(多多视频金币任务)----
    mode: str = "swipe"                # swipe = 纯定时上滑, coins = 金币任务
    coin_modules: str = "shop_video,live,water"   # 要做的模块,逗号分隔
    coin_max_minutes: int = 60         # 一轮最多跑多久
    coin_done_delay: int = 10          # 倒计时完成后等几秒再返回
    coin_monitor_interval: float = 3.5   # 倒计时中每隔几秒滑动一次(须小于 5 秒)
    coin_swipe_duration: int = 260     # 续命滑动时长(毫秒)
    coin_debug: bool = False           # 把每次识别的画框图存到 debug 目录
    coin_debug_dir: str = ""           # 识别截图存到哪,空 = 程序目录\debug
    coin_idle: bool = True             # 任务跑完后是否退回视频流挂机刷视频
    coin_idle_interval: float = 3.5    # 挂机时两次滑动的间隔(秒)
    coin_nn: bool = True               # 识别优先用训练出来的神经网络(模型在则生效)


def load_config():
    cfg = Config()
    if os.path.isfile(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            for key, val in data.items():
                if not hasattr(cfg, key):
                    continue
                cur = getattr(cfg, key)
                try:
                    if isinstance(cur, bool):
                        setattr(cfg, key, bool(val))
                    elif isinstance(cur, float):
                        setattr(cfg, key, float(val))
                    elif isinstance(cur, int):
                        setattr(cfg, key, int(val))
                    elif isinstance(cur, str):
                        setattr(cfg, key, str(val))
                except (TypeError, ValueError):
                    pass
            # 兼容旧版固定间隔 + 抖动:换算成区间
            if "interval_min" not in data and "interval" in data:
                base = float(data.get("interval") or 5.0)
                jit = float(data.get("jitter") or 0.0)
                cfg.interval_min = base - jit
                cfg.interval_max = base + jit
            # 兼容旧版"抖音专用"配置
            if "target_pkg" not in data and "douyin_pkg" in data:
                cfg.target_pkg = str(data.get("douyin_pkg") or "")
            if "launch_app" not in data and "auto_launch" in data:
                cfg.launch_app = bool(data.get("auto_launch"))
        except Exception:
            pass
    if not cfg.adb_path or not os.path.isfile(cfg.adb_path):
        cfg.adb_path = find_adb(cfg.adb_path)
    normalize_range(cfg)
    return cfg


def normalize_range(cfg):
    """保证冷却区间合法:下限至少 0.5 秒,上限不小于下限。"""
    try:
        low = max(0.5, float(cfg.interval_min))
        high = max(low, float(cfg.interval_max))
    except (TypeError, ValueError):
        low, high = 3.0, 6.0
    cfg.interval_min, cfg.interval_max = low, high


def save_config(cfg):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ---------------------------------------------------------------- ADB 封装

def find_adb(preferred=""):
    candidates = []
    if preferred:
        candidates.append(preferred)
    which = shutil.which("adb")
    if which:
        candidates.append(which)
    candidates += [
        os.path.join(APP_DIR, "platform-tools", "adb.exe"),
        r"D:\software\adb\platform-tools\adb.exe",
        r"D:\adb\platform-tools\adb.exe",
        r"C:\platform-tools\adb.exe",
        r"C:\Android\platform-tools\adb.exe",
        r"C:\adb\adb.exe",
        "/usr/local/bin/adb",
        "/usr/bin/adb",
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return "adb"


def adb_run(cfg, args, timeout=20):
    """执行 adb 命令,返回 (返回码, 合并后的输出)。"""
    exe = cfg.adb_path or find_adb()
    cmd = [exe] + list(args)
    extra = {}
    if CREATE_NO_WINDOW:
        extra["creationflags"] = CREATE_NO_WINDOW
        info = subprocess.STARTUPINFO()
        info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        info.wShowWindow = subprocess.SW_HIDE
        extra["startupinfo"] = info
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",   # adb 与手机端输出都是 UTF-8,别用系统 GBK 猜
            timeout=timeout,
            stdin=subprocess.DEVNULL,   # 没控制台时别让它等输入
            errors="replace",
            **extra,
        )
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except FileNotFoundError:
        return 127, "找不到 adb 程序,请在界面里指定 adb.exe 的完整路径。"
    except subprocess.TimeoutExpired:
        return 124, "adb 命令超时(手机可能已断开或卡住)。"
    except Exception as exc:  # noqa: BLE001
        return 1, "调用 adb 失败: %s" % exc


def list_devices(cfg):
    rc, out = adb_run(cfg, ["devices"], timeout=15)
    if rc != 0:
        return []
    devices = []
    for line in out.splitlines()[1:]:
        line = line.strip()
        if not line or line.startswith("*"):
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "device":
            devices.append(parts[0])
    return devices


def query_screen_size(cfg, serial):
    args = (["-s", serial] if serial else []) + ["shell", "wm", "size"]
    rc, out = adb_run(cfg, args, timeout=15)
    found = re.findall(r"(\d+)\s*x\s*(\d+)", out or "")
    if not found:
        raise RuntimeError("读取屏幕分辨率失败: %s" % (out.strip() or "无输出"))
    width, height = found[-1]
    return int(width), int(height)


# ---------------------------------------------------------------- 核心逻辑

class SwiperCore:
    """负责"每 N 秒上滑一次"的循环。GUI 与命令行共用。"""

    def __init__(self, cfg, log=None, dry_run=False):
        self.cfg = cfg
        self.log = log or (lambda msg: print(msg, flush=True))
        self.dry_run = dry_run
        self.count = 0
        self.next_at = 0.0
        self.last_error = ""
        self._size = None
        self._awake = False
        self._last_fg = ""
        self._stop = threading.Event()
        self._thread = None

    # -- 设备准备 ------------------------------------------------------
    def _adb(self, args, timeout=20):
        base = (["-s", self.cfg.serial] if self.cfg.serial else []) + list(args)
        return adb_run(self.cfg, base, timeout=timeout)

    def ensure_device(self):
        devices = list_devices(self.cfg)
        if self.cfg.serial in devices:
            return True
        # 无线调试:设备掉线时尝试重连一次
        if self.cfg.serial and ":" in self.cfg.serial:
            adb_run(self.cfg, ["connect", self.cfg.serial], timeout=15)
            devices = list_devices(self.cfg)
            if self.cfg.serial in devices:
                return True
        if devices:
            self.cfg.serial = devices[0]
            self.log("自动选择设备: %s" % devices[0])
            return True
        self.log("未检测到已连接的设备,请确认 USB 调试或无线调试已开启。")
        return False

    def probe_screen(self):
        self._size = query_screen_size(self.cfg, self.cfg.serial)
        self.log("屏幕分辨率: %d x %d" % self._size)
        return self._size

    def _launch_target(self):
        pkg = self.cfg.target_pkg
        if not pkg:
            return
        rc, out = self._adb(["shell", "monkey", "-p", pkg,
                             "-c", "android.intent.category.LAUNCHER", "1"], timeout=20)
        if rc == 0 and "No activities found" not in out:
            self.log("已切换到 %s" % pkg)
        else:
            lines = (out or "").strip().splitlines()
            detail = lines[-1] if lines else "无输出"
            self.log("启动 %s 失败(可能未安装): %s" % (pkg, detail.strip()[:120]))

    # 不同 Android 版本 / 厂商 ROM 把前台应用记在不同字段,按顺序往下试。
    # mFocusedApp 放在前面:下拉通知栏、锁屏时 mCurrentFocus 变成系统窗口,
    # 但 mFocusedApp 仍然是那个 App,拿来判断"现在在哪个 App"更准。
    FG_PROBES = (
        ('dumpsys window | grep -E "mCurrentFocus|mFocusedApp"', (
            r"mFocusedApp=\w+\{[^}]*?\s([A-Za-z][\w.]*)/",
            r"mCurrentFocus=\w+\{[^}]*?\s([A-Za-z][\w.]*)/",
        )),
        ('dumpsys activity activities | grep -E "mResumedActivity|topResumedActivity"', (
            r"(?:m|topR)esumedActivity=\w+\{[^}]*?\s([A-Za-z][\w.]*)/",
        )),
    )

    def current_package(self):
        """返回当前前台应用包名;识别不出时返回空串。"""
        for cmd, patterns in self.FG_PROBES:
            rc, out = self._adb(["shell", cmd], timeout=15)
            if rc != 0 or not out:
                continue
            for pattern in patterns:
                match = re.search(pattern, out)
                if match:
                    return match.group(1)
        return ""

    # -- 单次上滑 ------------------------------------------------------
    def _points(self):
        width, height = self._size
        x1 = width * 0.5 + random.uniform(-0.05, 0.05) * width
        y1 = height * 0.78 + random.uniform(-0.02, 0.02) * height
        x2 = x1 + random.uniform(-0.02, 0.02) * width
        y2 = height * 0.17 + random.uniform(-0.02, 0.02) * height
        return int(x1), int(y1), int(x2), int(y2)

    def swipe_once(self, quiet=False):
        if not self._size:
            self.probe_screen()
        x1, y1, x2, y2 = self._points()
        duration = max(80, int(self.cfg.duration_ms + random.uniform(-50, 50)))
        if self.dry_run:
            self.log("[演练] input swipe %d %d %d %d %d" % (x1, y1, x2, y2, duration))
            return True
        rc, out = self._adb(["shell", "input", "swipe",
                             str(x1), str(y1), str(x2), str(y2), str(duration)])
        if rc != 0:
            self.last_error = (out or "").strip()
            self.log("上滑失败: %s" % (self.last_error or "未知错误"))
            return False
        if not quiet:
            self.log("已上滑 (%d,%d) -> (%d,%d),用时 %d ms" % (x1, y1, x2, y2, duration))
        return True

    # -- 循环 ----------------------------------------------------------
    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        if self.running:
            return False
        if not self.ensure_device():
            return False
        if self.cfg.keep_awake:
            rc, _ = self._adb(["shell", "svc", "power", "stayon", "true"], timeout=15)
            if rc == 0:
                self._awake = True
                self.log("已开启屏幕常亮")
        if self.cfg.launch_app and self.cfg.target_pkg:
            self._launch_target()
        self.count = 0
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="swiper", daemon=True)
        self._thread.start()
        return True

    def stop(self, quiet=False):
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=4)
        self._thread = None
        self.next_at = 0.0
        if self._awake and not self.dry_run:
            self._awake = False
            self._adb(["shell", "svc", "power", "stayon", "false"], timeout=10)
        if not quiet:
            self.log("已停止,本次共上滑 %d 次" % self.count)

    def interval_text(self):
        low, high = self.cfg.interval_min, self.cfg.interval_max
        if abs(high - low) < 0.05:
            return "每 %.1f 秒上滑一次" % low
        return "冷却时间 %.1f–%.1f 秒随机" % (low, high)

    def _wait_next(self):
        delay = max(0.5, random.uniform(self.cfg.interval_min, self.cfg.interval_max))
        self.next_at = time.monotonic() + delay
        self._stop.wait(delay)

    def _loop(self):
        self.log("开始运行," + self.interval_text())
        if self.cfg.guard_app and self.cfg.target_pkg:
            self.log("前台保护已开启:只在 %s 在前台时上滑" % self.cfg.target_pkg)
        fails = 0
        self._last_fg = ""
        while not self._stop.is_set():
            # 前台保护:目标 App 不在最前面时,滑下去只会误伤别的界面
            if self.cfg.guard_app and self.cfg.target_pkg:
                fg = self.current_package()
                if fg and fg != self.cfg.target_pkg:
                    if fg != self._last_fg:
                        self._last_fg = fg
                        self.log("当前前台是 %s,不是目标应用,暂停上滑" % fg)
                    self._wait_next()
                    continue
                if fg == self.cfg.target_pkg and self._last_fg:
                    self.log("目标应用已回到前台,继续上滑")
                self._last_fg = ""

            try:
                if self._size is None:
                    self.probe_screen()
                ok = self.swipe_once(quiet=True)
            except Exception as exc:  # noqa: BLE001
                ok = False
                self.last_error = str(exc)
                self.log("出错: %s" % exc)

            if self._stop.is_set():
                break

            if ok:
                fails = 0
                self.count += 1
                self.log("第 %d 次上滑完成" % self.count)
            else:
                fails += 1
                if fails >= 5:
                    self.log("连续失败 5 次,已自动停止。请检查手机连接是否正常。")
                    break

            self._wait_next()
        self.next_at = 0.0


# ---------------------------------------------------------------- 金币任务

class CoinRunner:
    """多多视频金币任务。接口和 SwiperCore 对齐,GUI 那段代码不用分叉。"""

    def __init__(self, cfg, log=None, dry_run=False):
        self.cfg = cfg
        self.log = log or (lambda msg: print(msg, flush=True))
        self.dry_run = dry_run
        self.count = 0
        self.next_at = 0.0
        self.status = "待开始"
        self.last_error = ""
        self._stop = threading.Event()
        self._thread = None
        self._task = None
        self._awake = False

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    def _make_task(self):
        """延迟导入:没装 opencv 的人仍然能用纯滑动模式。"""
        from vision.coin_task import CoinTask
        from vision.recognizer import Recognizer
        from vision.screen import Screen

        screen = Screen(adb=self.cfg.adb_path or "adb", serial=self.cfg.serial, log=self.log)
        modules = [m.strip() for m in (self.cfg.coin_modules or "").split(",") if m.strip()]
        task_cfg = {
            "max_minutes": max(1, int(self.cfg.coin_max_minutes)),
            "done_delay": max(0, int(self.cfg.coin_done_delay)),
            "monitor_interval": max(1.5, float(self.cfg.coin_monitor_interval)),
            "swipe_duration": max(80, int(self.cfg.coin_swipe_duration)),
            "modules": modules or ["shop_video", "live", "water"],
            "target_pkg": self.cfg.target_pkg or COIN_APP_PKG,
            "launch_app": bool(self.cfg.launch_app),
            "idle_after": bool(self.cfg.coin_idle),
            "idle_interval": max(1.0, float(self.cfg.coin_idle_interval)),
            "nn": bool(self.cfg.coin_nn),
        }
        debug_dir = None
        if self.cfg.coin_debug:
            debug_dir = self.cfg.coin_debug_dir or os.path.join(APP_DIR, "debug")
            try:
                os.makedirs(debug_dir, exist_ok=True)
            except OSError:
                debug_dir = None
        return CoinTask(screen, recognizer=Recognizer(), cfg=task_cfg, log=self.log,
                        should_stop=self._stop.is_set,
                        on_status=self._on_status, debug_dir=debug_dir)

    def _on_status(self, text):
        self.status = text

    def _loop(self):
        try:
            self._task.run()
        except Exception as exc:  # noqa: BLE001
            name = type(exc).__name__
            if name != "StopRequested":
                self.last_error = "%s: %s" % (name, exc)
                self.log("任务异常: %s" % self.last_error)
        finally:
            self.status = "已结束"
            self.next_at = 0.0

    def start(self):
        if self.running:
            return False
        try:
            from vision.screen import list_devices as adb_devices, pick_serial
        except ImportError as exc:
            self.log("缺少依赖: %s(先运行 tools\\setup_env.py 装好 numpy + opencv)" % exc)
            return False
        serials = adb_devices(self.cfg.adb_path or "adb")
        if not serials:
            self.log("未检测到已连接的设备,请确认 USB 调试或无线调试已开启。")
            return False
        if self.cfg.serial not in serials:
            self.cfg.serial = pick_serial(serials)
            self.log("自动选择设备: %s" % self.cfg.serial)
        try:
            self._task = self._make_task()
        except ImportError as exc:
            self.log("缺少依赖: %s(先运行 tools\\setup_env.py 装好 numpy + opencv)" % exc)
            return False
        if self.cfg.keep_awake and not self.dry_run:
            rc, _ = adb_run(self.cfg, ["shell", "svc", "power", "stayon", "true"], timeout=15)
            self._awake = rc == 0
        self._stop.clear()
        self.status = "启动中"
        self._thread = threading.Thread(target=self._loop, name="coins", daemon=True)
        self._thread.start()
        return True

    def stop(self, quiet=False):
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=6)
        self._thread = None
        self.next_at = 0.0
        if self._awake and not self.dry_run:
            self._awake = False
            adb_run(self.cfg, ["shell", "svc", "power", "stayon", "false"], timeout=10)
        if not quiet:
            self.log("金币任务已停止")


# ---------------------------------------------------------------- 命令行模式

def run_coin_cli(cfg):
    """命令行跑金币任务,日志同时打到控制台和 logs\\ 目录。"""
    run_log = RunLog("coins")

    def log(msg):
        print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)
        run_log.write(msg)

    runner = CoinRunner(cfg, log=log)
    log("设备: %s" % (cfg.serial or "自动选择"))
    log("模块: %s" % cfg.coin_modules)
    log("日志文件: %s" % run_log.path)
    log("按 Ctrl+C 停止")
    if not runner.start():
        run_log.close("启动失败")
        return 2
    try:
        while runner.running:
            time.sleep(0.3)
    except KeyboardInterrupt:
        log("收到 Ctrl+C,正在停止…")
    finally:
        runner.stop(quiet=True)
        summary = "结束:金币任务 %s" % (runner.status or "已停止")
        log(summary)
        run_log.close(summary)
    return 0

def run_cli(cfg, count, dry_run):
    core = SwiperCore(cfg, dry_run=dry_run)
    run_log = RunLog("swipe")

    def log(msg):
        print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)
        run_log.write(msg)

    core.log = log
    log("设备: %s" % (cfg.serial or "自动选择"))
    log("目标应用: %s" % (cfg.target_pkg or "保持当前界面(不限制)"))
    log("日志文件: %s" % run_log.path)
    if not core.ensure_device():
        return 2
    if not dry_run and ":" in cfg.serial:
        log("已连接无线设备 %s" % cfg.serial)
    try:
        core.probe_screen()
    except Exception as exc:  # noqa: BLE001
        log("读取屏幕信息失败: %s" % exc)
        return 3
    try:
        if dry_run:
            for _ in range(max(count, 1)):
                core.swipe_once()
                core.count += 1
            return 0
        core.start()
        while core.running and not core._stop.is_set():
            if count and core.count >= count:
                break
            time.sleep(0.2)
    except KeyboardInterrupt:
        log("收到 Ctrl+C,正在停止…")
    finally:
        core.stop()
        run_log.close("结束:共上滑 %d 次" % core.count)
    return 0


# ---------------------------------------------------------------- 图形界面

def run_gui(cfg):
    import tkinter as tk
    from tkinter import filedialog, messagebox, simpledialog, ttk

    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass

    root = tk.Tk()
    root.title(APP_TITLE)
    root.geometry("740x650")
    root.minsize(700, 560)
    try:
        root.option_add("*Font", ("Microsoft YaHei UI", 10))
    except Exception:
        pass

    log_queue = queue.Queue()
    run_log = RunLog("gui")

    def put(msg):
        """界面 + 日志文件双写。"""
        run_log.write(msg)
        log_queue.put(msg)

    core = SwiperCore(cfg, log=put)
    coin = CoinRunner(cfg, log=put)
    ui_state = {"running": False}   # 界面当前是否处于"运行中"外观

    style = ttk.Style()
    if os.name == "nt":
        for theme_name in ("vista", "winnative", "xpnative"):
            if theme_name in style.theme_names():
                style.theme_use(theme_name)
                break
    else:
        try:
            style.theme_use("clam")
        except Exception:
            pass
    style.configure("Big.TLabel", font=("Microsoft YaHei UI", 10))

    # ---- 设备行
    top = ttk.Frame(root, padding=(12, 10, 12, 4))
    top.pack(fill="x")
    ttk.Label(top, text="设备:").pack(side="left")
    device_var = tk.StringVar(value=cfg.serial)
    device_box = ttk.Combobox(top, textvariable=device_var, width=26, state="readonly")
    device_box.pack(side="left", padx=(6, 6))

    def refresh_devices(auto_start=False):
        devices = list_devices(cfg)
        device_box["values"] = devices
        if devices:
            if device_var.get() not in devices:
                device_var.set(devices[0])
            if auto_start:
                put("发现设备: %s" % ", ".join(devices))
        else:
            device_var.set("")
            if auto_start:
                put("未发现设备,请检查 USB 调试 / 无线调试。")

    ttk.Button(top, text="刷新", width=6,
               command=lambda: refresh_devices(True)).pack(side="left")
    status_dot = ttk.Label(top, text="", foreground="#888")
    status_dot.pack(side="left", padx=8)

    # ---- 目标应用行
    approw = ttk.Frame(root, padding=(12, 2, 12, 2))
    approw.pack(fill="x")
    ttk.Label(approw, text="目标应用:").pack(side="left")
    app_map = {name: pkg for name, pkg in APP_PRESETS}
    app_var = tk.StringVar()
    app_box = ttk.Combobox(approw, textvariable=app_var, width=17, state="readonly",
                           values=list(app_map.keys()))
    app_box.pack(side="left", padx=(6, 10))
    launch_var = tk.BooleanVar(value=cfg.launch_app and bool(cfg.target_pkg))
    guard_var = tk.BooleanVar(value=cfg.guard_app and bool(cfg.target_pkg))
    launch_chk = ttk.Checkbutton(approw, text="开始前切换到该应用", variable=launch_var)
    launch_chk.pack(side="left")
    guard_chk = ttk.Checkbutton(approw, text="仅在该应用前台时上滑", variable=guard_var)
    guard_chk.pack(side="left", padx=(16, 0))

    def app_name_for(pkg):
        for name, value in APP_PRESETS:
            if value and value == pkg:
                return name
        if pkg:
            label = "自定义 (%s)" % pkg
            app_map[label] = pkg
            app_box.configure(values=list(app_map.keys()))
            return label
        return "保持当前界面"

    def sync_app_state():
        has_target = bool(app_map.get(app_var.get(), ""))
        state = "normal" if has_target else "disabled"
        launch_chk.configure(state=state)
        guard_chk.configure(state=state)
        if not has_target:
            launch_var.set(False)
            guard_var.set(False)

    def on_app_change(_event=None):
        if app_var.get() == "自定义…":
            answer = simpledialog.askstring(
                APP_TITLE, "输入应用包名(可用 adb shell pm list packages 查看):", parent=root)
            if answer and answer.strip():
                app_var.set(app_name_for(answer.strip()))
            else:
                app_var.set(app_name_for(cfg.target_pkg))
        sync_app_state()
        try:
            refresh_mode_options()
        except NameError:      # 界面还没构造到模式行(启动早期)
            pass
        if core.running:
            put("目标应用已改为 %s(前台保护立即生效,切换动作在下次开始时执行)"
                          % (app_map.get(app_var.get()) or "不限制"))

    app_box.bind("<<ComboboxSelected>>", on_app_change)

    # ---- 参数行
    mid = ttk.Frame(root, padding=(12, 4, 18, 4))
    mid.pack(fill="x")
    min_var = tk.StringVar(value=("%.1f" % cfg.interval_min))
    max_var = tk.StringVar(value=("%.1f" % cfg.interval_max))
    duration_var = tk.StringVar(value=str(cfg.duration_ms))
    keep_awake_var = tk.BooleanVar(value=cfg.keep_awake)

    ttk.Label(mid, text="冷却时间(秒)").grid(row=0, column=0, sticky="w")
    ttk.Spinbox(mid, from_=0.5, to=600.0, increment=0.5, width=6,
                textvariable=min_var).grid(row=0, column=1, padx=(4, 2))
    ttk.Label(mid, text="~").grid(row=0, column=2)
    ttk.Spinbox(mid, from_=0.5, to=600.0, increment=0.5, width=6,
                textvariable=max_var).grid(row=0, column=3, padx=(2, 16))
    ttk.Label(mid, text="滑动时长(ms)").grid(row=0, column=4, sticky="w")
    ttk.Spinbox(mid, from_=80, to=1500, increment=20, width=7,
                textvariable=duration_var).grid(row=0, column=5, padx=(4, 0))

    opts = ttk.Frame(root, padding=(12, 2, 12, 6))
    opts.pack(fill="x")
    ttk.Checkbutton(opts, text="保持屏幕常亮", variable=keep_awake_var).pack(side="left")

    # ---- 模式行
    moderow = ttk.Frame(root, padding=(12, 0, 12, 4))
    moderow.pack(fill="x")
    mode_var = tk.StringVar(value="coins" if cfg.mode == "coins" else "swipe")
    ttk.Label(moderow, text="模式:").pack(side="left")
    ttk.Radiobutton(moderow, text="纯滑动(定时上滑)", value="swipe",
                    variable=mode_var).pack(side="left", padx=(6, 4))
    # 金币任务只对多多视频有意义(模板、按钮坐标、页面结构全是它家的),
    # 所以这一块整体放在一个子 Frame 里,目标应用不是拼多多时直接不显示。
    coin_opts = ttk.Frame(moderow)
    ttk.Radiobutton(coin_opts, text="多多视频金币任务(识别倒计时)", value="coins",
                    variable=mode_var).pack(side="left", padx=(0, 10))
    coin_debug_var = tk.BooleanVar(value=cfg.coin_debug)
    ttk.Checkbutton(coin_opts, text="保存识别截图",
                    variable=coin_debug_var).pack(side="left")
    mode_hint = ttk.Label(root, text="", foreground="#666", padding=(12, 0, 12, 6))
    mode_hint.pack(fill="x")

    def refresh_mode_hint(*_args):
        if mode_var.get() == "coins":
            mode_hint.configure(
                text="金币任务:自动进金币页 → 依次做 看商品视频 / 看直播 / 喝水打卡,"
                     "只在右上角有倒计时气泡时滑动;完成后等 %d 秒再返回。识别参数见 config.json。" % cfg.coin_done_delay)
        else:
            mode_hint.configure(text="纯滑动:按冷却时间定时上滑,不识别画面。")

    def refresh_mode_options(*_args):
        """金币任务选项只在目标应用是拼多多(多多视频)时出现。"""
        want = app_map.get(app_var.get(), "") == COIN_APP_PKG
        managed = bool(coin_opts.winfo_manager())   # "pack" / "" 比 ismapped 可靠
        if want and not managed:
            coin_opts.pack(side="left")
        elif not want and managed:
            coin_opts.pack_forget()
            if mode_var.get() == "coins":
                # 切到别的 App 就别偷偷留着金币模式,否则点开始会去点拼多多的界面
                mode_var.set("swipe")
                put("目标应用不是多多视频,已切回纯滑动模式")
        refresh_mode_hint()

    mode_var.trace_add("write", refresh_mode_hint)

    # ---- ADB 路径
    adbrow = ttk.Frame(root, padding=(12, 0, 12, 6))
    adbrow.pack(fill="x")
    ttk.Label(adbrow, text="adb:").pack(side="left")
    adb_var = tk.StringVar(value=cfg.adb_path)
    adb_entry = ttk.Entry(adbrow, textvariable=adb_var)
    adb_entry.pack(side="left", fill="x", expand=True, padx=(6, 6))

    def browse_adb():
        path = filedialog.askopenfilename(
            title="选择 adb.exe",
            filetypes=[("adb", "adb.exe"), ("可执行文件", "*.exe"), ("所有文件", "*.*")])
        if path:
            adb_var.set(path)

    ttk.Button(adbrow, text="浏览…", width=7, command=browse_adb).pack(side="left")

    # ---- 控制按钮
    ctrl = ttk.Frame(root, padding=(12, 4, 12, 6))
    ctrl.pack(fill="x")
    start_btn = tk.Button(ctrl, text="▶ 开始", font=("Microsoft YaHei UI", 12, "bold"),
                          bg="#12a150", fg="white", activebackground="#0e8a44",
                          activeforeground="white", relief="flat", bd=0,
                          padx=20, pady=5, cursor="hand2")
    start_btn.pack(side="left")
    once_btn = ttk.Button(ctrl, text="上滑一次", width=10)
    once_btn.pack(side="left", padx=(10, 0))
    counter_lbl = ttk.Label(ctrl, text="已上滑 0 次", style="Big.TLabel")
    counter_lbl.pack(side="left", padx=(16, 0))

    def open_log_dir():
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            os.startfile(LOG_DIR)          # noqa: S606 - 只在 Windows 上用
        except Exception as exc:  # noqa: BLE001
            put("打不开日志目录: %s" % exc)

    ttk.Button(ctrl, text="日志", width=6, command=open_log_dir).pack(side="left", padx=(12, 0))
    hint_lbl = ttk.Label(ctrl, text="(F9 开始/停止)", foreground="#888")
    hint_lbl.pack(side="right")

    # ---- 日志
    logbox = ttk.Frame(root, padding=(12, 0, 12, 12))
    logbox.pack(fill="both", expand=True)
    log_text = tk.Text(logbox, height=12, wrap="word", state="disabled",
                       background="#111", foreground="#d8ffd8",
                       insertbackground="#d8ffd8", relief="flat")
    scroll = ttk.Scrollbar(logbox, orient="vertical", command=log_text.yview)
    log_text.configure(yscrollcommand=scroll.set)
    log_text.pack(side="left", fill="both", expand=True)
    scroll.pack(side="right", fill="y")

    # ---- 读取界面参数
    def pull_settings():
        try:
            cfg.adb_path = adb_var.get().strip()
            cfg.serial = device_var.get().strip()
            cfg.interval_min = max(0.5, float(min_var.get()))
            cfg.interval_max = max(cfg.interval_min, float(max_var.get()))
            cfg.duration_ms = int(max(80, min(1500, float(duration_var.get()))))
            cfg.keep_awake = bool(keep_awake_var.get())
            cfg.mode = "coins" if mode_var.get() == "coins" else "swipe"
            cfg.coin_debug = bool(coin_debug_var.get())
            pkg = app_map.get(app_var.get(), "")
            cfg.target_pkg = "" if pkg == "__custom__" else pkg
            cfg.launch_app = bool(launch_var.get()) and bool(cfg.target_pkg)
            cfg.guard_app = bool(guard_var.get()) and bool(cfg.target_pkg)
            return True
        except ValueError:
            messagebox.showwarning(APP_TITLE, "冷却时间 / 滑动时长必须是数字。")
            return False

    def engine():
        """当前模式对应的执行器。"""
        return coin if mode_var.get() == "coins" else core

    def toggle():
        if engine().running:
            stop()
        else:
            start()

    def set_running_ui(running):
        ui_state["running"] = running
        if running:
            start_btn.configure(text="■ 停止", bg="#c0392b", activebackground="#a93226")
            status_dot.configure(text="● 运行中", foreground="#12a150")
        else:
            start_btn.configure(text="▶ 开始", bg="#12a150", activebackground="#0e8a44")
            status_dot.configure(text="● 已停止", foreground="#888")

    def push_live(*_args):
        """运行中修改间隔 / 抖动 / 时长,立即生效。"""
        try:
            new_min = max(0.5, float(min_var.get()))
            new_max = max(new_min, float(max_var.get()))
            new_duration = int(max(80, min(1500, float(duration_var.get()))))
        except ValueError:
            return
        changed = (new_min, new_max, new_duration) != (cfg.interval_min, cfg.interval_max, cfg.duration_ms)
        cfg.interval_min, cfg.interval_max, cfg.duration_ms = new_min, new_max, new_duration
        if changed and core.running:
            put("参数已更新:冷却 %.1f–%.1f 秒,滑动 %d ms"
                          % (new_min, new_max, new_duration))

    for _var in (min_var, max_var, duration_var):
        try:
            _var.trace_add("write", push_live)
        except Exception:
            pass

    def start():
        if not pull_settings():
            return
        if not cfg.serial:
            refresh_devices(True)
            if not device_var.get():
                messagebox.showwarning(APP_TITLE, "没有找到已连接的手机。\n请确认 USB 调试或无线调试已开启。")
                return
            cfg.serial = device_var.get()
        save_config(cfg)
        try:
            fg = core.current_package()
            if fg:
                put("当前前台应用: %s" % fg)
        except Exception:
            pass
        eng = engine()
        if eng.start():
            set_running_ui(True)
        else:
            messagebox.showwarning(APP_TITLE, "启动失败,请查看下方日志。")

    def stop():
        core.stop(quiet=True)
        coin.stop(quiet=True)
        put("已停止")
        set_running_ui(False)

    def swipe_now():
        if not pull_settings():
            return
        if not cfg.serial:
            refresh_devices(True)
            cfg.serial = device_var.get()
        if not cfg.serial:
            messagebox.showwarning(APP_TITLE, "没有找到已连接的手机。")
            return
        try:
            core.swipe_once(quiet=False)
        except Exception as exc:  # noqa: BLE001
            put("上滑失败: %s" % exc)

    start_btn.configure(command=toggle)
    once_btn.configure(command=swipe_now)
    root.bind("<F9>", lambda _e: toggle())

    def drain():
        while True:
            try:
                msg = log_queue.get_nowait()
            except queue.Empty:
                break
            log_text.configure(state="normal")
            log_text.insert("end", "[%s] %s\n" % (time.strftime("%H:%M:%S"), msg))
            log_text.see("end")
            log_text.configure(state="disabled")

        eng = engine()
        if mode_var.get() == "coins":
            counter_lbl.configure(text="金币任务 · %s" % (eng.status or "就绪"))
        else:
            counter_lbl.configure(text="已上滑 %d 次" % core.count)
        if eng.running and eng.next_at:
            remain = max(0.0, eng.next_at - time.monotonic())
            status_dot.configure(text="● 运行中 · %.1fs 后上滑" % remain, foreground="#12a150")
        elif eng.running:
            status_dot.configure(text="● 运行中", foreground="#12a150")
        elif ui_state["running"]:
            # 线程自己结束了(连续失败 / 出错),把按钮和状态灯复位,
            # 否则界面会一直停在"停止",下次还要先点一次停止才能再开始。
            set_running_ui(False)
        root.after(100, drain)

    def on_close():
        try:
            pull_settings()
            save_config(cfg)
        except Exception:
            pass
        core.stop(quiet=True)
        coin.stop(quiet=True)
        run_log.close("关闭窗口:上滑 %d 次" % core.count
                      if cfg.mode != "coins" else "关闭窗口:金币任务")
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    refresh_devices()
    app_var.set(app_name_for(cfg.target_pkg))
    sync_app_state()
    refresh_mode_options()
    drain()
    if cfg.serial:
        put("上次使用的设备: %s" % cfg.serial)
    if run_log.path:
        put("日志文件: %s" % run_log.path)
    put("就绪:冷却时间默认 3–6 秒随机,点“开始”或按 F9。")
    root.mainloop()


# ---------------------------------------------------------------- 入口

def main():
    parser = argparse.ArgumentParser(description="通过 ADB 模拟上滑,定时刷竖屏视频(任意 App)。")
    parser.add_argument("--cli", action="store_true", help="命令行模式(不打开窗口)")
    parser.add_argument("--interval", type=float, help="固定冷却时间(秒),等价于最小=最大")
    parser.add_argument("--min-interval", type=float, dest="interval_min",
                        help="冷却时间下限(秒),默认 3")
    parser.add_argument("--max-interval", type=float, dest="interval_max",
                        help="冷却时间上限(秒),默认 6")
    parser.add_argument("--count", type=int, default=0, help="上滑次数上限,0 表示一直滑")
    parser.add_argument("--duration", type=int, help="单次滑动时长(毫秒)")
    parser.add_argument("--serial", type=str, help="指定设备序列号")
    parser.add_argument("--app", type=str,
                        help="目标应用: douyin / pdd / bili / kuaishou / xhs / 包名;默认保持当前界面")
    parser.add_argument("--guard", action="store_true", help="仅当目标应用在前台时才上滑")
    parser.add_argument("--mode", choices=["swipe", "coins"],
                        help="swipe=定时上滑(默认)  coins=多多视频金币任务")
    parser.add_argument("--coins", action="store_true", help="等价于 --mode coins")
    parser.add_argument("--modules", type=str,
                        help="金币任务要做的模块,逗号分隔: shop_video,live,water")
    parser.add_argument("--done-delay", type=int, dest="done_delay",
                        help="倒计时完成后等几秒再返回(默认 10)")
    parser.add_argument("--monitor-interval", type=float, dest="monitor_interval",
                        help="倒计时中每隔几秒滑动一次(默认 3.5,必须小于 5 秒)")
    parser.add_argument("--idle-interval", type=float, dest="idle_interval",
                        help="任务跑完后挂机刷视频的滑动间隔(默认 3.5 秒)")
    parser.add_argument("--no-idle", action="store_true", dest="no_idle",
                        help="任务跑完后不挂机,直接结束")
    parser.add_argument("--no-nn", action="store_true", dest="no_nn",
                        help="不用训练出来的神经网络,只用模板匹配")
    parser.add_argument("--bench-nn", action="store_true", dest="bench_nn",
                        help="离线对比模板匹配和神经网络的识别准确率,然后退出")
    parser.add_argument("--max-minutes", type=int, dest="max_minutes",
                        help="金币任务最多跑几分钟(默认 60)")
    parser.add_argument("--coin-debug", action="store_true", dest="coin_debug",
                        help="把每次识别的画框图存到 debug 目录")
    parser.add_argument("--dry-run", action="store_true", help="只打印滑动坐标,不操作手机")
    args = parser.parse_args()

    cfg = load_config()
    if args.mode:
        cfg.mode = args.mode
    if args.coins:
        cfg.mode = "coins"
    if args.modules:
        cfg.coin_modules = args.modules
    if args.done_delay is not None:
        cfg.coin_done_delay = max(0, args.done_delay)
    if args.monitor_interval is not None:
        cfg.coin_monitor_interval = max(1.5, args.monitor_interval)
    if args.idle_interval is not None:
        cfg.coin_idle_interval = max(1.0, args.idle_interval)
    if args.no_idle:
        cfg.coin_idle = False
    if args.no_nn:
        cfg.coin_nn = False
    if args.bench_nn:
        os.execv(sys.executable, [sys.executable,
                                  os.path.join(APP_DIR, "tools", "bench_nn.py")])
    if args.max_minutes is not None:
        cfg.coin_max_minutes = max(1, args.max_minutes)
    if args.coin_debug:
        cfg.coin_debug = True
    if args.app is not None:
        key = args.app.strip().lower()
        cfg.target_pkg = APP_ALIASES.get(key, args.app.strip())
        cfg.launch_app = bool(cfg.target_pkg)
    if args.guard:
        cfg.guard_app = True
    if args.interval is not None:
        cfg.interval_min = cfg.interval_max = args.interval
    if args.interval_min is not None:
        cfg.interval_min = args.interval_min
    if args.interval_max is not None:
        cfg.interval_max = args.interval_max
    normalize_range(cfg)
    if args.duration is not None:
        cfg.duration_ms = max(80, args.duration)
    if args.serial:
        cfg.serial = args.serial
    cfg.adb_path = find_adb(cfg.adb_path)

    # 金币任务的模板和坐标全是拼多多的,别的 App 上跑只会瞎点。
    if cfg.mode == "coins" and cfg.target_pkg != COIN_APP_PKG:
        print("注意:金币任务只支持多多视频(%s),目标应用是 %s,已改为纯滑动模式。"
              % (COIN_APP_PKG, cfg.target_pkg or "不限制"), file=sys.stderr)
        cfg.mode = "swipe"

    if (args.cli or args.dry_run) and cfg.mode == "coins" and not args.dry_run:
        sys.exit(run_coin_cli(cfg))
    if args.cli or args.dry_run:
        sys.exit(run_cli(cfg, args.count, args.dry_run))

    run_gui(cfg)
    save_config(cfg)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        import traceback
        report_fatal(traceback.format_exc())
        raise