# -*- coding: utf-8 -*-
"""用真机截图"喂"状态机,在没有手机的情况下验证整条链路。

做法:写一个假 Screen,它的 capture() 按预设顺序返回 shoot 里的截图,
把点按/滑动/返回动作记录下来。coin_task 完全不知道对面是假设备,
所以这里跑通了,连真机时剩下的问题只会有:页面长得不一样、点偏了。

    python tools/simulate.py            # 三个场景全跑
    python tools/simulate.py -v         # 打印动作流水
"""

import argparse
import os
import random
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.coin_task import CoinTask          # noqa: E402
from vision.imgio import imread                # noqa: E402
from vision.recognizer import Recognizer       # noqa: E402

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHOOT = os.path.join(APP_DIR, "shoot")

FEED = "紫色部分圈出来的是需要识别的部分.png"
HOME = "拼多多首页_底部导航.jpg"
COIN = "点击金币按钮进入的二级界面 红色部分可以滑动.png"
COIN_ROWS = "二级界面滑动一下.jpg"
WATER = "周末喝水翻倍赚钱.jpg"
WATCH = "看商品视频得现金_.png"
DONE = "倒计时任务完成.png"
COIN_COOL = "二级金币_粉色冷却.png"
COIN_PKG = "com.xunmeng.pinduoduo"
COIN_NO_LIVE = "二级金币_当前.png"      # 这版列表里只有喝水行,没有看直播行


JITTER = 0.0          # 0.12 = 整帧缩放 ±12%、平移 ±(0.6*12)% 屏高(同一次运行里固定)
USE_NN = True
_JIT_RNG = random.Random(20261115)
_JIT_M = None          # 整次运行共用的变换矩阵,避免每帧随机导致结果没法比较


def _warp(img):
    """模拟换机/控件错位:整体缩放 + 平移,空出来的边补黑。尺寸不变。

    同一次运行里矩阵固定 —— 这样测的是"布局整体偏了"这种真实情况,
    而不是逐帧闪烁(后者连人眼都跟不上,不该拿来当回归指标)。
    """
    global _JIT_M
    h, w = img.shape[:2]
    if _JIT_M is None:
        scale = 1.0 + _JIT_RNG.uniform(-JITTER, JITTER)
        dx = _JIT_RNG.uniform(-0.6, 0.6) * JITTER * w
        dy = _JIT_RNG.uniform(-0.6, 0.6) * JITTER * h
        m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 0.0, scale)
        m[0, 2] += dx
        m[1, 2] += dy
        _JIT_M = m
    return cv2.warpAffine(img, _JIT_M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


class FakeScreen:
    """按脚本吐图,记录所有输入动作。"""

    def __init__(self, frames, repeat_last=True, verbose=False, pkg="", pkg_seq=None):
        self.frames = [(imread(os.path.join(SHOOT, f)) if isinstance(f, str) else f)
                       for f in frames]
        self.repeat_last = repeat_last
        self.verbose = verbose
        self.actions = []
        self._i = 0
        self._size = None
        self.pkg = pkg          # 模拟"当前前台应用";launch_app 之后变成被启动的那个
        self.pkg_seq = list(pkg_seq or [])   # 按次返回的前台包名(模拟用户切 App)
        self._pkg_i = 0

    def _next(self):
        img = self.frames[min(self._i, len(self.frames) - 1)]
        if self._i < len(self.frames) - 1 or not self.repeat_last:
            self._i += 1
        if JITTER > 0:
            img = _warp(img)
        return img

    def size(self):
        if self._size is None:
            img = self.frames[0]
            self._size = (img.shape[1], img.shape[0])
        return self._size

    def capture(self):
        return self._next()

    def capture_settled(self, tries=4, delay=0.25, diff_thr=1.5):
        return self._next()

    def _log(self, what):
        self.actions.append(what)
        if self.verbose:
            print("    · %s" % what)

    def tap(self, x, y):
        self._log("tap %d,%d" % (x, y))

    def swipe(self, x1, y1, x2, y2, ms=260):
        self._log("swipe (%d,%d)->(%d,%d)" % (x1, y1, x2, y2))

    def back(self):
        self._log("back")

    def key(self, code):
        self._log("key %d" % code)

    # -- 前台/启动(mirror vision.screen.Screen 的接口)------------------
    def foreground_pkg(self):
        if self.pkg_seq:
            pkg = self.pkg_seq[min(self._pkg_i, len(self.pkg_seq) - 1)]
            self._pkg_i += 1
            self.pkg = pkg
        return self.pkg

    def launch_app(self, pkg):
        self._log("launch %s" % pkg)
        self.pkg = pkg
        return True

    def shell(self, *args):
        self._log("shell %s" % " ".join(args))
        return ""


def make_task(screen, should_stop=None, **cfg):
    rec = Recognizer(use_nn=USE_NN)
    task = CoinTask(screen, recognizer=rec, cfg=cfg, log=print if cfg.pop("_log", False) else None,
                    should_stop=should_stop)
    task._sleep = lambda s: None       # 不等真时间
    return task


def scenario_feed_to_coin(verbose):
    """在主信息流上:应该点金币入口进金币页。"""
    screen = FakeScreen([FEED, COIN, COIN])
    task = make_task(screen, _log=verbose)
    ok = task.ensure_coin_page()
    return ok and any(a.startswith("tap") for a in screen.actions), screen.actions


def scenario_close_popup(verbose):
    """卡在喝水弹窗:应该按返回。"""
    screen = FakeScreen([WATER, COIN])
    task = make_task(screen, _log=verbose)
    ok = task.ensure_coin_page()
    return ok and screen.actions and screen.actions[0] == "back", screen.actions


def scenario_water(verbose):
    """喝水弹窗:点打卡 → 接着出现的倒计时也要盯(滑续命)→ 完成才返回。"""
    screen = FakeScreen([WATER, WATER, WATER, WATCH, WATCH, DONE])
    task = make_task(screen, _log=verbose, done_delay=0, monitor_interval=1)
    result = task.handle_water()
    swipes = [a for a in screen.actions if a.startswith("swipe")]
    ok = (result == "ok" and any(a.startswith("tap") for a in screen.actions)
          and len(swipes) >= 1 and screen.actions[-1] == "back")
    return ok, screen.actions


def scenario_countdown(verbose):
    """倒计时:看到 watch 就滑,看到 done 就等一会儿再返回。"""
    screen = FakeScreen([WATCH, WATCH, WATCH, DONE])
    task = make_task(screen, _log=verbose, done_delay=0, monitor_interval=1)
    result = task.handle_countdown("shop_video")
    swipes = [a for a in screen.actions if a.startswith("swipe")]
    ok = result == "ok" and len(swipes) >= 1 and screen.actions[-1] == "back"
    return ok, screen.actions


def scenario_find_row(verbose):
    """金币页:滚一下应该能找到三个模块行。"""
    screen = FakeScreen([COIN, COIN_ROWS, COIN_ROWS])
    task = make_task(screen, _log=verbose)
    hit = task.find_row("shop_video")
    return hit is not None, screen.actions


def scenario_video_tab(verbose):
    """启动 App 停在「首页」:应该点底部「多多视频」标签,而不是一路按返回退出 App。"""
    screen = FakeScreen([HOME, FEED, FEED])
    task = make_task(screen, _log=verbose)
    ok = task.ensure_video_tab()
    tapped_tab = any(a.startswith("tap") for a in screen.actions)
    no_back = "back" not in screen.actions[:1]
    return ok and tapped_tab and no_back, screen.actions


def scenario_pink_cooldown(verbose):
    """只有浅粉「再逛逛」、没有红字的冷却行:必须判冷却并跳过,不点进去。"""
    screen = FakeScreen([COIN_COOL, COIN_COOL])
    task = make_task(screen, _log=verbose)
    hit = task.find_row("shop_video")
    return hit == "cooldown", screen.actions


def scenario_launch_first(verbose):
    """还没拉起拼多多(前台是桌面):必须先启动 App,再点「多多视频」标签。

    真机上踩过:任务一起跑就去找标签,那会儿 App 还没起来,点到了别处。
    """
    screen = FakeScreen([HOME, FEED, FEED], pkg="com.android.launcher3")
    task = make_task(screen, _log=verbose, target_pkg=COIN_PKG)
    ok = task.ensure_video_tab()
    acts = screen.actions
    launch_idx = next((i for i, a in enumerate(acts) if a.startswith("launch")), -1)
    tap_idx = next((i for i, a in enumerate(acts) if a.startswith("tap")), -1)
    ordered = launch_idx >= 0 and tap_idx > launch_idx
    return ok and ordered, acts


def scenario_row_missing(verbose):
    """列表里根本没有「看直播赚现金」:要快速放弃,不能来回滚几分钟。

    真机 log(23:05:44→23:07:36)就是这样白滚了两分钟 —— 列表一屏放得下,
    上下都滚不动,却还在 6+6 个来回试。
    """
    screen = FakeScreen([COIN_NO_LIVE])
    task = make_task(screen, _log=verbose)
    hit = task.find_row("live", rerolls=0)
    swipes = [a for a in screen.actions if a.startswith("swipe")]
    return hit is None and len(swipes) <= 4, screen.actions


def scenario_idle_feed(verbose):
    """金币任务全跑完:要退回视频流挂机,每 3~4 秒滑一次,直到用户停。

    用户要的收尾行为(结束之后别闲着)。这里让 fake screen 在第 3 次滑动时
    抛停止,断言:先退回 feed(有 back)、然后连续滑了至少 3 次。
    """
    from vision.coin_task import StopRequested

    screen = FakeScreen([COIN, FEED, FEED, FEED, FEED, FEED])
    state = {"n": 0}

    def should_stop():
        state["n"] = len([a for a in screen.actions if a.startswith("swipe")])
        return state["n"] >= 3

    task = make_task(screen, should_stop=should_stop, idle_interval=0.1)
    try:
        task.idle_feed()
    except StopRequested:
        pass
    acts = screen.actions
    swipes = [a for a in acts if a.startswith("swipe")]
    ok = "back" in acts and len(swipes) >= 3
    return ok, acts


def scenario_idle_after_run(verbose):
    """run() 收尾也要进挂机(默认 idle_after=True),不能直接结束。"""
    from vision.coin_task import StopRequested

    screen = FakeScreen([COIN_COOL, FEED, FEED, FEED])

    def should_stop():
        return len([a for a in screen.actions if a.startswith("swipe")]) >= 2

    task = make_task(screen, should_stop=should_stop, modules=["shop_video"],
                     max_attempts=1, idle_interval=0.1)
    try:
        task.run()
    except StopRequested:
        pass
    swipes = [a for a in screen.actions if a.startswith("swipe")]
    return len(swipes) >= 2, screen.actions


def scenario_user_back_to_feed(verbose):
    """正在盯倒计时,用户自己按返回回到了视频流:不能再滑,整轮收工。

    这是"我手动跳出来了"的第一种样子 —— 还在拼多多里,但已经不在看视频了。
    """
    screen = FakeScreen([WATCH, WATCH, FEED])
    task = make_task(screen, _log=verbose, done_delay=0, monitor_interval=1)
    result = task.handle_countdown("shop_video")
    return result == "user_exit", screen.actions


def scenario_left_app_watch(verbose):
    """盯倒计时时用户切到微信:立刻抛 LeftTargetApp,一个动作都不做。

    第二种样子 —— 前台整个换掉了。这是最要紧的:不能往用户正在用的界面里滑。
    """
    from vision.coin_task import LeftTargetApp

    screen = FakeScreen([WATCH, WATCH, WATCH], pkg=COIN_PKG,
                        pkg_seq=[COIN_PKG, "com.tencent.mm", "com.tencent.mm"])
    task = make_task(screen, _log=verbose, target_pkg=COIN_PKG,
                     done_delay=0, monitor_interval=1)
    before = len(screen.actions)
    try:
        task.handle_countdown("shop_video")
    except LeftTargetApp:
        return True, screen.actions
    return False, screen.actions[before:]


def scenario_left_app_idle(verbose):
    """挂机时用户切走:run() 要把这个异常翻译成干净收工,而不是报错。"""
    screen = FakeScreen([FEED, FEED, FEED, FEED], pkg=COIN_PKG,
                        pkg_seq=[COIN_PKG, COIN_PKG, "com.tencent.mm", "com.tencent.mm"])
    task = make_task(screen, _log=verbose, target_pkg=COIN_PKG,
                     modules=[], max_attempts=1, idle_interval=0.1)
    result = task.run()
    ok = isinstance(result, dict) and result.get("_stopped") == "left_app"
    return ok, screen.actions


def scenario_idle_standby(verbose):
    """挂机时用户点进二级菜单:不能滑、不能按返回;回到视频流后自动接着挂。"""
    screen = FakeScreen([FEED, FEED, COIN_ROWS, COIN_ROWS, COIN_ROWS, FEED, FEED, FEED],
                        pkg=COIN_PKG)
    msgs = []
    task = CoinTask(screen, recognizer=Recognizer(use_nn=USE_NN),
                    cfg=dict(target_pkg=COIN_PKG, modules=[], max_attempts=1,
                             idle_interval=0.05),
                    log=msgs.append, should_stop=lambda: len(screen.actions) >= 4)
    task._sleep = lambda s: None
    task.run()

    swipes = [a for a in screen.actions if a.startswith("swipe")]
    backs = [a for a in screen.actions if a == "back"]
    waited = any("先不滑了" in m for m in msgs)
    resumed = any("继续挂机" in m for m in msgs)
    ok = waited and resumed and not backs and len(swipes) >= 3
    if verbose:
        print("    · 滑动 %d 次,按返回 %d 次,待机日志=%s 恢复日志=%s"
              % (len(swipes), len(backs), waited, resumed))
    return ok, screen.actions


SCENARIOS = [
    ("主信息流 → 点金币入口", scenario_feed_to_coin),
    ("首页 → 点底部「多多视频」标签", scenario_video_tab),
    ("没启动 App → 先拉起再点标签", scenario_launch_first),
    ("弹窗挡着 → 按返回", scenario_close_popup),
    ("喝水弹窗 → 点打卡再返回", scenario_water),
    ("倒计时 → 边滑边等 → 完成后返回", scenario_countdown),
    ("金币页 → 滚动找到商品视频行", scenario_find_row),
    ("行不在列表里 → 快速放弃", scenario_row_missing),
    ("浅粉按钮 → 判冷却并跳过", scenario_pink_cooldown),
    ("任务跑完 → 退回视频流挂机", scenario_idle_feed),
    ("run() 结束 → 自动进挂机", scenario_idle_after_run),
    ("看视频时退回视频流 → 停止本轮", scenario_user_back_to_feed),
    ("看视频时切到微信 → 立刻停手", scenario_left_app_watch),
    ("挂机时切走 → 干净收工", scenario_left_app_idle),
    ("挂机时点进二级菜单 → 待机不滑,回来接着挂", scenario_idle_standby),
]


def main():
    parser = argparse.ArgumentParser(description="离线模拟状态机")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--jitter", type=float, default=0.0,
                        help="把整帧缩放/平移这么多(小数,例如 0.12 = ±12%%),"
                             "用来验证换机、控件错位时流程还走不走得通")
    parser.add_argument("--no-nn", action="store_true", help="关掉神经网络,只用模板")
    parser.add_argument("--seed", type=int, default=20261115, help="扰动随机种子")
    args = parser.parse_args()
    global JITTER, USE_NN, _JIT_RNG, _JIT_M
    JITTER = max(0.0, args.jitter)
    USE_NN = not args.no_nn
    _JIT_RNG = random.Random(args.seed)
    _JIT_M = None
    if JITTER:
        print("扰动模式:整帧缩放/平移 ±%d%%(seed %d)\n" % (round(JITTER * 100), args.seed))
    failed = 0
    for title, fn in SCENARIOS:
        ok, actions = fn(args.verbose)
        print("[%s] %s" % ("OK  " if ok else "FAIL", title))
        if not ok:
            failed += 1
            print("       动作: %s" % actions)
    print("\n%d/%d 场景通过" % (len(SCENARIOS) - failed, len(SCENARIOS)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())