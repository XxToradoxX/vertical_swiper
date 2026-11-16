# -*- coding: utf-8 -*-
"""多多视频金币任务的状态机。

它不"看视频",它按下面这条链子走:
    进金币页 → 滚到模块行 → 点行右侧按钮 → 判断落在哪个界面
        ├ 喝水弹窗:点「喝水打卡领1500金币」→ 返回
        ├ 倒计时气泡(再看 MM:SS):**只在有气泡时才滑动**,滑到「任务完成」
        │                            → 等 done_delay 秒 → 返回
        └ 都不是:返回,记为失败
    → 回到金币页 → 下一个模块

关键约束来自用户:以前是一启动就自动滑,现在**没有倒计时就不滑**。
所以滑动这件事只发生在 MONITOR 状态里,别的地方一律不滑。
"""

import os
import random
import time

import cv2

from .canvas import Canvas
from .digits import DigitReader
from .recognizer import MODULES, Recognizer

DIGIT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "templates", "digits")

DEFAULTS = {
    # 整轮最多跑多少分钟
    "max_minutes": 60,
    # 每个模块最多尝试几次(失败一次就换下一个模块,下一轮再来)
    "max_attempts": 2,
    # 找模块行时最多滑几下
    "scroll_swipes": 4,
    # 点进去之后,等页面出来的时间(秒)
    "page_wait": 12,
    # 倒计时进行中,每隔几秒滑一下"续命"。真机实测页面约 5 秒无操作
    # 就暂停计时,所以这个值要小(3~4 秒);太大会一直等不完。
    "monitor_interval": 3.5,
    # 单个倒计时最多盯多久(秒)
    "monitor_timeout": 300,
    # 看到"任务完成"后等多久再返回(秒)——用户要求 10 秒
    "done_delay": 10,
    # 滑动时长(毫秒)
    "swipe_duration": 260,
    # 模块全跑完后,退回视频流挂机刷视频(而不是直接停)
    "idle_after": True,
    # 挂机时两次滑动之间的间隔(秒),随机在 [x-0.5, x+0.5] 之间取
    "idle_interval": 3.5,
    # 模块顺序
    "modules": ["shop_video", "live", "water"],
    # 识别优先走神经网络(模型文件 nn\coindet.onnx 存在时才生效),
    # 它没认出来的类别仍然用模板匹配补 —— 两条路互补
    "nn": True,
}

MODULE_NAMES = {
    "shop_video": "看商品视频得现金",
    "live": "看直播赚现金",
    # 这个名字只用来打日志。拼多多 8.31 把它改成了「每日8次喝水赚钱」,
    # 认行靠模板(新旧两套),日志上写个中性的名字,免得看日志时对不上屏幕。
    "water": "喝水赚钱",
}

# 底部导航栏「多多视频」标签图标的位置(占画布的比例,实测 1080×1920 下
# 约在 (350, 1845))。点「目标应用」启动拼多多只会停在「首页/推荐」,
# 还得点一下这个标签才进视频流。
VIDEO_TAB = (0.324, 0.961)


class StopRequested(Exception):
    pass


class LeftTargetApp(Exception):
    """用户手动切走了(前台不是拼多多了)—— 立刻停手,别抢手机。"""

    def __init__(self, pkg=""):
        super().__init__(pkg or "unknown")
        self.pkg = pkg


class CoinTask:
    def __init__(self, screen, recognizer=None, cfg=None, log=None,
                 should_stop=None, on_status=None, debug_dir=None):
        self.screen = screen
        self.cfg = dict(DEFAULTS)
        if cfg:
            self.cfg.update({k: v for k, v in cfg.items() if v is not None})
        # 识别走哪条路:cfg["nn"] 为真且模型文件在,就优先用神经网络(对位置偏移
        # 更宽容),认不出来的类再退回模板匹配;否则纯模板匹配。
        self.rec = recognizer or Recognizer(use_nn=bool(self.cfg.get("nn")))
        self.log = log or (lambda msg: None)
        self.should_stop = should_stop or (lambda: False)
        self.on_status = on_status or (lambda text: None)
        self.debug_dir = debug_dir
        self.canvas = None
        self._debug_seq = 0
        self._digits = None
        self._stray = 0
        self._standby_logged = False   # 挂机待机的日志只打一次,别刷屏

    # -- 基础动作 ------------------------------------------------------
    def _check_stop(self):
        if self.should_stop():
            raise StopRequested()

    def _sleep(self, seconds):
        """可被打断的等待。"""
        end = time.time() + seconds
        while time.time() < end:
            self._check_stop()
            time.sleep(0.2)

    def _ensure_canvas(self):
        if self.canvas is None:
            w, h = self.screen.size()
            self.canvas = Canvas(w, h)
            self.log("设备 %dx%d → 画布 %dx%d(%s)"
                     % (w, h, self.canvas.width, self.canvas.height, self.canvas.policy))
        return self.canvas

    def view(self, settled=False):
        """截一帧 → 归一化到画布 → 识别。"""
        canvas = self._ensure_canvas()
        raw = self.screen.capture_settled() if settled else self.screen.capture()
        if raw is None:
            return None
        view = self.rec.classify(canvas.normalize(raw))
        self._last_img = canvas.normalize(raw)
        if self.debug_dir:
            self._debug_seq += 1
            path = "%s\\%03d_%s.png" % (self.debug_dir, self._debug_seq, view.page)
            try:
                self.rec.draw(canvas.normalize(raw), view, path)
            except OSError:
                pass
        return view

    def tap_canvas(self, cx, cy):
        dev = self._ensure_canvas().to_device(cx, cy)
        self.log("  点 (%d,%d) → 设备 %s" % (cx, cy, dev))
        self.screen.tap(*dev)

    def read_remaining(self, view):
        """读「再看 MM:SS」里的剩余时间。读不出来就返回 (None, None)。

        纯几何 + 两张模板,不用 OCR 引擎;认不准的位会给 "?",这里就当没读到。
        """
        if view is None or getattr(self, "_last_img", None) is None:
            return None, None
        hits = view.hits or {}
        # 新版气泡("再看"两个字)和旧版是两张模板,读数字时哪张命中都认
        hit = hits.get("bubble_watch") or hits.get("bubble_watch_new")
        if hit is None:
            return None, None
        if self._digits is None:
            self._digits = DigitReader(DIGIT_DIR)
        try:
            text, secs = self._digits.read_bubble(self._last_img, hit)
        except (cv2.error, ValueError):
            return None, None
        if not text:
            return None, None
        shown = (text[:2] + ":" + text[2:]) if len(text) >= 4 else text
        return shown, secs

    def idle_feed(self):
        """模块全跑完后：退回主信息流挂机刷视频,每 3~4 秒滑一次,直到用户停。

        这是用户要求的收尾行为 —— 任务都处理完了也别闲着,视频流继续刷,
        顺手还能把「看视频得金币」的日常进度往前走。
        """
        base = float(self.cfg.get("idle_interval") or 3.5)
        self.log("=== 回到视频流挂机(每 %.1f 秒滑一次,Ctrl+C / 点停止结束)===" % base)
        try:
            self._idle_feed_loop(base)
        except StopRequested:
            self.log("挂机结束(收到停止信号)")

    def _idle_feed_loop(self, base):
        self._check_stop()
        # 先从金币页退回视频流:金币页/弹窗/任务页都按返回,落地就停
        for _ in range(5):
            self._check_stop()
            self._guard_app()
            view = self.view()
            if view is not None and view.page == "feed":
                break
            self.log("  当前在「%s」,按返回回视频流" % (view.page if view else "未知"))
            self.screen.back()
            self._sleep(1.2)
        # 万一返回过头退到了 App 外面,再切回「多多视频」标签
        view = self.view()
        if view is None or view.page != "feed":
            self.ensure_video_tab()

        count = 0
        while True:
            self._check_stop()
            self._guard_app()
            # 每一轮都看一眼界面:挂机时你可能自己点回二级菜单去了,
            # 这时候再滑就是在往你的菜单里甩手指(会滚列表、误触)。
            view = self.view(settled=False)
            page = view.page if view is not None else "unknown"
            if page in ("coin_page", "water_popup", "task"):
                # 你在用手机 → 原地待机,一次都不滑,等你回到视频流再接着挂
                self._idle_standby(page)
                continue
            count += 1
            self.on_status("挂机刷视频(第 %d 条)" % count)
            if page == "unknown":
                # 落到活动落地页/商品页上了:先给一次满屏上滑("上滑继续看视频"),
                # 再按返回(返回在上面的待机/上一轮循环里处理)
                self.log("挂机:这页认不出来,上滑继续看视频")
                self.fling_up()
            else:
                self.swipe_keepalive()
            pause = base + random.uniform(-0.5, 0.5)
            self.log("挂机:第 %d 次滑动,%.1f 秒后继续" % (count, pause))
            self._sleep(pause)

    def _idle_standby(self, page):
        """挂机时的待机:你已经点进了别的页面,就先不滑,等回到视频流。

        用户要求(2026-11-15):"此时我又点进了二级菜单的话,应该停止滑动
        进入待机状态"。这里绝不按返回 —— 你在用的界面不能被动。
        """
        if not self._standby_logged:
            self._standby_logged = True
            self.log("看到你点进了「%s」,先不滑了 —— 待机中,回到视频流会自动接着挂"
                     % ("金币二级界面" if page == "coin_page" else page))
        self.on_status("待机中(你在用手机)")
        while True:
            self._check_stop()
            self._guard_app()
            self._sleep(1.2)
            view = self.view(settled=False)
            page = view.page if view is not None else "unknown"
            if page == "feed":
                self._standby_logged = False
                self.log("回到视频流了,继续挂机")
                return "resume"

    def swipe_up(self, x_frac=0.5, y1_frac=0.80, y2_frac=0.62, duration=None):
        """金币页里往下滚内容(看更下面的行)。

        真机要求:列表滑动要"细腻" —— 一次只挪一两行,别一下甩到底,否则
        目标行一闪而过。落点注意:起点如果落在顶部那张「限时拼单」推荐卡上
        (画布 y≈0.42 附近),手势会被卡片吃掉,整个列表纹丝不动,所以起止
        点都落在普通行区域里。
        """
        canvas = self._ensure_canvas()
        w, h = canvas.dev_w, canvas.dev_h
        self.screen.swipe(int(w * x_frac), int(h * y1_frac),
                          int(w * x_frac), int(h * y2_frac),
                          duration or self.cfg["swipe_duration"])

    def swipe_down(self, x_frac=0.5, y1_frac=0.62, y2_frac=0.80, duration=None):
        """金币页里往上滚内容(看更上面的行),和 swipe_up 方向相反。"""
        canvas = self._ensure_canvas()
        w, h = canvas.dev_w, canvas.dev_h
        self.screen.swipe(int(w * x_frac), int(h * y1_frac),
                          int(w * x_frac), int(h * y2_frac),
                          duration or self.cfg["swipe_duration"])

    def fling_up(self, duration=380):
        """满屏一甩:从屏幕底部甩到顶部。

        活动落地页(「三单福利·首单减50元」这种)底部写着「上滑继续看视频」,
        金币页里那种 18% 的小滑动它根本不认 —— 页面纹丝不动,看着就像卡住了。
        用户要求(2026-11-16):"这个界面你需要执行上滑操作,而不是点击领取五十元"。
        """
        canvas = self._ensure_canvas()
        w, h = canvas.dev_w, canvas.dev_h
        self.screen.swipe(int(w * 0.5), int(h * 0.85),
                          int(w * 0.5), int(h * 0.15), duration)

    def swipe_keepalive(self):
        """倒计时页面上的"续命"动作:甩到下一个视频。

        真机教训(两个方向都试过):
        - 太小(6% 屏高)不行:页面收到触摸了,但**没有切到下一个视频**,
          计时照样停,盯着 40 多次都不动。
        - 太大/太快也不行:会被当成"退出/切到直播间"。
        折中:接近半屏、350ms 的中速上滑 —— 足够触发"下一条",又不至于翻走。
        """
        canvas = self._ensure_canvas()
        w, h = canvas.dev_w, canvas.dev_h
        self.screen.swipe(int(w * 0.5), int(h * 0.78),
                          int(w * 0.5), int(h * 0.20),
                          max(self.cfg["swipe_duration"], 350))

    # -- 页面归位 ------------------------------------------------------
    def _recover_unknown(self, why=""):
        """认不出来的页面上的兜底:先上滑,再返回。

        真机(User 2026-11-16)要求:碰到「三单福利」这类中间页,页面底下就写着
        「上滑继续看视频」—— 先上滑往往直接就进了真正的视频页;上滑还认不出来,
        再按返回退回上一层,别在那儿干等到超时。

        返回兜底之后的 View(可能仍然是 unknown / None)。
        """
        if why:
            self.log("  %s,先上滑试试" % why)
        # 用满屏一甩,不用金币页里那种小滑动:落地页只认完整的上滑手势
        self.fling_up()
        self._sleep(1.5)
        view = self.view(settled=False)
        if view is not None and view.page != "unknown":
            return view
        self.log("  还是认不出来,按返回")
        self.screen.back()
        self._sleep(1.2)
        return self.view(settled=False)

    def _in_target_app(self, pkg):
        """前台是不是目标 App。取不到前台信息时返回 True(宁可继续,不空转)。"""
        try:
            fg = self.screen.foreground_pkg()
        except Exception:              # noqa: BLE001
            return True
        if not fg:
            return True
        return fg == pkg

    def _guard_app(self):
        """用户手动切走就立刻停手。

        为什么必须做:倒计时那套逻辑会"每 3.5 秒滑一下",如果这时你在用手机
        (回微信、接电话),程序还在往你当前界面滑动,那就成了抢手机。
        所以每轮都确认一次前台:连续两次不是目标 App 就抛 LeftTargetApp,
        整个任务立刻收工(不按返回、不点任何东西)。
        """
        pkg = self.cfg.get("target_pkg") or ""
        if not pkg:
            return
        if self._in_target_app(pkg):
            self._stray = 0
            return
        self._stray += 1
        try:
            fg = self.screen.foreground_pkg() or "未知"
        except Exception:              # noqa: BLE001
            fg = "未知"
        if self._stray >= 2:
            raise LeftTargetApp(fg)
        self.log("  前台变成「%s」了,再确认一次(可能只是跳页)" % fg)

    def ensure_app(self, pkg, launch=True, timeout=25):
        """确保 pkg 真的在前台。

        这一步必须在找「多多视频」标签之前做:真机上踩过 —— 任务一起跑就去找
        标签,可那会儿拼多多还没被拉起来(甚至还在桌面),结果点到了别处。
        """
        if not pkg:
            return False
        fg = ""
        try:
            fg = self.screen.foreground_pkg()
        except Exception:              # noqa: BLE001 - 老机型 dumpsys 输出千奇百怪
            fg = ""
        if fg == pkg:
            self.log("目标应用已在前台(%s)" % pkg)
            return True
        if not launch:
            self.log("目标应用不是 %s(当前 %s),按设置不去切换它" % (pkg, fg or "未知"))
            return False
        self.log("当前前台:%s → 启动 %s" % (fg or "未知", pkg))
        try:
            self.screen.launch_app(pkg)
        except Exception:              # noqa: BLE001
            pass
        start = time.time()
        while time.time() - start < timeout:
            self._check_stop()
            self._sleep(1.0)
            try:
                fg = self.screen.foreground_pkg()
            except Exception:          # noqa: BLE001
                fg = ""
            if fg == pkg:
                self.log("已切到 %s,等界面加载" % pkg)
                self._sleep(2.0)
                return True
        self.log("等了 %d 秒 %s 还没到前台" % (timeout, pkg))
        return False

    def ensure_video_tab(self, tries=3):
        """确保停在「多多视频」。

        用 adb monkey 启动拼多多只会到「首页/推荐」,不会自动进视频流 —— 所以
        先点底部导航栏里的「多多视频」标签。要是当前被商品详情之类的页面盖住、
        标签栏点不到,就先退一步再试。
        """
        guard = self.cfg.get("target_pkg") or ""
        for i in range(tries):
            self._check_stop()
            view = self.view()
            if view is None:
                self._sleep(1.0)
                continue
            if view.page in ("feed", "coin_page", "water_popup", "task"):
                self.log("已在多多视频(%s)" % view.label)
                return True
            # 认不出这页时先确认前台是不是目标 App:不是就绝不乱点(桌面/其他
            # App 上点底部标签会戳到别的东西)
            if guard and not self._in_target_app(guard):
                self.log("当前不在 %s 里(认不出这个页面),先不去点标签" % guard)
                if not self.ensure_app(guard, timeout=12):
                    continue
            if i == 0:
                self.log("当前界面:%s,先切到多多视频" % view)
            canvas = self._ensure_canvas()
            self.tap_canvas(int(canvas.width * VIDEO_TAB[0]),
                            int(canvas.height * VIDEO_TAB[1]))
            self._sleep(1.8)
            view = self.view()
            if view is not None and view.page in ("feed", "coin_page", "task"):
                self.log("已进入多多视频(%s)" % view.label)
                return True
            # 标签栏被别的页面挡着(商品详情、搜索页……),退一步再来
            self.screen.back()
            self._sleep(1.2)
        self.log("切不到多多视频,先按现在这页继续试")
        return False

    def ensure_coin_page(self, tries=5):
        """把界面弄回金币二级页。返回 True 表示到了。

        导航阶段用 settled=False:直播间/视频页的画面一直在动,判"画面稳住"会
        一直等不到,白花十几秒。
        """
        for i in range(tries):
            self._check_stop()
            self._guard_app()
            view = self.view()
            if view is None:
                self.log("截不到图,重试")
                self._sleep(1.0)
                continue
            self.log("当前界面:%s" % view)
            if view.page == "coin_page":
                return True
            if view.page == "feed":
                entry = view.hits.get("coin_entry")
                if entry is not None:
                    self.log("主信息流 → 点金币入口进金币页")
                    self.tap_canvas(entry.cx, entry.cy)
            elif view.page == "unknown" and i == 0:
                # 认不出来的多半是「三单福利」这类活动落地页,页面底下写着
                # 「上滑继续看视频」—— 先满屏上滑一次,能回到视频流最好;
                # 还认不出来就下一轮按返回(用户要求:先上滑,再返回)。
                self.log("  看不懂这页,先上滑试试")
                self.fling_up()
            else:
                self.log("  按返回")
                self.screen.back()
            self._sleep(1.2)
        return False

    @staticmethod
    def _list_sig(view):
        """当前列表的"指纹":看得见哪些行 + 内容区画面。

        只看"认出哪些行"不够:滚到没有已知行的那一段时,连续两帧都认不出东西,
        会被误判成"滑不动了"。所以再叠一层内容区的粗粒度图像指纹。
        """
        return tuple(sorted(view.hits))

    def _content_sig(self):
        """内容区(画布 0.33~0.95 高)的粗指纹,用来判断滑动到底有没有生效。"""
        img = getattr(self, "_last_img", None)
        if img is None:
            return None
        try:
            import cv2
            h = img.shape[0]
            crop = img[int(h * 0.33):int(h * 0.95), :]
            small = cv2.resize(crop, (24, 32), interpolation=cv2.INTER_AREA)
            return small.tobytes()
        except Exception:
            return None

    def _sig(self, view):
        """(认出的行, 内容区画面) —— 判断"滑动到底动没动"用。"""
        return (self._list_sig(view), self._content_sig())

    def _reenter_coin_page(self):
        """退回信息流再点金币入口重进。

        真机关键发现:金币页的模块列表是**每次进入重新生成的** —— 同一时刻
        退出去再进来,行、顺序、奖励数额都可能不一样(见过「看商品视频得现金
        ￥24万/再逛逛」和「￥1000/去逛逛」两种)。所以要找的模块不在当前这份
        列表里时,重新进一次比反复滑动有效得多。
        """
        self.log("重新进一次金币页(列表会重新刷)")
        self.screen.back()
        self._sleep(1.5)
        return self.ensure_coin_page(tries=4)

    def _scan_rows(self, module, name):
        """在当前这份列表里扫一遍(先看屏,再上下来回滑几次)。"""
        tries = int(self.cfg["scroll_swipes"])
        # 整轮扫描的墙钟上限:一份列表看来看去也就那么几屏,真机上「这一行今天
        # 不在列表里」时最容易白滚两分钟 —— 所以宁可快点放弃。
        deadline = time.monotonic() + float(self.cfg.get("scan_seconds", 40))
        ended = 0
        for direction in ("往下", "往上"):
            last = None
            for i in range(tries):
                self._check_stop()
                self._guard_app()
                if time.monotonic() > deadline:
                    self.log("找「%s」超时,先不找了" % name)
                    return None
                view = self.view(settled=True)
                if view is None:
                    continue
                if view.page != "coin_page":
                    self.log("不在金币页了(%s),先归位" % view.page)
                    if not self.ensure_coin_page():
                        return "reenter"
                    continue
                hit = view.row(module)
                if hit is not None:
                    if view.is_cooldown(module):
                        self.log("「%s」还在冷却(按钮下方有「MM:SS后继续」),这次先跳过" % name)
                        return "cooldown"
                    self.log("找到「%s」score=%.2f" % (name, hit.score))
                    return hit
                sig = self._sig(view)
                if last is not None and sig == last:
                    # 滑了一下画面一模一样 ⇒ 这个方向到头了(列表一屏放得下时
                    # 第一次就会命中)。原来要连滑 3 次才能判出来,白等 10 秒。
                    self.log("这个方向滚不动了(到头)")
                    ended += 1
                    break
                last = sig
                self.log("没看到「%s」,%s滚 %d/%d" % (name, direction, i + 1, tries))
                if direction == "往下":
                    self.swipe_up()
                else:
                    self.swipe_down()
                self._sleep(0.8)
            else:
                continue
        if ended >= 2:
            self.log("这份列表上下都滚不动,「%s」应该不在里面" % name)
        return None

    def find_row(self, module, rerolls=1):
        """找到模块行:返回 Hit / "cooldown" / None。

        找不到就重新进金币页让列表重刷一次(re-roll),最多 `rerolls` 次。
        """
        if MODULES.get(module) is None:
            return None
        name = MODULE_NAMES.get(module, module)
        for attempt in range(int(rerolls) + 1):
            self._check_stop()
            got = self._scan_rows(module, name)
            if got == "cooldown":
                return "cooldown"
            if got == "reenter":
                if not self._reenter_coin_page():
                    return None
                continue
            if got is not None:
                return got
            if attempt < int(rerolls) and self._reenter_coin_page():
                continue
            break
        return None

    # -- 模块处理 ------------------------------------------------------
    def handle_water(self):
        """喝水:弹窗里点打卡按钮,然后返回。"""
        deadline = time.time() + self.cfg["page_wait"]
        while time.time() < deadline:
            self._check_stop()
            self._guard_app()
            # 直播间/视频页一直在动,判“画面稳住”只会白等十几秒 → 不等
            view = self.view(settled=False)
            if view is None:
                continue
            self.log("当前界面:%s" % view)
            if view.page == "water_popup":
                btn = view.hits.get("water_checkin_btn")
                if btn is None:
                    self.log("弹窗里没看到打卡按钮(可能今天已打完),关闭")
                    self.screen.back()
                    return "skip"
                self.tap_canvas(btn.cx, btn.cy)
                self.log("已点「喝水打卡」")
                self._sleep(2.5)
                # 打完卡不是"就完事了":真机上接着会出来一个「再看 MM:SS」的
                # 倒计时,得盯着看到点才能领。这里直接接进统一的盯气泡流程,
                # 不按返回 —— 按返回会把那个倒计时页面按掉。
                return self._watch_countdown("water")
            if view.page == "coin_page":
                self.log("还停在金币页(按钮没点开弹窗)")
                return "fail"
            self._sleep(1.0)
        self.log("等喝水弹窗超时")
        self.screen.back()
        return "fail"

    def handle_countdown(self, module):
        """视频/直播:盯气泡,有倒计时才滑,完成后再等 done_delay 秒返回。"""
        label = MODULE_NAMES.get(module, module)
        # 1) 等页面出来、气泡出现
        started = time.time()
        deadline = started + self.cfg["page_wait"]
        view = None
        recovered = False
        while time.time() < deadline:
            self._check_stop()
            self._guard_app()
            # 直播间/视频页一直在动,判“画面稳住”只会白等十几秒 → 不等
            view = self.view(settled=False)
            if view is None:
                continue
            self.log("当前界面:%s" % view)
            if view.page == "task":
                break
            if view.page == "coin_page":
                self.log("没有进入「%s」页面,可能今天做完了" % label)
                return "skip"
            if view.page == "unknown" and not recovered and time.time() > started + 3:
                # 点「去逛逛」之后先到的是中间页(如「三单福利」),气泡还没出来。
                # 这种页面上滑就能继续看视频,先上滑;不行再返回(用户要求)。
                recovered = True
                view = self._recover_unknown("这页认不出来") or view
                if view.page == "task":
                    break
                if view.page == "coin_page":
                    self.log("上滑/返回之后回到了金币页,这一行当成做过了")
                    return "skip"
                continue
            self._sleep(1.5)
        if view is None or view.page != "task":
            # 点进去一直没出现气泡 ⇒ 这一行今天已经领过了(真机上见过这种:
            # 按钮还是灰粉的「再逛逛」,连红字冷却提示都不给)。别死等,
            # 返回金币页换下一个模块。
            self.log("一直没等到「%s」的倒计时气泡,当成这轮已领过" % label)
            self.screen.back()
            self._sleep(1.2)
            return "unavailable"

        return self._watch_countdown(module)

    def _watch_countdown(self, module):
        """盯着「再看 MM:SS」气泡:每隔几秒滑一下续命,直到「任务完成」。"""
        label = MODULE_NAMES.get(module, module)
        # 盯到完成
        # 真机实测:页面大约 5 秒没有操作就会暂停计时,所以续命滑动得压到
        # 3~4 秒一次。排期按"墙钟时间"算 —— 一次截图本身要 1~2 秒,如果写成
        # sleep(间隔),真实间隔就变成 间隔+截图耗时,计时又会停。
        started = time.time()
        self._seen_task = False
        interval = max(1.5, float(self.cfg.get("monitor_interval") or 3.5))
        last_swipe = time.monotonic() - interval
        last_text = None
        last_secs = None
        while True:
            self._check_stop()
            self._guard_app()
            self.on_status("等待「%s」倒计时" % label)
            view = self.view(settled=False)
            if view is None:
                self._sleep(0.5)
                continue
            if view.page == "task":
                self._seen_task = True
            secs = None
            if view.page == "task" and view.bubble == "watch":
                # 把气泡上的剩余时间也写进日志(用户要求:能看出还剩多久)
                shown, secs = self.read_remaining(view)
                if shown and shown != last_text:
                    last_text = shown
                    if secs is None:
                        self.log("倒计时进行中:再看 %s(有一位没认准)" % shown)
                    else:
                        self.log("倒计时进行中:再看 %s(还剩 %d 秒)" % (shown, secs))
            # 续命滑动只在"计时真的停住了"的时候做 —— 数字还在往下减就说明
            # 页面自己在计时,这时候再滑反而有把视频翻走、倒计时从头开始的风险
            # (真机实测:每 3.5 秒滑一次的跑法,喝水那次 55 秒被翻成 116 秒重来)。
            counting = secs is not None and last_secs is not None and secs < last_secs
            if secs is not None:
                last_secs = secs
            if not counting and time.monotonic() - last_swipe >= interval:
                self.swipe_keepalive()
                last_swipe = time.monotonic()
            if view.page == "task" and view.bubble == "done":
                self.log("倒计时完成,等 %d 秒后返回" % self.cfg["done_delay"])
                self._sleep(self.cfg["done_delay"])
                self.screen.back()
                self._sleep(1.0)
                return "ok"
            if view.page == "task" and view.bubble == "cooldown":
                # 真机确认:右上角从「再看 MM:SS」变成「MM:SS 后领」,就是这个
                # 模块今天已经领到了(可能在我们盯之前就到了),再滑也没用,
                # 直接回金币页做下一个。
                self.log("气泡已变成「MM:SS 后领」(今天已领到),直接返回金币页")
                self._sleep(1.0)
                self.screen.back()
                self._sleep(1.0)
                return "ok"
            if time.time() - started > self.cfg["monitor_timeout"]:
                self.log("盯了 %d 秒还没完成,先撤" % self.cfg["monitor_timeout"])
                self.screen.back()
                return "timeout"
            if view.page == "coin_page":
                # 单帧判成金币页别急着退出:视频页偶发误判会让计时白等。
                # 隔一会儿再确认一次,连续两次才认。
                self._sleep(1.5)
                confirm = self.view(settled=False)
                if confirm is not None and confirm.page == "coin_page":
                    # 注意:这条路上从来不是"领到了"(领到会先看到 done / 后领气泡
                    # 走上面两个分支)。能走到这里说明视频页被顶掉了,钱多半没拿到,
                    # 所以如实记 dropped,别再报 ok 骗自己。
                    self.log("已经退回金币页,但没看到完成气泡 —— 这次大概率没拿到,记 dropped")
                    return "dropped"
                self.log("刚才那帧像是误判(%s),继续盯" % (confirm.page if confirm else "无图"))
            elif view.page == "feed":
                # 正在盯视频,却回到了主信息流 —— 这是"手动退出观看"的典型样子
                # (按了一次返回)。别再滑了,整轮收工。
                self.log("看着看着回到了视频流(你可能手动退出了观看),停止本轮任务")
                return "user_exit"
            elif view.page != "task":
                # 打卡之后弹窗会先关一下再跳去视频页,中间有几帧看不懂。
                # 给一段宽限期,过了宽限期还是没气泡才算失败。
                if time.time() < started + self.cfg["page_wait"]:
                    self.log("  界面还在切换(%s),再等等" % view.page)
                else:
                    if self._seen_task:
                        # 见过气泡之后掉到看不懂的界面:可能只是被活动弹窗盖住,
                        # 也可能你已经手动退出去逛别的了。先上滑、再按一次返回试试:
                        # 能回到任务页/金币页就继续盯;否则当作你走了。
                        self.log("界面变成 %s" % view.page)
                        probe = self._recover_unknown("界面看不懂")
                        if probe is not None and probe.page in ("task", "coin_page"):
                            continue
                        self.log("返回后也不在任务页了,当作你手动退出,停止本轮")
                        return "user_exit"
                    # 从头到尾没出现过倒计时气泡 ⇒ 这一行今天已经领过了/没在放量。
                    # 真机上就见过这种:按钮还是「再逛逛」但连红字提示都不给。
                    self.log("点进去一直没有倒计时气泡,当成这轮已领过")
                    self.screen.back()
                    self._sleep(1.2)
                    return "unavailable"
            # 补足到下一次该滑的时间点
            pause = last_swipe + interval - time.monotonic()
            self._sleep(min(max(pause, 0.2), interval))

    # -- 主流程 --------------------------------------------------------
    def run(self):
        """对外入口:把"用户切走了"翻译成一次干净的收工,而不是报错。"""
        try:
            return self._run_once()
        except LeftTargetApp as exc:
            self.log("前台已经是「%s」,不是拼多多了 —— 你可能去用手机了," % exc.pkg)
            self.log("立刻停手:不再滑动、不按返回、不点任何东西,程序到此停止")
            self.on_status("已停止(前台切到了 %s)" % exc.pkg)
            return {"_stopped": "left_app", "_pkg": exc.pkg}

    def _run_once(self):
        cfg = self.cfg
        self.log("=== 多多视频金币任务开始,超时 %d 分钟 ===" % cfg["max_minutes"])
        dead_line = time.time() + cfg["max_minutes"] * 60
        attempts = {m: 0 for m in cfg["modules"]}
        results = {}

        # 启动 App 后一般停在「首页」,先切到多多视频,否则下面一路按返回会把 App 退掉。
        # 顺序要紧:必须先确认目标 App 真在前台,再找「多多视频」标签 ——
        # 否则会在桌面/别的 App 上乱点。
        pkg = cfg.get("target_pkg") or ""
        if pkg:
            self.ensure_app(pkg, launch=bool(cfg.get("launch_app", True)))
        self.ensure_video_tab()

        while time.time() < dead_line:
            self._check_stop()
            self._guard_app()
            pending = [m for m in cfg["modules"] if attempts[m] < cfg["max_attempts"]]
            if not pending:
                break
            module = pending[0]
            attempts[module] += 1
            name = MODULE_NAMES.get(module, module)
            self.log("--- 处理「%s」(第 %d 次)---" % (name, attempts[module]))
            self.on_status("处理「%s」" % name)

            if not self.ensure_coin_page():
                self.log("回不到金币页,收工")
                break
            hit = self.find_row(module)
            if hit == "cooldown":
                results[module] = "cooldown"
                attempts[module] = cfg["max_attempts"]
                continue
            if hit is None:
                self.log("找不到「%s」(这轮列表里没有这一行)" % name)
                results[module] = "not_found"
                # 列表是每次进页面重新生成的,但同一次运行里再扫一遍多半还是
                # 没有 —— 别把 6 分钟耗在来回滚屏上,直接判这个模块本轮缺货。
                attempts[module] = cfg["max_attempts"]
                continue

            self.tap_canvas(*self.rec.tap_point_for_row(hit, self._ensure_canvas().width))
            self._sleep(2.0)
            # 真机上点空的场景:行太靠下/被底部卡片压住时会停在原页。
            # 再点一次(往下压一点),还不行就当这行点不动。
            for retry in range(2):
                check = self.view()
                if check is None or check.page != "coin_page":
                    break
                self.log("好像没点开(还在金币页),再点一次")
                self.tap_canvas(*self.rec.tap_point_for_row(hit, self._ensure_canvas().width))
                self._sleep(1.5)

            if module == "water":
                outcome = self.handle_water()
            else:
                outcome = self.handle_countdown(module)
            results[module] = outcome
            self.log("「%s」结果:%s" % (name, outcome))

            # 收尾:确保回到金币页,然后等一会儿再动下一个
            self._sleep(1.0)
            self.ensure_coin_page()
            if outcome in ("ok", "skip", "unavailable"):
                # 这轮做完了 / 今天已经领过 —— 都不再重试
                attempts[module] = cfg["max_attempts"]
            elif outcome == "user_exit":
                # 用户自己退出了观看:别再自己往金币页钻,整轮收工(也不挂机)
                self.log("检测到手动退出,提前收工,不再自动滑动")
                return results
            else:
                self._sleep(3.0)

        self.log("=== 结束,结果: %s ===" % (results or "无"))
        if cfg.get("idle_after", True):
            self.idle_feed()
        return results