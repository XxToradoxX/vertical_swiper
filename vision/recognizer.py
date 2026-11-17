# -*- coding: utf-8 -*-
"""把「看画面」变成「当前是什么页面 / 有哪些模块 / 倒计时到了哪一步」。

设计原则(抄 MAA 的三条)
-----------------------
1. 搜索区域先验:每个模板都在一个归一化 ROI 里找,不全屏瞎扫。
   全屏搜"去领取"这种通用文字,一定会匹配到别的行;限定在右上角气泡区
   只找倒计时,就不会误判。
2. 阈值和缩放档进 manifest.json,调参不改代码。
3. 命中必须能画出框来:debug 时把框和分数画在画布上存盘,调不准直接看图。
"""

import os

import cv2
import numpy as np

from .canvas import Canvas
from .matcher import TemplateSet, _prep, match

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TPL_DIR = os.path.join(APP_DIR, "vision", "templates")

# ---- 归一化 ROI(x0, y0, x1, y1,相对 1080x1920 画布)-------------------
ROI = {
    # 金币页顶部「我的金币」。这块只框住标题本身那一小条 —— 开大了视频画面
    # 里的纯色/花纹会假阳性到 0.90(真机踩过),把商品视频页认成金币页。
    "coin_header": (0.02, 0.235, 0.27, 0.292),
    # 金币页的可滚动内容区(模块行都在这里)
    "content": (0.00, 0.330, 1.00, 1.000),
    # 视频页右上角的倒计时气泡。气泡的纵向位置随页面而变(商品页贴着状态栏,
    # 直播间偏下),所以这块 ROI 特意开得很大,靠字形匹配去定位,不靠位置。
    "bubble": (0.60, 0.080, 1.00, 0.500),
    # 主信息流右上角「金币」入口。
    # 上沿原来定在 0.150,结果 21:9 的手机(Xperia 1,1096x2560)翻车:那块屏幕
    # 更长,入口圆环被顶到画布 y≈225,整块落在 ROI 外面,信息流直接被判成"无法识别"。
    # 再往下踩一个坑(2026-11-16 真机):带「直播/关注/短剧/推荐」标签栏的那版信息流,
    # 圆环又比别的高一截 —— 设备 y≈150..280,画布 y≈112..210,连 0.100 都不够。
    # 干脆放到 0.060(设备 y≈154 起),两种排版都罩住;模板要 0.72 才算命中,
    # 往上放宽一点不会误判。
    "coin_entry": (0.60, 0.060, 1.00, 0.360),
    # 大转盘(「天天抽现金」)页面正中间那个红色「抽」按钮。轮盘页盖住整屏,
    # 这块 ROI 只在中上部,别的页面上不会出现同样的东西。
    "wheel_btn": (0.33, 0.230, 0.67, 0.390),
    # 抽完之后那张「恭喜抽中 …」结果弹窗里的「继续抽奖」大按钮
    "wheel_cta": (0.30, 0.590, 0.72, 0.700),
    # 喝水弹窗标题
    "water_title": (0.15, 0.235, 0.92, 0.345),
    # 喝水弹窗底部大按钮
    "water_btn": (0.10, 0.630, 0.92, 0.775),
}

# 模块行右侧按钮的固定横坐标(画布像素)。行标题命中后,点这个 x + 标题中心 y。
# 因为画布是定宽 1080,右对齐的按钮在任何手机上都在这个 x 附近。
ROW_TAP_X = 925

# 行标题底边超过这里就算"只露了一半",不去点它(容易点歪)
ROW_SAFE_BOTTOM = 0.955

# 一个模块可能有好几种标题写法 —— 拼多多会改文案(「看直播赚现金」后来变成了
# 「去直播间下单」,8.31 又把「周末喝水翻倍赚钱」改成了「每日8次喝水赚钱」)。
# 所以每个模块给一组候选模板,任意一个命中就算看到这一行。
# 带 _new 后缀的是拼多多 8.31 改版后的字形(字号/字距都变了,旧模板在那些
# 屏幕上只有 0.49~0.86,过不了 0.85 的阈值),两台手机的截图各标一套。
MODULES = {
    "wheel": ("row_wheel",),
    "shop_video": ("row_shop_video", "row_shop_video_new"),
    "water": ("row_water", "row_water_new"),
    "live": ("row_live", "row_live2", "row_live_new"),
}

# 页面级锚点也都有"新旧两版字形",按候选列表依次试,取分最高的那个
PAGE_HEADERS = ("page_coin", "page_coin_new")
COIN_ENTRIES = ("coin_entry", "coin_entry_new", "coin_entry_daily", "coin_entry_daily2")
WATER_TITLES = ("water_popup_title", "water_popup_title_new")
WATER_BTNS = ("water_checkin_btn", "water_checkin_btn_new")

# 模板名 -> 模块名(反向索引;判定"这行露没露全"时用)
ROW_TPLS = {tpl: mod for mod, tpls in MODULES.items() for tpl in tpls}

LABELS = {
    "coin_page": "金币二级界面",
    "water_popup": "喝水打卡弹窗",
    "wheel": "大转盘",
    "wheel_result": "大转盘抽奖结果",
    "task_watch": "倒计时进行中",
    "task_done": "倒计时已完成",
    "task_cooldown": "任务已完成,冷却中",
    "feed": "主信息流",
    "unknown": "无法识别",
}


class View:
    """一帧画面的识别结论。"""

    def __init__(self, page, hits=None, bubble=None, cooldowns=None, tones=None):
        self.page = page
        self.hits = hits or {}
        self.bubble = bubble        # "watch" / "done" / "cooldown" / None
        # 冷却中的模块名("water" 等),由 classify 数红字像素 + 看按钮颜色得出
        self.cooldowns = set(cooldowns or ())
        # 每个模块行右侧按钮的色调:{"shop_video": "hot"/"pink"/"none"}
        self.tones = dict(tones or {})

    def tone(self, name):
        """这一行右侧按钮的色调("hot" 能点 / "pink" 已领过 / "none")。"""
        return self.tones.get(name, "none")

    @property
    def label(self):
        if self.page == "task":
            return LABELS["task_watch"] if self.bubble == "watch" else (
                LABELS["task_cooldown"] if self.bubble == "cooldown" else LABELS["task_done"])
        return LABELS.get(self.page, self.page)

    def row(self, name):
        """取某模块行的命中框(没看到就是 None)。"""
        for tpl in MODULES.get(name, ()):
            hit = self.hits.get(tpl)
            if hit is not None:
                return hit
        return None

    def rows_visible(self):
        return [m for m in MODULES if self.row(m) is not None]

    def is_cooldown(self, name):
        """这一行是不是在冷却中。

        两条判据(真机上经常只有第二条):
        ①标题下面那行小字里的红色倒计时 ——「07:01后继续」「24:23后,打卡领取2000」
          「37:30后,打卡领取1500」,共同点是「MM:SS」红字,说明灰字;不认字,数红像素。
        ②右侧按钮洗成了浅粉色(「再逛逛」「去看看」),可用时是橙/红(「去完成」
          「去逛逛」「去领取」)。按钮文案会变,颜色不会,所以按饱和度判断。
        """
        return name in self.cooldowns

    def __repr__(self):
        parts = [self.label]
        if self.rows_visible():
            parts.append("行:" + ",".join(self.rows_visible()))
        if self.cooldowns:
            parts.append("冷却:" + ",".join(sorted(self.cooldowns)))
        if self.bubble:
            parts.append("气泡:" + self.bubble)
        return " ".join(parts)


def _has_red_note(hsv_img, hit, dx=420, dy=64, need=180):
    """标题正下方那一窄条里有没有红字(冷却倒计时)。

    真机上冷却那一行的小字长这样:「24:23后,打卡领取2000」——「24:23」是
    红色,后面全是灰色。可用状态下这一条没有红字,所以数红色像素就够,不用
    认字:数字每次都不一样,认字反而容易错。
    """
    h, w = hsv_img.shape[:2]
    y0 = min(h - 1, hit.bottom + 2)
    y1 = min(h, hit.bottom + dy)
    x0 = max(0, hit.x - 4)
    x1 = min(w, hit.x + dx)
    if y1 - y0 < 4 or x1 - x0 < 8:
        return False
    band = hsv_img[y0:y1, x0:x1]
    hue, sat, val = band[:, :, 0], band[:, :, 1], band[:, :, 2]
    red = ((hue <= 14) | (hue >= 166)) & (sat >= 110) & (val >= 100)
    return int(red.sum()) >= need


def _button_tone(hsv_img, hit, dx=100, y_up=25, y_down=55, min_frac=0.18, hot_s=150):
    """右侧按钮是"能点"的鲜色,还是"已经领过"的浅粉色。

    真机上同一排按钮只有两种长相(实测按钮填色的饱和度中位数):
        去完成(橙) 176 / 去邀请(红) 254 / 去查看(红) 254 / 明日再来(红) 214
        再逛逛(浅粉,冷却) 101 / 去看看(浅粉,冷却) ~100
    "浅粉"就是被洗白的那一档,阈值 150 正好落在两群中间。不认字、不做模板:
    按钮文案会变(去逛逛/去完成/再逛逛/去看看),颜色不会。

    返回 "hot"(能点) / "pink"(已领过) / "none"(这块啥都没有)。
    """
    h, w = hsv_img.shape[:2]
    x0 = max(0, ROW_TAP_X - dx)
    x1 = min(w, ROW_TAP_X + dx)
    y0 = max(0, hit.cy - y_up)
    y1 = min(h, hit.cy + y_down)
    if x1 - x0 < 20 or y1 - y0 < 20:
        return "none"
    band = hsv_img[y0:y1, x0:x1]
    sat, val = band[:, :, 1], band[:, :, 2]
    colored = (sat >= 60) & (val >= 150)
    if colored.mean() < min_frac:
        return "none"
    return "hot" if float(np.median(sat[colored])) >= hot_s else "pink"


class Recognizer:
    def __init__(self, tpl_dir=TPL_DIR, use_nn=False, nn_path=None):
        self.templates = TemplateSet(tpl_dir)
        self.tones_nn = None
        # 神经网络这条路(训练出来的检测器)。use_nn 打开且模型文件存在时优先用它,
        # 它认不出来的时候仍然可以退回模板匹配 —— 两条路各有所长:
        #   模板:零训练、可解释、对"字形完全一致"极稳,但位置/尺度一变就掉分;
        #   NN  :位置、缩放、底色都见过,框歪一点也能认出,但对从没见过的控件无能为力。
        self.use_nn = bool(use_nn)
        self.nn_path = nn_path
        self._net = None
        # 神经网络的分数低于这个值就不采信(留给模板匹配补)
        self.nn_min_score = 0.45

    # -- 神经网络 ------------------------------------------------------
    @property
    def net(self):
        if self._net is None:
            from .nn import CoinNet
            self._net = CoinNet(self.nn_path)
        return self._net

    def nn_available(self):
        try:
            return self.net.available()
        except Exception:
            return False

    def backend_label(self):
        """日志用:整帧识别这一路到底跑什么、跑在 CPU 还是 GPU。"""
        if not self.use_nn:
            return "模板匹配 CPU(cv2 %s)" % cv2.__version__
        try:
            nn = self.net.describe()
        except Exception:              # noqa: BLE001 - 模型坏了也要能继续跑模板
            nn = "神经网络不可用"
        return "%s + 模板匹配 CPU(cv2 %s)" % (nn, cv2.__version__)

    def _hits_from_nn(self, canvas_img):
        """跑一次检测器,把结果变成和模板匹配同构的 hits 字典。

        低于 `nn_min_score` 的结果**直接丢掉**:它们留在 hits 里会让模板匹配
        以为"这一类已经认过了"而不去补(实测踩过:金币页被 NN 以 0.3 分认出
        「我的金币」,模板那一路就被跳过了,结果页面判成 unknown)。
        """
        from .matcher import Hit
        hits = {}
        dets = self.net.detect(canvas_img)
        h, w = canvas_img.shape[:2]
        # 每类只留分数最高的一个(同一类在画面上出现多次时,认最像的那个;
        # 行标题可能出现两行同名 —— 但那属于同一模块,取高分的就够)。
        best = {}
        for name, score, x0, y0, x1, y1 in dets:
            if score < self.nn_min_score:
                continue
            if name not in best or score > best[name][1]:
                best[name] = (name, score, x0, y0, x1, y1)
        for name, (nm, score, x0, y0, x1, y1) in best.items():
            x0 = max(0, min(w - 2, int(x0)))
            y0 = max(0, min(h - 2, int(y0)))
            bw = max(2, min(w - x0, int(x1) - x0))
            bh = max(2, min(h - y0, int(y1) - y0))
            hits[nm] = Hit(nm, float(score), x0, y0, bw, bh)
        # 同一个位置被认出两个类(行标题最容易:同一行同时被打成 row_live 和
        # row_shop_video),按分数留最高的那个 —— 否则冷却集合里会多出一个模块。
        keep = {}
        for name in sorted(hits, key=lambda k: -hits[k].score):
            hit = hits[name]
            clash = [k for k, other in keep.items()
                     if abs(other.cx - hit.cx) < 40 and abs(other.cy - hit.cy) < 40]
            if not clash:
                keep[name] = hit
        return keep

    def classify_nn(self, canvas_img):
        """只用神经网络的判定(给工具做对比评估用)。"""
        if canvas_img is None or not self.nn_available():
            return View("unknown")
        return self._decide(self._hits_from_nn(canvas_img), canvas_img)

    def _nn_confident(self, hits, page_thr=0.70, elem_thr=0.50):
        """神经网络的结论够不够可信 —— 够就直接用,不再跑模板补。

        判据:N N 认出了"能定性这个页面"的主元素(head、water_title 或气泡),
        而且分数够高。只要有一点含糊,就老老实实退回模板再扫一遍 ——
        这一步省的是时间,不是准确率,所以宁可保守。
        """
        if "water_popup_title" in hits:
            # 弹窗这页要"标题 + 打卡按钮"都在才算认全:只认出标题就接管的话,
            # handle_water 会因为没有按钮而干等(实测 周末喝水翻倍赚钱.jpg 只给 0.77 标题)
            return "water_checkin_btn" in hits
        head = hits.get("page_coin")
        if head is not None and head.score >= page_thr:
            rows = [t for t in ROW_TPLS if t in hits and hits[t].score >= elem_thr]
            if rows:
                return True
        bubble = max((hits[t].score for t in
                      ("bubble_watch", "bubble_watch_new", "bubble_done",
                       "bubble_done_new", "bubble_cooldown",
                       "bubble_cooldown_new", "bubble_cooldown2")
                      if t in hits), default=0.0)
        if bubble >= page_thr:
            return True
        entry = max((hits[t].score for t in COIN_ENTRIES if t in hits),
                    default=0.0)
        if entry >= page_thr:
            return True
        return False

    def reload(self):
        self.templates = TemplateSet(self.templates.dir)

    # -- 主入口 --------------------------------------------------------
    def classify(self, canvas_img):
        """输入标准画布帧,返回 View。"""
        if canvas_img is None:
            return View("unknown")
        use_nn = self.use_nn and self.nn_available()
        h, w = canvas_img.shape[:2]
        hits = {}
        prepped = {}

        def find(name, roi_key):
            """按模板自己的 mode 预处理一次整帧,再在 ROI 里找。"""
            if name not in self.templates.items:
                return None
            mode = self.templates.mode(name)
            if mode not in prepped:
                prepped[mode] = _prep(canvas_img, mode)
            return match(prepped[mode], self.templates, name,
                         roi=self._roi(ROI[roi_key], w, h), mode=mode, prepped=True)

        def find_best(names, roi_key):
            """一组候选模板(同一控件的新旧字形)里取分最高的那个。"""
            best = None
            for nm in names:
                hit = hits.get(nm) if nm in nn_hits else find(nm, roi_key)
                if hit is not None and (best is None or hit.score > best.score):
                    best = hit
            return best

        # 先看神经网络有没有给出结论:它够确定就直接用(12 毫秒),否则**整帧**交给
        # 模板匹配(0.8 秒)。两边不混用:混用会让"按钮颜色"这类按坐标取样的
        # 判据找不到地方(踩过:NN 给的行框和模板的不一样,冷却就判不出来)。
        nn_hits = self._hits_from_nn(canvas_img) if use_nn else {}
        if nn_hits and self._nn_confident(nn_hits):
            return self._decide(dict(nn_hits), canvas_img)
        nn_hits = {}
        hits.update(nn_hits)

        def need(name):
            return name not in nn_hits

        bubble_hit = None
        bubble_state = None
        for cand, state in self.BUBBLE_STATES:
            hit = hits.get(cand) if cand in nn_hits else (find(cand, "bubble") if need(cand) else None)
            if hit is not None and (bubble_hit is None or hit.score > bubble_hit.score):
                bubble_hit, bubble_state = hit, state

        water_title = find_best(WATER_TITLES, "water_title")
        header = find_best(PAGE_HEADERS, "coin_header")
        entry = find_best(COIN_ENTRIES, "coin_entry")
        # 大转盘的两张锚点:中间的「抽」按钮,和结果弹窗里的「继续抽奖」
        wheel_spin = find_best(("wheel_spin",), "wheel_btn")
        wheel_cta = find_best(("wheel_continue",), "wheel_cta")

        rows = {}
        row_mods = {}          # 模板名 -> 模块名
        for mod, tpls in MODULES.items():
            best = None
            for tpl in tpls:
                hit = hits.get(tpl) if tpl in nn_hits else (
                    find(tpl, "content") if need(tpl) else None)
                if hit is not None and hit.bottom <= h * ROW_SAFE_BOTTOM:
                    # 同一模块挂着多个模板变体(旧 UI / 新 UI 各一套)。
                    # 取分数最高的那个,而不是顺序上第一个命中的:旧变体只要
                    # 勉强蹭到阈值就会把更准的新变体挡在门外(换分辨率时尤其
                    # 危险,0.855 的旧模板会先 break,1.000 的新模板根本没机会)。
                    if best is None or hit.score > best.score:
                        best = hit
            if best is not None:
                rows[best.name] = best
                row_mods[best.name] = mod

        if water_title is not None:
            hits["water_popup_title"] = water_title
            if "water_checkin_btn" not in hits:
                btn = find_best(WATER_BTNS, "water_btn")
                if btn is not None:
                    hits["water_checkin_btn"] = btn
        if header is not None:
            hits["page_coin"] = header
        if entry is not None:
            hits["coin_entry"] = entry
        if wheel_spin is not None:
            hits["wheel_spin"] = wheel_spin
        if wheel_cta is not None:
            hits["wheel_continue"] = wheel_cta
        if bubble_hit is not None:
            hits[bubble_hit.name] = bubble_hit
        hits.update(rows)
        return self._decide(hits, canvas_img)

    # -- 判定 ----------------------------------------------------------
    BUBBLE_STATES = (("bubble_watch", "watch"),
                     ("bubble_watch_new", "watch"),
                     ("bubble_done", "done"),
                     ("bubble_done_new", "done"),
                     ("bubble_cooldown", "cooldown"),
                     ("bubble_cooldown_new", "cooldown"),
                     ("bubble_cooldown2", "cooldown"))

    def _decide(self, hits, canvas_img):
        """hits(不管来自模板还是神经网络)-> View。判定规则只写一份。"""
        h, w = canvas_img.shape[:2]

        bubble_hit = None
        bubble_state = None
        for name, state in self.BUBBLE_STATES:
            hit = hits.get(name)
            if hit is not None and (bubble_hit is None or hit.score > bubble_hit.score):
                bubble_hit, bubble_state = hit, state

        water_title = hits.get("water_popup_title")
        header = hits.get("page_coin")
        entry = hits.get("coin_entry")

        rows = {}
        row_mods = {}          # 模板名 -> 模块名
        for mod, tpls in MODULES.items():
            best = None
            for tpl in tpls:
                hit = hits.get(tpl)
                if hit is not None and hit.bottom <= h * ROW_SAFE_BOTTOM:
                    if best is None or hit.score > best.score:
                        best = hit
            if best is not None:
                rows[best.name] = best
                row_mods[best.name] = mod
        # 露了一半的行标题不参与判定(点歪风险),也不留在 hits 里误导调试图
        for tpl in list(hits):
            if tpl in ROW_TPLS and tpl not in rows:
                hits.pop(tpl, None)

        # 冷却:标题正下方那条小字里的红色「MM:SS」。三种冷却(后继续 /
        # 后,打卡领取 N / 后,再领 N)的共同点就是时分秒是红字,所以只数红像素。
        cooldowns = set()
        tones = {}
        if rows:
            hsv = None
            for tpl, hit in rows.items():
                mod = row_mods[tpl]
                if hsv is None:
                    hsv = cv2.cvtColor(canvas_img, cv2.COLOR_BGR2HSV)
                tone = _button_tone(hsv, hit)
                tones[mod] = tone
                # 冷却有两种长相:①下方有红字「24:23后…」②右侧按钮洗成了浅粉色
                # (「再逛逛」「去看看」)。真机上经常只有②,所以两条都要看。
                if tone == "pink" or _has_red_note(hsv, hit):
                    cooldowns.add(mod)

        if water_title is not None:
            return View("water_popup", hits, bubble_hit and "watch")

        # 大转盘盖住整屏,优先级排在气泡/金币页前面:它出现时右上角那块、
        # 「我的金币」标题都会被自己的界面盖掉,拿别的锚点去判断只会误判。
        if hits.get("wheel_continue") is not None:
            return View("wheel_result", hits)
        if hits.get("wheel_spin") is not None:
            return View("wheel", hits)

        if bubble_hit is not None:
            # 倒计时气泡比"金币页标题"更可信:任务页一定有气泡,金币页一定没有。
            # 先判气泡,免得商品视频画面里飘过一段像"我的金币"的文字把任务页
            # 认成金币页(真机上就踩过这个坑,结果把还在计时的视频页给按出去了)。
            #
            # 但反过来也有假阳性:金币页滚到某些位置时,右上角那块会被误判成
            # 「任务完成」(真机 log 23:06:53 就是这样,结果把金币页按没了)。
            # 所以补一条:金币页标题 + 至少一个模块行同时成立时,不信 done 气泡。
            false_done = (bubble_state == "done" and header is not None
                          and header.score >= 0.90 and bool(rows))
            if not false_done:
                return View("task", hits, bubble_state)

        if header is not None and (rows or header.score >= 0.95):
            return View("coin_page", hits, None, cooldowns, tones)

        if entry is not None:
            return View("feed", hits, None)

        return View("unknown", hits, None)

    # -- 工具 ----------------------------------------------------------
    @staticmethod
    def _roi(hits_roi, w, h):
        x0, y0, x1, y1 = hits_roi
        return (int(x0 * w), int(y0 * h),
                max(2, int((x1 - x0) * w)), max(2, int((y1 - y0) * h)))

    def tap_point_for_row(self, hit, w):
        """模块行该点哪里:右侧按钮的横坐标 + 标题的垂直中心。"""
        return int(min(max(ROW_TAP_X, 0), w - 1)), hit.cy

    def draw(self, canvas_img, view, path=None):
        """把命中框画出来存盘(调参时最有用的一张图)。"""
        img = canvas_img.copy()
        for name, hit in view.hits.items():
            color = (0, 255, 0) if hit.score >= 0.8 else (0, 200, 255)
            cv2.rectangle(img, (hit.x, hit.y), (hit.right, hit.bottom), color, 2)
            cv2.putText(img, "%s %.2f" % (name, hit.score), (hit.x, max(12, hit.y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        cv2.putText(img, view.label, (16, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 255), 2, cv2.LINE_AA)
        if path:
            from .imgio import imwrite
            imwrite(path, img)
        return img