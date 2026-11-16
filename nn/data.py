# -*- coding: utf-8 -*-
"""训练数据:把 shoot 里的真机截图变成"检测器"要的标注样本。

思路(为什么这么干):
- 只有 25 张真机图,靠人工标框不现实,但 **我们自己知道模板是从哪张图的哪个框裁出来的**
  (tools/template_spec.json),这就是精确标注;
- 同一张图里没被裁过的元素,用现有模板匹配去认(阈值 0.92,只收高置信),
  相当于让"上一位老师"给新学生标一遍;
- 25 张图不够泛化,所以训练时在线做**几何 + 色彩 + 模糊 + 遮挡**增广:
  平移/缩放/旋转直接模拟"控件位置有偏差、换手机分辨率不同",
  色彩抖动模拟不同主题和视频背景,遮挡模拟活动弹窗。

坐标系约定:所有标注都在 **1080x1920 标准画布**坐标里;喂给网络前再整体缩放到
INPUT_W x INPUT_H,标注乘同一个比例。
"""

import json
import os
import random

import cv2
import numpy as np

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHOOT = os.path.join(APP_DIR, "shoot")
SPEC = os.path.join(APP_DIR, "tools", "template_spec.json")

CANVAS_W, CANVAS_H = 1080, 1920
INPUT_W, INPUT_H = 448, 800          # 必须能被 32 整除
STRIDE = 8
GRID_W, GRID_H = INPUT_W // STRIDE, INPUT_H // STRIDE   # 56 x 100

# 模板名 -> 类别号。同一类有多个变体的(直播间两种文案、气泡两种「后领」)合并成一类,
# 因为对状态机来说它们是同一个东西。
CLASSES = [
    "row_shop_video",
    "row_water",
    "row_live",
    "coin_entry",
    "bubble_watch",
    "bubble_done",
    "bubble_cooldown",
    "water_checkin_btn",
    "page_coin",
    "water_popup_title",
]
TEMPLATE_TO_CLASS = {
    "row_shop_video": "row_shop_video",
    "row_water": "row_water",
    "row_live": "row_live",
    "row_live2": "row_live",
    "coin_entry": "coin_entry",
    "bubble_watch": "bubble_watch",
    "bubble_done": "bubble_done",
    "bubble_cooldown": "bubble_cooldown",
    "bubble_cooldown2": "bubble_cooldown",
    "water_checkin_btn": "water_checkin_btn",
    "page_coin": "page_coin",
    "water_popup_title": "water_popup_title",
}
NUM_CLASSES = len(CLASSES)

# 验证集(留出,不参与训练):挑的是"样子最不一样"的几张 —— 冷色直播间、
# 粉色冷却页、首页导航。
VAL_FILES = {
    "看直播赚现金.jpg",
    "二级金币_粉色冷却.png",
    "拼多多首页_底部导航.jpg",
    "直播界面 右上角提示任务完成 此时说明该板块获取金币已经进入冷却.png",
    "二级金币_当前2.png",
}


def load_spec():
    with open(SPEC, encoding="utf-8") as f:
        return json.load(f)


def load_canvas(path):
    """读图并归一化到 1080x1920 画布(stretch,和线上一致)。"""
    from vision.imgio import imread
    img = imread(path)
    if img is None:
        raise IOError("读不了 %s" % path)
    return cv2.resize(img, (CANVAS_W, CANVAS_H), interpolation=cv2.INTER_AREA)


def spec_boxes(spec, fname, img):
    """这张图上已知的精确标注:模板的 source 就是它自己的框。"""
    out = []
    for name, t in spec["templates"].items():
        if t.get("source") != fname:
            continue
        cls = TEMPLATE_TO_CLASS.get(name)
        if cls is None:
            continue
        x0, y0, x1, y1 = t["box"]
        out.append((CLASSES.index(cls), float(x0), float(y0), float(x1), float(y1), 1.0))
    return out


def auto_boxes(spec, fname, img, threshold=0.92):
    """其它元素:用现有模板匹配补标(只收高置信度,宁缺勿滥)。"""
    from vision.matcher import TemplateSet, match
    tpl_dir = os.path.join(APP_DIR, "vision", "templates")
    ts = TemplateSet(tpl_dir)
    out = []
    for name in spec["templates"]:
        cls = TEMPLATE_TO_CLASS.get(name)
        if cls is None:
            continue
        if spec["templates"][name].get("source") == fname:
            continue
        try:
            hit = match(img, ts, name, threshold=threshold)
        except Exception:                      # noqa: BLE001
            hit = None
        if hit is None:
            continue
        out.append((CLASSES.index(cls), float(hit.x), float(hit.y),
                    float(hit.right), float(hit.bottom), float(hit.score)))
    return out


def collect_samples(spec=None, verbose=False, refresh=False):
    """返回 [(fname, canvas_img, boxes)]。

    模板匹配标一遍 25 张图要 60 多秒,所以标注缓存在 nn/labels.json;
    改了 template_spec.json 之后加 --refresh 重标。
    """
    spec = spec or load_spec()
    cache_path = os.path.join(APP_DIR, "nn", "labels.json")
    cache = {}
    if not refresh and os.path.exists(cache_path):
        try:
            with open(cache_path, encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:                      # noqa: BLE001
            cache = {}
    files = sorted(f for f in os.listdir(SHOOT)
                   if f.lower().endswith((".png", ".jpg", ".jpeg")))
    out = []
    dirty = False
    for f in files:
        img = load_canvas(os.path.join(SHOOT, f))
        boxes = cache.get(f)
        if boxes is None:
            boxes = spec_boxes(spec, f, img) + auto_boxes(spec, f, img)
            cache[f] = boxes
            dirty = True
        if verbose:
            print("  %-40s %d 个标注" % (f, len(boxes)))
        out.append((f, img, [tuple(b) for b in boxes]))
    if dirty:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as fh:
            json.dump(cache, fh, ensure_ascii=False, indent=1)
    return out


# ---------------------------------------------------------------- 增广
def _warp(img, boxes, angle, scale, tx, ty):
    """绕画布中心做相似变换,标注跟着走。"""
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
    m[0, 2] += tx
    m[1, 2] += ty
    warped = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT_101)
    new = []
    for cls, x0, y0, x1, y1, _s in boxes:
        pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
        pts = cv2.transform(pts.reshape(1, -1, 2), m).reshape(-1, 2)
        nx0, ny0 = pts.min(axis=0)
        nx1, ny1 = pts.max(axis=0)
        # 变换后跑出画布太多的框直接丢掉(只留一小块没有训练价值)
        if nx1 < 4 or ny1 < 4 or nx0 > w - 4 or ny0 > h - 4:
            continue
        new.append((cls, float(max(0, nx0)), float(max(0, ny0)),
                    float(min(w - 1, nx1)), float(min(h - 1, ny1)), 1.0))
    return warped, new


def _color(img, rng):
    img = img.astype(np.float32)
    # 亮度 / 对比度
    alpha = rng.uniform(0.75, 1.3)
    beta = rng.uniform(-28, 28)
    img = img * alpha + beta
    # 饱和度(在 HSV 上调 S,模拟不同色主题)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.6:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * rng.uniform(0.6, 1.4), 0, 255)
        hsv[:, :, 0] = (hsv[:, :, 0] + rng.uniform(-8, 8)) % 180
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return img


def _degrade(img, rng):
    """视频页会动、会糊、会二次压缩,训练时就把这些劣化造出来。"""
    if rng.random() < 0.35:
        k = rng.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    if rng.random() < 0.30:
        # 先缩小再放大 = 模拟截图被压缩过
        f = rng.uniform(0.45, 0.85)
        h, w = img.shape[:2]
        small = cv2.resize(img, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.25:
        noise = rng.normal(0, rng.uniform(3, 10), img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


def _occlude(img, boxes, rng):
    """贴几块"弹窗/广告"上去;被盖住的标注按覆盖比例丢掉。"""
    h, w = img.shape[:2]
    keep = list(boxes)
    for _ in range(int(rng.integers(0, 3))):
        ow = int(rng.integers(w // 8, w // 2))
        oh = int(rng.integers(h // 14, h // 3))
        ox = int(rng.integers(0, w - ow))
        oy = int(rng.integers(0, h - oh))
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        if rng.random() < 0.5:
            img = img.copy()
            cv2.rectangle(img, (ox, oy), (ox + ow, oy + oh), color, -1)
            if rng.random() < 0.5:
                cv2.rectangle(img, (ox + 6, oy + 6), (ox + ow - 6, oy + oh - 6),
                              (255, 255, 255), 3)
        keep = [b for b in keep if _overlap_ratio(b, ox, oy, ox + ow, oy + oh) < 0.45]
    return img, keep


def _overlap_ratio(box, x0, y0, x1, y1):
    _, bx0, by0, bx1, by1, _ = box
    ix = max(0.0, min(bx1, x1) - max(bx0, x0))
    iy = max(0.0, min(by1, y1) - max(by0, y0))
    area = max(1e-6, (bx1 - bx0) * (by1 - by0))
    return ix * iy / area


def augment(img, boxes, rng):
    angle = rng.uniform(-2.5, 2.5)
    scale = rng.uniform(0.82, 1.20)
    tx = rng.uniform(-70, 70)
    ty = rng.uniform(-90, 90)
    img, boxes = _warp(img, boxes, angle, scale, tx, ty)
    img = _color(img, rng)
    img = _degrade(img, rng)
    img, boxes = _occlude(img, boxes, rng)
    return img, boxes


# ---------------------------------------------------------------- 编码
def _splat(hm, gx, gy, radius):
    """在热力图上以 (gx,gy) 为中心打一个高斯斑(CenterNet 的做法)。

    一开始用"中心格=1、周围=0"的硬标签训练,结果损失降不下去、假阳性还多:
    控件中心稍微偏一格,这个格子就被当成了负样本,模型学成"抖动的簇"。
    改成高斯之后,中心附近的格子目标值在 0~1 之间,不会被硬压成负样本,
    收敛明显更稳。
    """
    r = int(max(1, round(radius)))
    sigma = max(0.7, r / 3.0)
    for dy in range(-r, r + 1):
        y = gy + dy
        if not (0 <= y < GRID_H):
            continue
        for dx in range(-r, r + 1):
            x = gx + dx
            if not (0 <= x < GRID_W):
                continue
            v = np.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))
            if v > hm[0, y, x]:
                hm[0, y, x] = v


def encode(boxes):
    """CenterNet 式编码:中心点所在格子为正样本。

    返回 (hm, cls, wh, off, mask),形状分别是 (1,GH,GW) (C,GH,GW) (2,GH,GW)
    (2,GH,GW) (GH,GW)。
    """
    hm = np.zeros((1, GRID_H, GRID_W), np.float32)
    cls = np.zeros((NUM_CLASSES, GRID_H, GRID_W), np.float32)
    wh = np.zeros((2, GRID_H, GRID_W), np.float32)
    off = np.zeros((2, GRID_H, GRID_W), np.float32)
    mask = np.zeros((GRID_H, GRID_W), np.float32)
    sx = INPUT_W / float(CANVAS_W)
    sy = INPUT_H / float(CANVAS_H)
    for c, x0, y0, x1, y1, _s in boxes:
        px0, px1 = x0 * sx, x1 * sx
        py0, py1 = y0 * sy, y1 * sy
        cx, cy = (px0 + px1) / 2.0, (py0 + py1) / 2.0
        bw, bh = max(1.0, (px1 - px0)), max(1.0, (py1 - py0))
        gx, gy = int(cx / STRIDE), int(cy / STRIDE)
        if not (0 <= gx < GRID_W and 0 <= gy < GRID_H):
            continue
        hm[0, gy, gx] = 1.0
        cls[c, gy, gx] = 1.0
        wh[0, gy, gx] = bw / STRIDE
        wh[1, gy, gx] = bh / STRIDE
        off[0, gy, gx] = cx / STRIDE - gx
        off[1, gy, gx] = cy / STRIDE - gy
        mask[gy, gx] = 1.0
        # 中心点周围打高斯斑:半径按控件短边算,行标题这种细长条用高度定半径
        _splat(hm, gx, gy, min(bw, bh) / STRIDE * 0.45)
    return hm, cls, wh, off, mask


def preprocess(img):
    """画布图 -> 网络输入(float32, BGR->RGB, /255)。"""
    small = cv2.resize(img, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)
    small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    # 增广里的乘除(亮度/对比度抖动)偶尔会溢出成 nan/inf,喂进网络就是 loss nan。
    # 这里兜一下:nan 归 0,其余压回 [0,1]。
    return np.clip(np.nan_to_num(small, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def decode(hm, cls, wh, off, thr=0.35, topk=40):
    """网络输出 -> [(cls, score, canvas_x0..y1)]。"""
    hm = 1.0 / (1.0 + np.exp(-hm[0]))                 # (GH,GW)
    cls_prob = 1.0 / (1.0 + np.exp(-cls))             # 类别头也是 logit,要过 sigmoid
    flat = hm.reshape(-1)
    idx = np.argsort(-flat)[:topk]
    sx = CANVAS_W / float(INPUT_W)
    sy = CANVAS_H / float(INPUT_H)
    out = []
    for i in idx:
        score = float(flat[i])
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
        x0 = (cx - bw / 2) * sx
        y0 = (cy - bh / 2) * sy
        x1 = (cx + bw / 2) * sx
        y1 = (cy + bh / 2) * sy
        out.append((c, score * cscore, x0, y0, x1, y1))
    return out
# ---------------------------------------------------------------- 合成样本
# 25 张真机图里,有些元素只出现过一两次(比如喝水按钮)。光靠增广,网络见不到
# "它出现在别的位置"的样子。所以再把裁下来的字块贴到"糊掉的背景"上自由摆放 ——
# 位置、尺度、底色都随机,标注天然精确。这本质上是把模板匹配的"素材"变成
# 训练数据,而不是让人手工标几千张图。
PLACE = {
    "row_shop_video": (40, 880, 700, 1800),
    "row_water": (40, 880, 700, 1800),
    "row_live": (40, 880, 700, 1800),
    "coin_entry": (880, 280, 1040, 440),
    "bubble_watch": (840, 260, 1040, 520),
    "bubble_done": (840, 260, 1040, 520),
    "bubble_cooldown": (840, 260, 1040, 520),
    "water_checkin_btn": (160, 1180, 900, 1460),
    "page_coin": (20, 430, 320, 560),
    "water_popup_title": (180, 470, 900, 660),
}


def _crop_pool(samples, spec):
    """按类别收一批"字块"素材(直接从真机图上裁)。"""
    pool = {c: [] for c in CLASSES}
    for fname, img, boxes in samples:
        for c, x0, y0, x1, y1, _s in boxes:
            name = CLASSES[int(c)]
            x0i, y0i = int(max(0, x0)), int(max(0, y0))
            x1i, y1i = int(min(CANVAS_W, x1)), int(min(CANVAS_H, y1))
            if x1i - x0i < 8 or y1i - y0i < 8:
                continue
            pool[name].append(img[y0i:y1i, x0i:x1i].copy())
    return {k: v for k, v in pool.items() if v}


def _mush_background(img, rng):
    """把真机背景糊成"没有字的色块":贴上去的字块才不会被背景上的真字干扰。

    注意:高斯模糊放在**缩小的副本**上做,再放大回去 —— 直接对 1080x1920 做
    GaussianBlur(31,31) 一次要 50ms,是整个数据管线里最贵的一步。
    """
    h, w = img.shape[:2]
    f = float(rng.uniform(0.06, 0.12))
    sw, sh = max(4, int(w * f)), max(4, int(h * f))
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (5, 5), 0)
    bg = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(bg.astype(np.float32) * rng.uniform(0.55, 0.95), 0, 255).astype(np.uint8)


def synthesize(pool, samples, rng):
    """背景糊掉 + 随机贴 3~6 个字块 -> 一张"像是拼多多界面"的样本。"""
    bg = _mush_background(samples[int(rng.integers(0, len(samples)))][1], rng)
    classes = list(pool.keys())
    n = int(rng.integers(3, 7))
    boxes = []
    for _ in range(n):
        cname = classes[int(rng.integers(0, len(classes)))]
        patch = pool[cname][int(rng.integers(0, len(pool[cname])))]
        s = float(rng.uniform(0.85, 1.2))
        ph, pw = patch.shape[:2]
        nw, nh = max(8, int(pw * s)), max(8, int(ph * s))
        patch = cv2.resize(patch, (nw, nh), interpolation=cv2.INTER_AREA)
        # 贴上去之前也做个色彩抖动,避免网络只认那一种底色
        patch = _color(patch, rng)
        rx0, ry0, rx1, ry1 = PLACE[cname]
        px = int(rng.integers(rx0, max(rx0 + 1, rx1 - nw)))
        py = int(rng.integers(ry0, max(ry0 + 1, ry1 - nh)))
        if px + nw >= CANVAS_W or py + nh >= CANVAS_H:
            continue
        alpha = np.ones((nh, nw), np.float32)
        alpha = cv2.GaussianBlur(alpha, (9, 9), 0)[:, :, None]
        roi = bg[py:py + nh, px:px + nw].astype(np.float32)
        bg[py:py + nh, px:px + nw] = (patch * alpha + roi * (1 - alpha)).astype(np.uint8)
        boxes.append((CLASSES.index(cname), float(px), float(py),
                      float(px + nw), float(py + nh), 1.0))
    return bg, boxes
