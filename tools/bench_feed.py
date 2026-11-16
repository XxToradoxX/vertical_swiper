# -*- coding: utf-8 -*-
"""判页小网络的评测台:两条后端各跑一遍全部样本,给出分数表 + 延迟。

跑法(Scroller 或 env_gpu 都行,后者能测 GPU):
    python tools\\bench_feed.py                # 默认两条后端都测
    python tools\\bench_feed.py --backend onnx
    python tools\\bench_feed.py --thr 0.5       # 换个阈值看误判

输出三块:
1. 逐图分数(按分数排序,正样本应该在 1.0 附近、负样本接近 0);
2. 阈值扫描(0.30~0.90)下的 TP/FP/FN,用来挑默认阈值;
3. 单帧延迟(纯推理,不含截图)。
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
if APP not in sys.path:
    sys.path.insert(0, APP)

from nn.feed_data import CACHE                                    # noqa: E402
from vision.feedcheck import FeedCheck                            # noqa: E402


def load_cache():
    if not os.path.exists(CACHE):
        raise SystemExit("缺少 %s,先跑 python nn\\feed_data.py" % CACHE)
    data = np.load(CACHE, allow_pickle=True)
    return data["x"], data["y"].astype(np.int32), [str(p) for p in data["paths"]]


def run_backend(chk, xs, ys, paths, thr):
    if not chk.available():
        print("  [%s] 这个环境里用不了,跳过" % chk.want)
        return None
    scores = []
    t0 = time.time()
    for x in xs:
        scores.append(chk.score(x))
    dt = (time.time() - t0) / max(1, len(xs)) * 1000.0
    scores = np.array(scores, dtype=np.float32)
    pred = scores >= thr
    tp = int(np.sum(pred & (ys == 1)))
    fp = int(np.sum(pred & (ys == 0)))
    fn = int(np.sum(~pred & (ys == 1)))
    pos_idx = np.where(ys == 1)[0]
    neg_idx = np.where(ys == 0)[0]
    pos = scores[pos_idx]
    neg = scores[neg_idx]
    name = chk.backend or chk.want
    print("\n== 后端 %s ==" % name)
    print("样本 %d(正 %d / 负 %d)阈值 %.2f → 正确 %d/%d (%.1f%%),漏报 %d,误报 %d"
          % (len(ys), len(pos), len(neg), thr,
             int(np.sum(pred == (ys == 1))), len(ys),
             float(np.sum(pred == (ys == 1))) / len(ys) * 100.0, fn, fp))
    if len(pos):
        i = int(pos_idx[int(np.argmin(pos))])
        print("正样本分数 min %.3f / 中位 %.3f(最低的是:%s)"
              % (pos.min(), float(np.median(pos)), paths[i]))
    if len(neg):
        i = int(neg_idx[int(np.argmax(neg))])
        print("负样本分数 max %.3f / 中位 %.3f(最高的是:%s)"
              % (neg.max(), float(np.median(neg)), paths[i]))
    print("单帧 %.2f ms(不含截图)" % dt)
    return scores


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="all", choices=("all", "onnx", "torch"))
    ap.add_argument("--thr", type=float, default=0.60)
    ap.add_argument("--list", action="store_true", help="打印逐图分数")
    args = ap.parse_args(argv)

    xs, ys, paths = load_cache()
    print("缓存 %s:%d 张(正 %d)" % (CACHE, len(ys), int(np.sum(ys == 1))))

    backs = ["onnx", "torch"] if args.backend == "all" else [args.backend]
    results = {}
    for b in backs:
        results[b] = run_backend(FeedCheck(backend=b, thr=args.thr), xs, ys, paths, args.thr)

    if args.list and results:
        key = "onnx" if results.get("onnx") is not None else list(results)[0]
        print("\n== 逐图分数(%s)==" % key)
        order = np.argsort(-results[key])
        for i in order:
            print("  %.3f  %s  %s" % (results[key][i], "信息流" if ys[i] == 1 else "其他  ", paths[i]))

    print("\n== 阈值扫描 ==")
    for thr in (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90):
        row = []
        for b in backs:
            s = results.get(b)
            if s is None:
                continue
            pred = s >= thr
            fn = int(np.sum(~pred & (ys == 1)))
            fp = int(np.sum(pred & (ys == 0)))
            row.append("%s: 漏报 %d / 误报 %d" % (b, fn, fp))
        print("  %.2f  %s" % (thr, "   ".join(row)))
    return 0


if __name__ == "__main__":
    sys.exit(main())