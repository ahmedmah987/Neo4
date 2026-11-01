"""Test the bound of the paper's section 6 on a holdout run whose level_log carries the lives each
completed level took (from the Phase 72 logging on): per level, its actual share min(1.15, (h/a)^2)
against 1/L^2 (lives only) and (1+(L-1)b)^-2 (lives and the measured cost b*h of a lost life).

    PYTHONHASHSEED=0 python -m neo4.run run --agent library-holdout --seeds 5 --label hold72-lives
    python scripts/lives_bound.py hold72-lives
"""
import json, sys, logging, statistics as st
from collections import defaultdict
from pathlib import Path
logging.disable(logging.INFO)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from neo4.run import RAW, _baselines
B = _baselines()
rows = [json.loads(l) for l in open(RAW / f"{sys.argv[1]}.jsonl") if l.strip()]
lv = []  # (game, seed, i, h, a, w, L)
score = defaultdict(lambda: [0.0, 0.0, 0.0]); nseed = defaultdict(int)
for r in rows:
    if "error" in r: continue
    b = B.get(r["game"], []); n = len(b); W = sum(range(1, n + 1))
    nseed[r["game"]] += 1
    for i, e in enumerate(r["stats"].get("level_log", [])):
        if i >= n: break
        a, d, w, L = e[0], e[1], e[2], e[3]
        h = b[i]
        act = min(1.15, (h / max(a, 1)) ** 2)
        p1 = min(1.15, 1 / L ** 2)
        p2 = min(1.15, (h / (h + a - w)) ** 2)
        lv.append((r["game"], r["seed"], i, h, a, w, L, act, p1, p2))
        for k, v in enumerate((act, p1, p2)):
            score[r["game"]][k] += 100 * v * (i + 1) / W
tot = [sum(score[g][k] / nseed[g] for g in nseed) / 25 for k in range(3)]
print(f"levels {len(lv)}; holdout (mean of 25 games, %): actual {tot[0]:.2f}  1/L^2 {tot[1]:.2f}  "
      f"(1+(L-1)b)^-2 {tot[2]:.2f}")
Ls = [x[6] for x in lv]
print("lives: median", st.median(Ls), " L=1:", sum(L == 1 for L in Ls), " L>=10:", sum(L >= 10 for L in Ls))
bs = [(x[4] - x[5]) / ((x[6] - 1) * x[3]) for x in lv if x[6] > 1]
print("b (lost-life cost / h): median %.2f  quartiles %.2f %.2f" % (st.median(bs), *st.quantiles(bs, n=4)[::2]))
ws = [x[5] / x[3] for x in lv]
print("winning life / h: median %.2f" % st.median(ws), " first levels %.2f" % st.median([x[5] / x[3] for x in lv if x[2] == 0]),
      " later %.2f" % st.median([x[5] / x[3] for x in lv if x[2] > 0] or [0]))
print("lost-life share of actions: %.0f%%" % (100 * sum(x[4] - x[5] for x in lv) / sum(x[4] for x in lv)))
def rank(v):  # average ranks for ties
    o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); k = 0
    while k < len(o):
        j = k
        while j + 1 < len(o) and v[o[j + 1]] == v[o[k]]: j += 1
        for m in range(k, j + 1): r[o[m]] = (k + j) / 2
        k = j + 1
    return r
def spear(x, y):
    rx, ry = rank(x), rank(y); return st.correlation(rx, ry)
act = [x[7] for x in lv]
print("Spearman(actual, 1/L^2) %.2f   Spearman(actual, (1+(L-1)b)^-2) %.2f" % (spear(act, [x[8] for x in lv]), spear(act, [x[9] for x in lv])))
within2 = sum(1 for x in lv if x[7] > 0 and 0.5 <= x[8] / x[7] <= 2)
print(f"1/L^2 within a factor 2 of the actual share: {within2}/{len(lv)}")
shares = [100 * x[7] for x in lv]
by = defaultdict(float)
for x in lv:
    W = sum(range(1, len(B[x[0]]) + 1)); by[min(x[6], 6)] += 100 * x[7] * (x[2] + 1) / W / nseed[x[0]] / 25
print("holdout score by lives (1..5, 6+):", {k: round(v, 3) for k, v in sorted(by.items())})
print()
print(f"{'lives':>8s} {'levels':>6s} {'actual':>7s} {'1/L^2':>7s} {'b-form':>7s} {'b med':>6s} {'win/h':>6s}")
for lo, hi in ((1, 1), (2, 2), (3, 5), (6, 20), (21, 10**9)):
    g = [x for x in lv if lo <= x[6] <= hi]
    if not g: continue
    bb = [(x[4] - x[5]) / ((x[6] - 1) * x[3]) for x in g if x[6] > 1]
    print(f"{(str(lo) if lo == hi else f'{lo}-{hi if hi < 10**9 else chr(8734)}'):>8s} {len(g):6d} "
          f"{st.mean(x[7] for x in g):7.3f} {st.mean(x[8] for x in g):7.3f} {st.mean(x[9] for x in g):7.3f} "
          f"{(st.median(bb) if bb else float('nan')):6.2f} {st.median(x[5] / x[3] for x in g):6.2f}")
