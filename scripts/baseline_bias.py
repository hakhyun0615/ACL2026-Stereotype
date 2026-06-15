#!/usr/bin/env python3
"""Baseline 편향 비대칭: 영어 처리가 일본어 처리보다 더 고정관념적인가 (패칭 전 baseline)."""
import json
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parent.parent / "results"
rng = np.random.default_rng(0)
print(f"{'model':9} {'P_stereo_EN':>11} {'P_stereo_TGT':>12} {'diff':>7} {'95% CI':>20} {'frac(EN>TGT)':>12} {'gap_EN':>7} {'gap_TGT':>7}")
for m in ["llama2", "swallow", "llm_jp"]:
    recs = json.load(open(R / m / "patching.json"))["records"]
    en = np.array([r["pstereo_en"] for r in recs])
    tg = np.array([r["pstereo_tgt"] for r in recs])
    d = en - tg
    idx = rng.integers(0, len(d), size=(10000, len(d)))
    bm = d[idx].mean(1)
    lo, hi = np.percentile(bm, 2.5), np.percentile(bm, 97.5)
    ge = np.mean([r["gap_en"] for r in recs]); gt = np.mean([r["gap_tgt"] for r in recs])
    sig = "*" if (lo > 0 or hi < 0) else " "
    print(f"{m:9} {en.mean():>11.3f} {tg.mean():>12.3f} {d.mean():>+7.3f} "
          f"[{lo:>+.3f},{hi:>+.3f}]{sig} {(d>0).mean():>11.2f} {ge:>+7.3f} {gt:>+7.3f}")
print("\n* = 95% CI가 0 제외. P_stereo: 3지선다라 chance=0.333. gap=logp(stereo)-logp(anti).")
