#!/usr/bin/env python3
"""단일 패칭 결과 파일의 transplant 효과 + bootstrap CI 출력."""
import argparse, json
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--file", required=True)
ap.add_argument("--n-boot", type=int, default=10000)
args = ap.parse_args()
rng = np.random.default_rng(0)

d = json.load(open(args.file))
recs = d["records"]
g_en = np.array([r["gap_en"] for r in recs])
g_tgt = np.array([r["gap_tgt"] for r in recs])
g_pp = np.array([r["gap_paired"] for r in recs])
g_rp = np.array([r["gap_random"] for r in recs])
ps_pp = np.array([r["pstereo_paired"] for r in recs])
ps_rp = np.array([r["pstereo_random"] for r in recs])


def ci(x):
    idx = rng.integers(0, len(x), size=(args.n_boot, len(x)))
    m = x[idx].mean(axis=1)
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


transplant = g_pp - g_rp
dps = ps_pp - ps_rp
toward = np.sign(g_en.mean() - g_tgt.mean())
print("file:", args.file)
print("patch_layers:", d["summary"].get("patch_layers", d["summary"].get("hub_layer")))
print("n:", len(recs))
print("baseline gap_EN=%.3f  gap_TGT=%.3f  (toward_en sign=%+d)" % (g_en.mean(), g_tgt.mean(), toward))
print("Δgap_paired=%.4f  Δgap_random=%.4f" % ((g_pp - g_tgt).mean(), (g_rp - g_tgt).mean()))
m, lo, hi = ci(transplant)
sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "ns"
print("transplant(paired-random) gap = %.4f  CI[%.4f, %.4f]  %s" % (m, lo, hi, sig))
print("  toward_en-signed = %.4f" % (m * toward))
m2, lo2, hi2 = ci(dps)
sig2 = "SIGNIFICANT" if (lo2 > 0 or hi2 < 0) else "ns"
print("ΔP(stereo) paired-random = %.4f  CI[%.4f, %.4f]  %s" % (m2, lo2, hi2, sig2))
