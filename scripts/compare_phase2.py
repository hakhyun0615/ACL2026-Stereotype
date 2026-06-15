#!/usr/bin/env python3
"""Phase 2 패칭 결과 비교: 허브 패칭의 편향-이식 효과가 Phase 1 허브 의존 순서와 일치하는가.

각 모델 results/<key>/patching.json(records) + metrics.json(Phase 1 latin mass)을 읽어
- baseline bias gap (EN vs TGT)
- paired/random 패칭의 gap shift
- control-corrected transplant = mean(gap_paired - gap_random) + bootstrap 95% CI
- "toward English" 방향 정렬
을 모델별로 출력한다.
"""
import argparse, json
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--results", default=str(Path(__file__).resolve().parent.parent / "results"))
ap.add_argument("--models", nargs="+", default=["llama2", "swallow", "llm_jp"])
ap.add_argument("--n-boot", type=int, default=5000)
args = ap.parse_args()
R = Path(args.results)
rng = np.random.default_rng(0)


def boot_ci(x, n_boot, lo=2.5, hi=97.5):
    x = np.asarray(x, float)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, lo)), float(np.percentile(means, hi))


def midlatin(metrics):
    lm = metrics["script_mass_tgt"]["latin"]
    return float(np.mean([lm[str(l)] for l in range(10, 26)]))


print(f"{'model':9} {'n':>4} {'gap_EN':>7} {'gap_TGT':>7} | "
      f"{'Δpaired':>8} {'Δrandom':>8} | {'transplant(paired-random)':>26} | {'towardEN':>9} | {'P1 latin':>8}")
rows = {}
for m in args.models:
    pj = json.load(open(R / m / "patching.json"))
    recs = pj["records"]
    g_en = np.array([r["gap_en"] for r in recs])
    g_tgt = np.array([r["gap_tgt"] for r in recs])
    g_pp = np.array([r["gap_paired"] for r in recs])
    g_rp = np.array([r["gap_random"] for r in recs])
    transplant = g_pp - g_rp                      # control-corrected, per item
    ci = boot_ci(transplant, args.n_boot)
    toward = np.sign(g_en.mean() - g_tgt.mean())  # EN이 더 stereo면 +1
    transplant_toward_en = float(transplant.mean() * toward)
    mt = midlatin(json.load(open(R / m / "metrics.json")))
    rows[m] = dict(transplant=float(transplant.mean()), ci=ci, toward=float(toward),
                   transplant_toward_en=transplant_toward_en, midlatin=mt,
                   gap_en=float(g_en.mean()), gap_tgt=float(g_tgt.mean()))
    sig = "*" if (ci[0] > 0 or ci[1] < 0) else " "
    print(f"{m:9} {len(recs):>4} {g_en.mean():>7.3f} {g_tgt.mean():>7.3f} | "
          f"{g_pp.mean()-g_tgt.mean():>8.3f} {g_rp.mean()-g_tgt.mean():>8.3f} | "
          f"{transplant.mean():>10.3f} [{ci[0]:.3f},{ci[1]:.3f}]{sig:>2} | "
          f"{transplant_toward_en:>9.3f} | {mt:>8.3f}")

print("\n해석: transplant(paired-random) CI가 0을 포함하지 않으면(*) 유의. "
      "transplant_toward_en(>0이면 영어 편향 방향으로 이식) 크기가 P1 latin(허브 의존)과 같은 순서면 메커니즘 일관.")
print("\n=== ΔP(stereotyped): paired - random ===")
for m in args.models:
    recs = json.load(open(R / m / "patching.json"))["records"]
    dp = np.array([r["pstereo_paired"] - r["pstereo_random"] for r in recs])
    ci = boot_ci(dp, args.n_boot)
    print(f"  {m:9}: {dp.mean():+.4f}  CI[{ci[0]:+.4f},{ci[1]:+.4f}]  "
          f"(P_stereo EN={np.mean([r['pstereo_en'] for r in recs]):.3f} "
          f"TGT={np.mean([r['pstereo_tgt'] for r in recs]):.3f})")

json.dump(rows, open(R / "phase2_compare.json", "w"), indent=2)
print(f"\nsaved {R/'phase2_compare.json'}")
