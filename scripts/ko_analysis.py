#!/usr/bin/env python3
"""KO behavioral analysis (no GPU): aggregate EN vs KO with bootstrap CI,
per-category breakdown, and the chrF fidelity-filter demonstration (Step 4)."""
import json, random
from pathlib import Path
from collections import defaultdict

R = Path("results")
ORDER = ["Age", "Disability_status", "Gender_identity", "Physical_appearance", "Sexual_orientation"]


def boot_ci(vals, n_boot=5000):
    rng = random.Random(0)
    n = len(vals)
    means = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def load(model):
    return json.load(open(R / model / "ko_bias.json"))["records"]


chrf = json.load(open(R / "chrf_ko.json"))["records"]

for model in ["llama2_ko", "koen"]:
    recs = load(model)
    en = [r["pstereo_en"] for r in recs]
    ko = [r["pstereo_ko"] for r in recs]
    diffs = [en[i] - ko[i] for i in range(len(recs))]
    lo, hi = boot_ci(diffs)
    print(f"\n=== {model}  (n={len(recs)}) ===")
    print(f"  P_EN={sum(en)/len(en):.3f}  P_KO={sum(ko)/len(ko):.3f}  "
          f"diff={sum(diffs)/len(diffs):+.3f}  95%CI[{lo:+.3f},{hi:+.3f}]"
          f"  {'SIG' if (lo>0 or hi<0) else 'ns'}")
    # per-category
    by = defaultdict(list)
    for i, r in enumerate(recs):
        by[r["category"]].append((en[i], ko[i]))
    print("  per-category:")
    for c in ORDER:
        if c not in by:
            continue
        es = [x for x, _ in by[c]]; ks = [y for _, y in by[c]]
        ds = [es[i] - ks[i] for i in range(len(es))]
        n = len(ds)
        tag = "(insufficient n)" if n < 20 else ""
        if n >= 20:
            l2, h2 = boot_ci(ds)
            sig = "*" if (l2 > 0 or h2 < 0) else " "
        else:
            sig = " "
        print(f"    {c:20} n={n:>3} EN={sum(es)/n:.3f} KO={sum(ks)/n:.3f} diff={sum(ds)/n:+.3f}{sig} {tag}")
    # chrF filter demo (Step 4)
    assert len(chrf) == len(recs)
    cs = [chrf[i]["chrf"] for i in range(len(recs))]
    med = sorted(cs)[len(cs) // 2]
    for thr_name, thr in [("median", med), ("50", 50.0)]:
        keep = [i for i in range(len(recs)) if cs[i] >= thr]
        d = [diffs[i] for i in keep]
        print(f"  chrF>={thr_name}({thr:.1f}): kept {len(keep)}/{len(recs)}, "
              f"diff {sum(diffs)/len(diffs):+.3f} -> {sum(d)/len(d):+.3f}")

ja = json.load(open(R / "chrf_ja.json"))["summary"]
print(f"\nchrF JA fidelity: mean={ja['mean']:.1f} median={ja['median']:.1f} p25={ja['p25']:.1f}")
print(f"chrF KO fidelity: mean={sum(c['chrf'] for c in chrf)/len(chrf):.1f}")
