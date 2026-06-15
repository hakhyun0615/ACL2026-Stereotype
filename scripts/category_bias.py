#!/usr/bin/env python3
"""Per-category English vs Japanese stereotype probability, from saved patching records.
Pools the three Japanese-side models (no GPU; uses results/<m>/patching.json)."""
import json, random
from pathlib import Path
from collections import defaultdict

R = Path(__file__).resolve().parent.parent / "results"
MODELS = ["llama2", "swallow", "llm_jp"]
ORDER = ["Age", "Disability_status", "Gender_identity", "Physical_appearance", "Sexual_orientation"]

bycat = defaultdict(lambda: {"en": [], "ja": []})
for m in MODELS:
    for r in json.load(open(R / m / "patching.json"))["records"]:
        bycat[r["category"]]["en"].append(r["pstereo_en"])
        bycat[r["category"]]["ja"].append(r["pstereo_tgt"])


def boot_ci(vals, n_boot=5000):
    rng = random.Random(0)
    n = len(vals)
    means = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


print(f"{'category':20} {'n':>5} {'P_EN':>6} {'P_JA':>6} {'diff':>7}   95% CI(diff)")
for c in ORDER:
    if c not in bycat:
        continue
    en, ja = bycat[c]["en"], bycat[c]["ja"]
    n = len(en)
    me, mj = sum(en) / n, sum(ja) / n
    diffs = [en[i] - ja[i] for i in range(n)]
    lo, hi = boot_ci(diffs)
    sig = "*" if (lo > 0 or hi < 0) else " "
    print(f"{c:20} {n:>5} {me:>6.3f} {mj:>6.3f} {me-mj:>+7.3f}{sig}  [{lo:+.3f}, {hi:+.3f}]")
