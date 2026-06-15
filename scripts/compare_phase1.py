#!/usr/bin/env python3
"""Phase 1 모델 비교: CKA / Logit Lens JSD / multi-anchor script-mass.

각 모델의 results/<key>/metrics.json을 읽어 핵심 비교를 출력하고,
results/phase1_compare.json으로 합쳐 저장(플롯/표 생성용)한다.
"""
import argparse, json
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--results", default=str(Path(__file__).resolve().parent.parent / "results"))
ap.add_argument("--models", nargs="+", default=["llama2", "swallow", "llm_jp"])
args = ap.parse_args()

R = Path(args.results)
d = {m: json.load(open(R / m / "metrics.json")) for m in args.models}
M = args.models
L = max(d[m]["num_layers"] for m in M)


def row(label, vals, fmt="{:>10.3f}"):
    return f"{label:<7}" + "".join(fmt.format(v) for v in vals)


print(f"{'model':10} {'tied':5} {'L':>3} {'vocab':>7} {'CKApeak':>9} {'peakL':>5} {'JSD@L-4':>8}")
for m in M:
    x = d[m]; nl = x["num_layers"]
    print(f"{m:10} {str(x['tied_embeddings']):5} {nl:>3} {x['vocab_size']:>7} "
          f"{x['cka_peak']['value']:>9.3f} {x['cka_peak']['layer']:>5} {x['logit_lens_jsd'][str(nl-4)]:>8.3f}")

print("\n=== CKA by layer ===")
print("layer  " + "".join(f"{m:>10}" for m in M))
for l in range(0, L + 1, 5):
    print(row(str(l), [d[m]["cka"].get(str(l), float('nan')) for m in M]))

print("\n=== Logit Lens JSD by layer ===")
print("layer  " + "".join(f"{m:>10}" for m in M))
for l in range(0, L + 1, 5):
    print(row(str(l), [d[m]["logit_lens_jsd"].get(str(l), float('nan')) for m in M]))

print("\n=== JA-input latin (English) script mass — HUB ROUTING signal ===")
print("layer  " + "".join(f"{m:>10}" for m in M))
for l in range(0, L + 1, 5):
    print(row(str(l), [d[m]["script_mass_tgt"]["latin"].get(str(l), float('nan')) for m in M]))

print("\n=== JA-input (kana+han) Japanese script mass ===")
print("layer  " + "".join(f"{m:>10}" for m in M))
for l in range(0, L + 1, 5):
    print(row(str(l), [d[m]["script_mass_tgt"]["kana"].get(str(l), 0) + d[m]["script_mass_tgt"]["han"].get(str(l), 0) for m in M]))

print("\n=== mid-layer (10-25) mean latin mass for JA input (lower = less English-hub) ===")
for m in M:
    mid = np.mean([d[m]["script_mass_tgt"]["latin"][str(l)] for l in range(10, 26)])
    print(f"  {m:10}: {mid:.3f}")

# combined dump for plotting
out = {m: {"cka": d[m]["cka"], "jsd": d[m]["logit_lens_jsd"],
           "latin_ja": d[m]["script_mass_tgt"]["latin"],
           "japanese_ja": {str(l): d[m]["script_mass_tgt"]["kana"][str(l)] + d[m]["script_mass_tgt"]["han"][str(l)]
                           for l in range(d[m]["num_layers"] + 1)},
           "cka_peak": d[m]["cka_peak"], "tied": d[m]["tied_embeddings"],
           "num_layers": d[m]["num_layers"]} for m in M}
json.dump(out, open(R / "phase1_compare.json", "w"), indent=2)
print(f"\nsaved {R / 'phase1_compare.json'}")
