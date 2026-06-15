#!/usr/bin/env python3
"""KO hub patching: does injecting the English hub-layer state into Korean
processing transplant the stereotype preference? Reuses hub_patching.py's
machinery (score_candidates / residuals_at / gap_and_probs) with the flat
KoBBQ structure and ko_bias.py's role parsing. Same 261-item sample, hub layer
= CKA peak (15 for both KO models). Output schema matches patching.json so the
same analysis/figure code applies (target = Korean, stored as *_tgt)."""
import argparse, json, sys, random
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src.data_prep.load_bbq_jbbq import sample_paired_templates  # noqa: E402
from ko_bias import en_roles, ko_roles, make_prompt  # noqa: E402
from hub_patching import score_candidates, residuals_at, gap_and_probs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-id", required=True)
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--pairs", default=str(ROOT / "data/processed/paired_ko.json"))
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data = json.load(open(args.pairs))
    items = []
    for p in sample_paired_templates(data, max_per_context=3):
        er = en_roles(p)
        kc, kr = ko_roles(p)
        if er is None or kr is None:
            continue
        items.append((p, er, kc, kr))
    m = json.load(open(Path(args.out_dir) / args.model_key / "metrics.json"))
    hub = int(m["cka_peak"]["layer"])
    layer_list = [hub]
    print(f"[{args.model_key}] usable={len(items)} hub_layer={hub}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.hf_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dt = getattr(torch, args.dtype)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.hf_id, dtype=dt, device_map={"": 0})
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.hf_id, torch_dtype=dt, device_map={"": 0})
    model.eval()

    rng = random.Random(args.seed)
    recs = []
    for idx, (p, er, kc, kr) in enumerate(items):
        en_cands = [p["ans0"], p["ans1"], p["ans2"]]
        en_prompt = make_prompt(p["context_en"], p["question_en"])
        ko_prompt = make_prompt(p["context_ko"], p["question_ko"])
        en_scores = score_candidates(model, tok, en_prompt, en_cands)
        ko_scores = score_candidates(model, tok, ko_prompt, kc)
        en_res = residuals_at(model, tok, en_prompt, layer_list)
        patched = score_candidates(model, tok, ko_prompt, kc, patch=en_res)
        rp = items[rng.randrange(len(items))][0]
        rres = residuals_at(model, tok, make_prompt(rp["context_en"], rp["question_en"]), layer_list)
        rand = score_candidates(model, tok, ko_prompt, kc, patch=rres)

        g_en, ps_en, _ = gap_and_probs(en_scores, *er)
        g_tgt, ps_tgt, _ = gap_and_probs(ko_scores, *kr)
        g_pp, ps_pp, _ = gap_and_probs(patched, *kr)
        g_rp, ps_rp, _ = gap_and_probs(rand, *kr)
        recs.append(dict(category=p["category"], gap_en=g_en, gap_tgt=g_tgt,
                         gap_paired=g_pp, gap_random=g_rp,
                         pstereo_en=ps_en, pstereo_tgt=ps_tgt,
                         pstereo_paired=ps_pp, pstereo_random=ps_rp))
        if (idx + 1) % 50 == 0:
            print(f"  {idx+1}/{len(items)}", flush=True)

    arr = lambda k: np.array([r[k] for r in recs], dtype=float)
    transplant = arr("gap_paired") - arr("gap_random")
    summary = dict(model_key=args.model_key, patch_layers=layer_list, n=len(recs),
                   mean_gap_en=float(arr("gap_en").mean()), mean_gap_tgt=float(arr("gap_tgt").mean()),
                   mean_gap_paired=float(arr("gap_paired").mean()),
                   mean_gap_random=float(arr("gap_random").mean()),
                   delta_gap_paired=float((arr("gap_paired") - arr("gap_tgt")).mean()),
                   delta_gap_random=float((arr("gap_random") - arr("gap_tgt")).mean()),
                   transplant_effect=float(transplant.mean()))
    out = Path(args.out_dir) / args.model_key
    json.dump({"summary": summary, "records": recs}, open(out / "patching.json", "w"), indent=2)
    print("[summary]", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
