#!/usr/bin/env python3
"""KO behavioral bias: P(stereotyped) for English vs Korean processing.

Mirrors the Japanese baseline exactly (same make_prompt / score_candidates /
P_stereo = softmax over the 3 options as hub_patching.py), but identifies answer
roles from KoBBQ fields (choices / biased_answer / answer) because KoBBQ does not
preserve BBQ's ans0/1/2 order. Same 261-item sample as the KO representation
analysis (sample_paired_templates, seed 42), so representation and behavior sit
on the same data. No patching here, just the baseline readout.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data_prep.load_bbq_jbbq import sample_paired_templates  # noqa: E402

UNKNOWN_GROUPS = {"unknown", "undetermined", "cannot be determined",
                  "can't be determined", "not enough info"}

# KoBBQ codes gender stereotyped_groups as F/M/trans while answer_info uses
# man/woman/boy/girl/trans/nonTrans; canonicalize so the two sides match.
GENDER_CANON = {
    "f": "F", "woman": "F", "girl": "F",
    "m": "M", "man": "M", "boy": "M",
    "trans": "trans", "transgender women": "trans", "transgender men": "trans",
    "transgender woman": "trans", "transgender man": "trans", "nontrans": "nonTrans",
}


def _canon(g):
    return GENDER_CANON.get(str(g).lower(), str(g).lower())


def en_roles(item):
    """English (stereo, anti, unknown) indices from answer_info + stereotyped_groups."""
    info = item.get("answer_info", {})
    sg = {_canon(g) for g in item.get("stereotyped_groups", [])}
    roles = {}
    for i in range(3):
        ai = info.get(f"ans{i}")
        if not ai:
            return None
        grp = str(ai[-1]).lower()
        if grp in UNKNOWN_GROUPS:
            roles.setdefault("unknown", i)
        elif _canon(grp) in sg:
            roles.setdefault("stereo", i)
        else:
            roles.setdefault("anti", i)
    if set(roles) != {"stereo", "anti", "unknown"}:
        return None
    return roles["stereo"], roles["anti"], roles["unknown"]


def ko_roles(item):
    """Korean (choices, (stereo, anti, unknown)) from choices / biased_answer / answer."""
    choices = item["choices"]
    if isinstance(choices, str):
        import ast
        choices = ast.literal_eval(choices)
    if len(choices) != 3:
        return None, None
    try:
        s = choices.index(item["biased_answer"])
        u = choices.index(item["answer"])
    except ValueError:
        return None, None
    rest = {0, 1, 2} - {s, u}
    if len(rest) != 1:
        return None, None
    return choices, (s, rest.pop(), u)


def make_prompt(context, question):
    return f"{context}\n{question}\n"


@torch.no_grad()
def score_candidates(model, tok, prompt, cands):
    """Sum log P(cand tokens | prompt) for each candidate (teacher forced)."""
    p_ids = tok(prompt, add_special_tokens=True)["input_ids"]
    plen = len(p_ids)
    c_ids = [tok(c, add_special_tokens=False)["input_ids"] for c in cands]
    seqs = [p_ids + c for c in c_ids]
    maxlen = max(len(s) for s in seqs)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    inp = torch.full((len(seqs), maxlen), pad, dtype=torch.long, device=model.device)
    att = torch.zeros((len(seqs), maxlen), dtype=torch.long, device=model.device)
    for i, s in enumerate(seqs):
        inp[i, :len(s)] = torch.tensor(s, device=model.device)
        att[i, :len(s)] = 1
    logits = model(input_ids=inp, attention_mask=att, use_cache=False).logits.float()
    logp = torch.log_softmax(logits, dim=-1)
    scores = []
    for i, c in enumerate(c_ids):
        s = sum(float(logp[i, plen + j - 1, tid]) for j, tid in enumerate(c))
        scores.append(s)
    return np.array(scores)


def pstereo(scores, s_i, a_i, u_i):
    p = np.exp(scores - scores.max())
    p = p / p.sum()
    return float(p[s_i]), float(scores[s_i] - scores[a_i]), int(np.argmax(scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-id", required=True)
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--pairs", default=str(ROOT / "data/processed/paired_ko.json"))
    ap.add_argument("--max-per-context", type=int, default=3)
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    data = json.load(open(args.pairs))
    sampled = sample_paired_templates(data, max_per_context=args.max_per_context)
    items = []
    for p in sampled:
        er = en_roles(p)
        kc, kr = ko_roles(p)
        if er is None or kr is None:
            continue
        items.append((p, er, kc, kr))
    print(f"[{args.model_key}] usable items: {len(items)} / sampled {len(sampled)}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.hf_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dt = getattr(torch, args.dtype)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.hf_id, dtype=dt, device_map={"": 0})
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.hf_id, torch_dtype=dt, device_map={"": 0})
    model.eval()

    recs = []
    for idx, (p, er, kc, kr) in enumerate(items):
        en_cands = [p["ans0"], p["ans1"], p["ans2"]]
        en_scores = score_candidates(model, tok, make_prompt(p["context_en"], p["question_en"]), en_cands)
        ko_scores = score_candidates(model, tok, make_prompt(p["context_ko"], p["question_ko"]), kc)
        ps_en, gap_en, am_en = pstereo(en_scores, *er)
        ps_ko, gap_ko, am_ko = pstereo(ko_scores, *kr)
        recs.append(dict(category=p["category"], pstereo_en=ps_en, pstereo_ko=ps_ko,
                         gap_en=gap_en, gap_ko=gap_ko, argmax_en=am_en, argmax_ko=am_ko))
        if (idx + 1) % 50 == 0:
            print(f"  {idx+1}/{len(items)}", flush=True)

    arr = lambda k: np.array([r[k] for r in recs], dtype=float)
    summary = dict(model_key=args.model_key, hf_id=args.hf_id, n=len(recs),
                   mean_pstereo_en=float(arr("pstereo_en").mean()),
                   mean_pstereo_ko=float(arr("pstereo_ko").mean()),
                   mean_diff=float((arr("pstereo_en") - arr("pstereo_ko")).mean()))
    out_dir = Path(args.out_dir) / args.model_key
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": summary, "records": recs}, open(out_dir / "ko_bias.json", "w"), indent=2)
    print("[summary]", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
