#!/usr/bin/env python3
"""Phase 2: 인과적 허브 패칭 (bias outcome).

영어 prompt의 hub-layer 마지막 토큰 residual을 타깃언어(JA/KO) 처리에 주입했을 때,
타깃언어의 편향(고정관념 답 선호)이 영어쪽으로 이식되는지 측정한다.

결과변수:
  - primary  : stereotyped vs anti-stereotyped 답의 log-prob gap = logp(stereo) - logp(anti)
  - secondary: ΔP(stereotyped) (3개 답 softmax)
  - tertiary : argmax 답 (sAMB/sDIS 집계용)

조건: no_patch(baseline) / paired_english / random_english.
main의 route_audit/intervention.py 패칭 훅을 차용·적응.

untied 모델이라 Logit Lens는 보조였고, 이 패칭이 lens-free 주(主) 인과 증거다.
"""
import argparse, json, sys, random
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data_prep.load_bbq_jbbq import sample_paired_templates  # noqa: E402

UNKNOWN_GROUPS = {"unknown", "undetermined", "cannot be determined", "can't be determined", "not enough info"}


def answer_roles(bbq_item):
    """answer_info + stereotyped_groups로 (stereo_idx, anti_idx, unknown_idx) 반환. 못 풀면 None."""
    info = bbq_item.get("answer_info", {})
    stereo_groups = [g.lower() for g in bbq_item.get("additional_metadata", {}).get("stereotyped_groups", [])]
    roles = {}
    for i in range(3):
        ai = info.get(f"ans{i}")
        if not ai:
            return None
        grp = str(ai[-1]).lower()
        if grp in UNKNOWN_GROUPS:
            roles.setdefault("unknown", i)
        elif grp in stereo_groups:
            roles.setdefault("stereo", i)
        else:
            roles.setdefault("anti", i)
    if set(roles) != {"stereo", "anti", "unknown"}:
        return None
    return roles["stereo"], roles["anti"], roles["unknown"]


def make_prompt(context, question):
    return f"{context}\n{question}\n"


def patch_hook(module, pos, vec):
    def hook(_, __, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        h = h.clone()
        h[:, pos, :] = vec.to(dtype=h.dtype, device=h.device)
        return (h, *output[1:]) if is_tuple else h
    return module.register_forward_hook(hook)


@torch.no_grad()
def residuals_at(model, tok, prompt, layers):
    """prompt 마지막 토큰의 여러 layer hidden state (hidden_states[L]) dict."""
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    out = model(input_ids=ids, output_hidden_states=True)
    return {int(L): out.hidden_states[L][0, -1, :].clone() for L in layers}


@torch.no_grad()
def score_candidates(model, tok, prompt, cands, patch=None):
    """후보별 log P(cand | prompt) 합. patch={layer:vec} 주어지면 각 layer block 출력의 마지막 prompt 위치를 교체."""
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

    handles = []
    if patch:
        for L, vec in patch.items():
            handles.append(patch_hook(model.model.layers[L - 1], plen - 1, vec))
    try:
        logits = model(input_ids=inp, attention_mask=att, use_cache=False).logits.float()
        logp = torch.log_softmax(logits, dim=-1)
    finally:
        for h in handles:
            h.remove()

    scores = []
    for i, c in enumerate(c_ids):
        s = 0.0
        for j, tid in enumerate(c):
            s += float(logp[i, plen + j - 1, tid])
        scores.append(s)
    return np.array(scores)  # [3] in cand order


def gap_and_probs(scores, stereo_i, anti_i, unk_i):
    gap = scores[stereo_i] - scores[anti_i]
    p = np.exp(scores - scores.max()); p = p / p.sum()
    return float(gap), float(p[stereo_i]), int(np.argmax(scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-id", required=True)
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--pairs", default=str(ROOT / "data/processed/paired_templates.json"))
    ap.add_argument("--src-ctx", default="context_en")        # 영어 context 키
    ap.add_argument("--tgt-ctx", default="context_ja")        # 타깃 context 키
    ap.add_argument("--src-item", default="bbq_item")         # 영어 답/메타 키
    ap.add_argument("--tgt-item", default="jbbq_item")        # 타깃 답 텍스트 키
    ap.add_argument("--hub-layer", type=int, default=0, help="0이면 metrics.json의 CKA peak 사용")
    ap.add_argument("--hub-layers", default="", help="예: '10-25' (주면 multi-layer 패칭, --hub-layer 무시)")
    ap.add_argument("--tag", default="", help="출력 파일명 접미사 (예: _multi)")
    ap.add_argument("--n-items", type=int, default=200)
    ap.add_argument("--ambig-only", action="store_true", default=True)
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.hub_layers:
        a, b = args.hub_layers.split("-")
        layer_list = list(range(int(a), int(b) + 1))
    else:
        hl = args.hub_layer
        if hl == 0:
            m = json.load(open(Path(args.out_dir) / args.model_key / "metrics.json"))
            hl = int(m["cka_peak"]["layer"])
        layer_list = [hl]
    print(f"[{args.model_key}] patch_layers={layer_list}", flush=True)

    data = json.load(open(args.pairs))
    sampled = sample_paired_templates(data, max_per_context=3)
    items = []
    for p in sampled:
        bi = p.get(args.src_item, {})
        if args.ambig_only and bi.get("context_condition") != "ambig":
            continue
        roles = answer_roles(bi)
        if roles is None:
            continue
        items.append((p, roles))
        if len(items) >= args.n_items:
            break
    print(f"[{args.model_key}] usable ambiguous items: {len(items)}", flush=True)

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
    for idx, (p, roles) in enumerate(items):
        s_i, a_i, u_i = roles
        bi, ti = p[args.src_item], p[args.tgt_item]
        en_cands = [bi["ans0"], bi["ans1"], bi["ans2"]]
        tgt_cands = [ti["ans0"], ti["ans1"], ti["ans2"]]
        en_prompt = make_prompt(p[args.src_ctx], bi["question"])
        tgt_prompt = make_prompt(p[args.tgt_ctx], ti["question"])

        en_scores = score_candidates(model, tok, en_prompt, en_cands)
        tgt_scores = score_candidates(model, tok, tgt_prompt, tgt_cands)
        en_res = residuals_at(model, tok, en_prompt, layer_list)
        patched = score_candidates(model, tok, tgt_prompt, tgt_cands, patch=en_res)
        # random_english control: 다른 아이템의 영어 residual
        rp, _ = items[rng.randrange(len(items))]
        rres = residuals_at(model, tok, make_prompt(rp[args.src_ctx], rp[args.src_item]["question"]), layer_list)
        rand = score_candidates(model, tok, tgt_prompt, tgt_cands, patch=rres)

        g_en, ps_en, am_en = gap_and_probs(en_scores, s_i, a_i, u_i)
        g_tgt, ps_tgt, am_tgt = gap_and_probs(tgt_scores, s_i, a_i, u_i)
        g_pp, ps_pp, am_pp = gap_and_probs(patched, s_i, a_i, u_i)
        g_rp, ps_rp, am_rp = gap_and_probs(rand, s_i, a_i, u_i)
        recs.append(dict(category=p["category"],
                         gap_en=g_en, gap_tgt=g_tgt, gap_paired=g_pp, gap_random=g_rp,
                         pstereo_en=ps_en, pstereo_tgt=ps_tgt, pstereo_paired=ps_pp, pstereo_random=ps_rp,
                         argmax_tgt=am_tgt, argmax_paired=am_pp, stereo_i=s_i, anti_i=a_i, unk_i=u_i))
        if (idx + 1) % 50 == 0:
            print(f"  {idx+1}/{len(items)}", flush=True)

    arr = lambda k: np.array([r[k] for r in recs], dtype=float)
    # paired가 random보다 gap을 EN쪽으로 더 옮기는가?
    d_paired = arr("gap_paired") - arr("gap_tgt")
    d_random = arr("gap_random") - arr("gap_tgt")
    toward_en = np.sign(arr("gap_en") - arr("gap_tgt"))   # EN이 TGT보다 큰 방향(+) 작은 방향(-)
    summary = dict(
        model_key=args.model_key, patch_layers=layer_list, n=len(recs),
        mean_gap_en=float(arr("gap_en").mean()), mean_gap_tgt=float(arr("gap_tgt").mean()),
        mean_gap_paired=float(arr("gap_paired").mean()), mean_gap_random=float(arr("gap_random").mean()),
        delta_gap_paired=float(d_paired.mean()), delta_gap_random=float(d_random.mean()),
        # 핵심: paired 패칭이 gap을 EN 방향으로 옮긴 정도 (control 대비)
        signed_shift_toward_en_paired=float((d_paired * toward_en).mean()),
        signed_shift_toward_en_random=float((d_random * toward_en).mean()),
        mean_pstereo_en=float(arr("pstereo_en").mean()), mean_pstereo_tgt=float(arr("pstereo_tgt").mean()),
        mean_pstereo_paired=float(arr("pstereo_paired").mean()), mean_pstereo_random=float(arr("pstereo_random").mean()),
    )
    out_dir = Path(args.out_dir) / args.model_key
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": summary, "records": recs}, open(out_dir / f"patching{args.tag}.json", "w"), indent=2)
    print("[summary]", json.dumps(summary, indent=2), flush=True)
    print(f"[{args.model_key}] saved {out_dir / f'patching{args.tag}.json'}", flush=True)


if __name__ == "__main__":
    main()
