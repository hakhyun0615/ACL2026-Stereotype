#!/usr/bin/env python3
"""순차 단일 모델 추출 + 메트릭 (EN-JA / EN-KO 편향 벤치마크 감사).

한 모델을 로드해 paired context(EN vs target)의 최종 토큰 hidden state를 전 레이어에서 뽑고,
같은 forward에서 (1) semantic-bin Logit Lens 분포, (2) script-mass(multi-anchor) 분포를 계산한다.
이어서 레이어별 linear CKA, Logit Lens JSD, script-mass 프로파일을 산출해 JSON으로 저장한다.

층 수는 model.config.num_hidden_layers에서 직접 읽으므로 검증용 작은 모델(예: Llama-3.2-1B, 16층)에도
그대로 동작한다. 디스크가 빠듯하므로 모델을 한 번에 하나만 들고, 끝나면 호출측에서 가중치를 지운다.

주의: 기존 논문 코드와 동일하게 raw lm_head 투영(final RMSNorm 미적용)을 쓴다(--final-norm으로 변경 가능).
모든 대상 모델은 tie_word_embeddings=False(untied)라 Logit Lens는 caution과 함께 해석하고
패칭(Phase 2)으로 교차검증한다.
"""
import argparse, json, gc, sys, unicodedata, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.metrics.cka import linear_cka                      # noqa: E402
from src.utils import stable_jsd_squared, check_tied_embeddings  # noqa: E402
from src.extraction.extract_logit_lens import SEMANTIC_BINS  # noqa: E402
from src.data_prep.load_bbq_jbbq import sample_paired_templates  # noqa: E402

SCRIPT_CLASSES = ["latin", "kana", "han", "hangul", "digit", "other"]


def classify_script(s: str) -> str:
    """디코딩된 토큰 문자열의 지배적 script를 반환."""
    counts = {}
    for ch in s:
        if ch.isspace():
            continue
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        if "LATIN" in name:
            k = "latin"
        elif "HIRAGANA" in name or "KATAKANA" in name:
            k = "kana"
        elif "CJK" in name:                 # 한자(일본 Kanji = 중국어와 공유)
            k = "han"
        elif "HANGUL" in name:
            k = "hangul"
        elif "DIGIT" in name:
            k = "digit"
        else:
            k = "other"
        counts[k] = counts.get(k, 0) + 1
    return max(counts, key=counts.get) if counts else "other"


def build_script_masks(tokenizer, vocab_size, device):
    """vocab 전체를 script로 분류해 클래스별 boolean mask를 미리 만든다(1회)."""
    labels = np.array([classify_script(tokenizer.decode([tid])) for tid in range(vocab_size)])
    masks = {}
    for c in SCRIPT_CLASSES:
        masks[c] = torch.tensor(labels == c, device=device)
    return masks


def build_semantic_token_ids(tokenizer, vocab_size):
    """semantic bin별 term들을 토큰 id 집합으로 미리 변환."""
    bin_ids = {}
    for name, terms in SEMANTIC_BINS.items():
        ids = set()
        for t in terms:
            for tid in tokenizer.encode(t, add_special_tokens=False):
                if 0 <= tid < vocab_size:
                    ids.add(tid)
        bin_ids[name] = sorted(ids)
    return bin_ids


@torch.no_grad()
def process_text(model, tokenizer, text, script_masks, bin_ids, apply_final_norm):
    """한 텍스트 -> (hidden[L+1,H] fp16 numpy, layer별 script_mass, layer별 semantic_bin)."""
    ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    out = model(input_ids=ids, output_hidden_states=True)
    hs = torch.stack([h[0, -1, :] for h in out.hidden_states])      # [L+1, H]
    proj_in = model.model.norm(hs) if apply_final_norm else hs
    logits = model.lm_head(proj_in.to(model.lm_head.weight.dtype))   # [L+1, V]
    probs = torch.softmax(logits.float(), dim=-1)                    # [L+1, V]

    smass = {c: probs[:, m].sum(dim=1).cpu().numpy() for c, m in script_masks.items()}  # each [L+1]
    sem = {b: (probs[:, ids_].sum(dim=1).cpu().numpy() if ids_ else np.zeros(probs.shape[0]))
           for b, ids_ in bin_ids.items()}
    return hs.cpu().half().numpy(), smass, sem


def jsd_from_bins(sem_en_layer, sem_ja_layer):
    bins = sorted(sem_en_layer.keys())
    p = np.array([sem_en_layer[b] for b in bins], dtype=np.float64)
    q = np.array([sem_ja_layer[b] for b in bins], dtype=np.float64)
    if p.sum() < 1e-12 or q.sum() < 1e-12:
        return np.nan
    return stable_jsd_squared(p / p.sum(), q / q.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-id", required=True)
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--pairs", default=str(ROOT / "data/processed/paired_templates.json"))
    ap.add_argument("--src-key", default="context_en")
    ap.add_argument("--tgt-key", default="context_ja")
    ap.add_argument("--n-pairs", type=int, default=0, help="0 = 전체 샘플(2142)")
    ap.add_argument("--max-per-context", type=int, default=3)
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--final-norm", action="store_true")
    ap.add_argument("--save-hidden", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    data = json.load(open(args.pairs))
    sampled = sample_paired_templates(data, max_per_context=args.max_per_context)
    if args.n_pairs:
        sampled = sampled[:args.n_pairs]
    print(f"[{args.model_key}] pairs={len(sampled)} (from {len(data)})", flush=True)

    tok = AutoTokenizer.from_pretrained(args.hf_id)
    dt = getattr(torch, args.dtype)
    try:  # transformers >=5 uses dtype=; <5 uses torch_dtype=
        model = AutoModelForCausalLM.from_pretrained(
            args.hf_id, dtype=dt, device_map={"": 0}, output_hidden_states=True)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.hf_id, torch_dtype=dt, device_map={"": 0}, output_hidden_states=True)
    model.eval()
    tied = check_tied_embeddings(model)
    L = model.config.num_hidden_layers
    V = model.config.vocab_size
    print(f"[{args.model_key}] hf_id={args.hf_id} layers={L} vocab={V} tied={tied}", flush=True)

    script_masks = build_script_masks(tok, V, model.device)
    bin_ids = build_semantic_token_ids(tok, V)
    print(f"[{args.model_key}] script/bin tables built ({time.time()-t0:.0f}s)", flush=True)

    H = model.config.hidden_size
    n = len(sampled)
    hs_en = np.zeros((n, L + 1, H), dtype=np.float16)
    hs_ja = np.zeros((n, L + 1, H), dtype=np.float16)
    sm_en = {c: np.zeros((n, L + 1)) for c in SCRIPT_CLASSES}
    sm_ja = {c: np.zeros((n, L + 1)) for c in SCRIPT_CLASSES}
    sem_en = {b: np.zeros((n, L + 1)) for b in SEMANTIC_BINS}
    sem_ja = {b: np.zeros((n, L + 1)) for b in SEMANTIC_BINS}
    cats = []

    for i, pair in enumerate(sampled):
        cats.append(pair["category"])
        he, se, be = process_text(model, tok, pair[args.src_key], script_masks, bin_ids, args.final_norm)
        hj, sj, bj = process_text(model, tok, pair[args.tgt_key], script_masks, bin_ids, args.final_norm)
        hs_en[i], hs_ja[i] = he, hj
        for c in SCRIPT_CLASSES:
            sm_en[c][i], sm_ja[c][i] = se[c], sj[c]
        for b in SEMANTIC_BINS:
            sem_en[b][i], sem_ja[b][i] = be[b], bj[b]
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{n} ({time.time()-t0:.0f}s)", flush=True)

    # ---- 메트릭 ----
    layers = list(range(L + 1))
    cka = {l: linear_cka(hs_en[:, l, :].astype(np.float32), hs_ja[:, l, :].astype(np.float32)) for l in layers}
    ll_jsd = {}
    for l in layers:
        vals = [jsd_from_bins({b: sem_en[b][i, l] for b in SEMANTIC_BINS},
                              {b: sem_ja[b][i, l] for b in SEMANTIC_BINS}) for i in range(n)]
        ll_jsd[l] = float(np.nanmean(vals))
    smass_en = {c: {l: float(sm_en[c][:, l].mean()) for l in layers} for c in SCRIPT_CLASSES}
    smass_ja = {c: {l: float(sm_ja[c][:, l].mean()) for l in layers} for c in SCRIPT_CLASSES}
    peak = max(cka, key=cka.get)

    out_dir = Path(args.out_dir) / args.model_key
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "model_key": args.model_key, "hf_id": args.hf_id,
        "tied_embeddings": bool(tied), "num_layers": L, "vocab_size": V,
        "n_pairs": n, "src_key": args.src_key, "tgt_key": args.tgt_key,
        "final_norm": args.final_norm,
        "cka": {str(l): cka[l] for l in layers},
        "logit_lens_jsd": {str(l): ll_jsd[l] for l in layers},
        "script_mass_en": {c: {str(l): smass_en[c][l] for l in layers} for c in SCRIPT_CLASSES},
        "script_mass_tgt": {c: {str(l): smass_ja[c][l] for l in layers} for c in SCRIPT_CLASSES},
        "cka_peak": {"layer": int(peak), "value": float(cka[peak])},
        "categories": cats,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    if args.save_hidden:
        np.savez_compressed(out_dir / "hidden_states.npz", hs_en=hs_en, hs_ja=hs_ja,
                            categories=np.array(cats))
    print(f"[{args.model_key}] DONE peak_layer={peak} cka={cka[peak]:.3f} "
          f"jsd_late={ll_jsd[L]:.3f} ({time.time()-t0:.0f}s) -> {out_dir/'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
