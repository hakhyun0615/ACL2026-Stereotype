#!/usr/bin/env python3
"""Surface-form translation fidelity via round-trip chrF.

Step 4 of the audit protocol recommends a surface-form metric (chrF), not a
representation-space cosine. We demonstrate it on our own data: back-translate
each target-language context to English with NLLB-200, then chrF against the
English source. KO is computed on exactly the 261 bias items (same sample/order
as ko_bias.py) so fidelity can be joined to bias per item; JA is a distribution.
"""
import json, sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sacrebleu

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src.data_prep.load_bbq_jbbq import sample_paired_templates  # noqa: E402
from ko_bias import en_roles, ko_roles  # identical KO filtering -> aligned with ko_bias  # noqa: E402

NLLB = "facebook/nllb-200-distilled-600M"
DEV = "cuda"


def load_nllb():
    tok = AutoTokenizer.from_pretrained(NLLB)
    model = AutoModelForSeq2SeqLM.from_pretrained(NLLB, torch_dtype=torch.float16).to(DEV).eval()
    bos = tok.convert_tokens_to_ids("eng_Latn")
    return tok, model, bos


@torch.no_grad()
def translate(tok, model, bos, texts, src_lang, bs=16):
    tok.src_lang = src_lang
    out = []
    for i in range(0, len(texts), bs):
        chunk = texts[i:i + bs]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=400).to(DEV)
        gen = model.generate(**enc, forced_bos_token_id=bos, max_length=400, num_beams=1)
        out.extend(tok.batch_decode(gen, skip_special_tokens=True))
    return out


def chrf(hyp, ref):
    return sacrebleu.sentence_chrf(hyp, [ref]).score


def main():
    tok, model, bos = load_nllb()

    # --- KO: aligned with the 261 ko_bias items ---
    ko_data = json.load(open(ROOT / "data/processed/paired_ko.json"))
    ko_items = [p for p in sample_paired_templates(ko_data, max_per_context=3)
                if en_roles(p) is not None and ko_roles(p)[1] is not None]
    ko_bt = translate(tok, model, bos, [p["context_ko"] for p in ko_items], "kor_Hang")
    ko_recs = [dict(category=p["category"], chrf=chrf(bt, p["context_en"]))
               for p, bt in zip(ko_items, ko_bt)]
    json.dump({"n": len(ko_recs), "records": ko_recs},
              open(ROOT / "results/chrf_ko.json", "w"), indent=2)
    ko_scores = [r["chrf"] for r in ko_recs]
    print(f"[KO] n={len(ko_scores)} chrF mean={sum(ko_scores)/len(ko_scores):.1f} "
          f"min={min(ko_scores):.1f} max={max(ko_scores):.1f}", flush=True)

    # --- JA: fidelity distribution ---
    ja_data = json.load(open(ROOT / "data/processed/paired_templates.json"))
    ja_pairs = sample_paired_templates(ja_data, max_per_context=3)[:300]
    ja_bt = translate(tok, model, bos, [p["context_ja"] for p in ja_pairs], "jpn_Jpan")
    ja_recs = [dict(category=p.get("category"), chrf=chrf(bt, p["context_en"]))
               for p, bt in zip(ja_pairs, ja_bt)]
    ja_scores = sorted(r["chrf"] for r in ja_recs)
    summary = dict(n=len(ja_scores), mean=sum(ja_scores) / len(ja_scores),
                   median=ja_scores[len(ja_scores) // 2], p25=ja_scores[len(ja_scores) // 4])
    json.dump({"summary": summary, "records": ja_recs},
              open(ROOT / "results/chrf_ja.json", "w"), indent=2)
    print(f"[JA] {summary}", flush=True)


if __name__ == "__main__":
    main()
