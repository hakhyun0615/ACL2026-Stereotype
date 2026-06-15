#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_kobbq_pairs.py

EN-KO paired dataset builder for the cross-lingual bias-benchmark study.
Mirrors the existing EN-JA paired file (paired_templates.json) but for Korean (KoBBQ).

Restricted to:
  - label_annotation == "ST"  (Simply-Transferred templates: clean English BBQ counterparts)
  - the 5 categories shared with the EN-JA file:
      Age, Disability_status, Gender_identity, Physical_appearance, Sexual_orientation
  - the AMBIGUOUS context condition (mirrors EN-JA, whose context_en/context_ja are ambiguous-only)

ALIGNMENT (KoBBQ -> English BBQ), established empirically (see report):
  1) Parse KoBBQ HF `sample_id` (e.g. "age-001a-001-amb-bsd")
       -> (category, template_id=001, version=a, instance, cond=amb/dis, bias=bsd/cnt)
     The template_id matches `ID` in KoBBQ_templates.tsv (100% join, 0 misses).
  2) The HF `bbq_id` column IS the original English BBQ pointer; it equals the
     `BBQ_id` column in KoBBQ_templates.tsv for that (Category, ID, version) (100% match),
     and it equals the English BBQ `question_index` within the same category.
       -> English BBQ context = a canonical ambig+neg row with
          (category == bbq_category, question_index == int(bbq_id)).
  Both relationships verified at 11816/11816 rows with zero exceptions.

Output: data/processed/paired_ko.json  (JSON list).
"""

import os
import re
import csv
import ast
import json
import argparse
import subprocess
from collections import defaultdict

SHARED = ["Age", "Disability_status", "Gender_identity",
          "Physical_appearance", "Sexual_orientation"]

SID_PAT = re.compile(
    r"^(?P<cat>.+)-(?P<tid>\d+)(?P<ver>[a-z])-(?P<inst>\d+)-(?P<cond>amb|dis)-(?P<bias>bsd|cnt)$"
)


def load_en_bbq(bbq_dir):
    """category -> list of EN BBQ rows."""
    en = {}
    for cat in SHARED:
        rows = []
        with open(os.path.join(bbq_dir, "%s.jsonl" % cat), encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
        en[cat] = rows
    return en


def canonical_en_rows(en):
    """
    (category, question_index) -> canonical EN ambig+neg row.
    Canonical = lowest example_id among ambiguous, negative-polarity rows
    (deterministic, one representative context per template), mirroring how the
    EN-JA file keeps a single example_id per template.
    Falls back to any ambiguous row if no neg row exists.
    """
    out = {}
    for cat, rows in en.items():
        buckets = defaultdict(list)
        for r in rows:
            if r["context_condition"] == "ambig":
                buckets[r["question_index"]].append(r)
        for qidx, rs in buckets.items():
            neg = [r for r in rs if r["question_polarity"] == "neg"]
            pool = neg if neg else rs
            pool.sort(key=lambda r: r["example_id"])
            out[(cat, qidx)] = pool[0]
    return out


def load_templates(tsv_path):
    """(Category, ID, version) -> template row from KoBBQ_templates.tsv."""
    tmap = {}
    with open(tsv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            tmap[(r["Category"], r["ID"], r["version"])] = r
    return tmap


def build(bbq_dir, tsv_path, out_path, conditions):
    from datasets import load_dataset

    en = load_en_bbq(bbq_dir)
    en_canon = canonical_en_rows(en)
    tmap = load_templates(tsv_path)
    ds = load_dataset("naver-ai/kobbq")["test"]

    items = []
    unaligned = []
    n_total = 0

    for i in range(len(ds)):
        if ds["label_annotation"][i] != "ST":
            continue
        cat = ds["bbq_category"][i]
        if cat not in SHARED:
            continue
        sid = ds["sample_id"][i]
        m = SID_PAT.match(sid)
        if not m:
            unaligned.append((sid, "sample_id did not parse"))
            continue
        cond = m.group("cond")
        bias = m.group("bias")
        if cond not in conditions:          # keep only requested conditions (default: amb)
            continue
        n_total += 1

        tid = str(int(m.group("tid")))      # "001" -> "1"
        ver = m.group("ver")

        # English pointer: HF bbq_id == EN question_index
        bbq_id = ds["bbq_id"][i]
        qidx = int(bbq_id)
        en_key = (cat, qidx)
        en_row = en_canon.get(en_key)
        if en_row is None:
            unaligned.append((sid, "no EN row for (%s, qidx=%d)" % (cat, qidx)))
            continue

        tmpl = tmap.get((cat, tid, ver))    # optional enrichment / cross-check

        # parse KoBBQ choices "['a','b','c']" -> list
        try:
            choices = ast.literal_eval(ds["choices"][i])
        except Exception:
            choices = ds["choices"][i]

        item = {
            "category": cat,                                   # one of the 5 shared
            "context_en": en_row["context"],                  # canonical EN ambig+neg context
            "context_ko": ds["context"][i],                   # KoBBQ Korean context

            # identifiers / alignment provenance
            "bbq_id": bbq_id,                                  # == EN question_index
            "kobbq_sample_id": sid,
            "kobbq_template_id": tid,                          # ID in KoBBQ_templates.tsv
            "kobbq_version": ver,                              # a / b
            "kobbq_context_condition": cond,                  # amb / dis
            "kobbq_question_bias": bias,                       # bsd (biased) / cnt (counter-biased)
            "label_annotation": ds["label_annotation"][i],    # ST
            "en_example_id": en_row["example_id"],
            "en_question_index": en_row["question_index"],

            # --- Korean bias fields (KoBBQ) ---
            "choices": choices,                               # parsed list
            "biased_answer": ds["biased_answer"][i],          # stereotyped choice text (KO)
            "answer": ds["answer"][i],                        # correct/unknown choice text (KO)
            "question_ko": ds["question"][i],

            # --- English bias fields (BBQ), from the aligned canonical row ---
            "question_en": en_row["question"],
            "ans0": en_row["ans0"],
            "ans1": en_row["ans1"],
            "ans2": en_row["ans2"],
            "answer_info": en_row["answer_info"],
            "stereotyped_groups": en_row["additional_metadata"]["stereotyped_groups"],
            "label": en_row.get("answer_label", en_row.get("label")),  # correct-answer index
            "en_target_label": en_row.get("target_label"),             # stereotyped-answer index
            "en_question_polarity": en_row["question_polarity"],
            "en_source": en_row["additional_metadata"].get("source", ""),
        }
        # cross-check field from template TSV (English-translated KoBBQ context)
        if tmpl is not None:
            item["kobbq_context_translated"] = tmpl.get("Ambiguous_context_translated", "")
            item["kobbq_bbq_id_tsv"] = tmpl.get("BBQ_id", "")

        items.append(item)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    return items, unaligned, n_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbq_dir", default=os.path.expanduser("~/stereotype-workshop/data/bbq"))
    ap.add_argument("--tsv", default="/tmp/KoBBQ/data/KoBBQ_templates.tsv",
                    help="KoBBQ_templates.tsv from https://github.com/naver-ai/KoBBQ")
    ap.add_argument("--out", default=os.path.expanduser("~/stereotype-workshop/data/processed/paired_ko.json"))
    ap.add_argument("--conditions", default="amb",
                    help="comma list of KoBBQ context conditions to keep (amb,dis). Default amb.")
    args = ap.parse_args()

    conditions = set(c.strip() for c in args.conditions.split(",") if c.strip())
    items, unaligned, n_total = build(args.bbq_dir, args.tsv, args.out, conditions)

    # ---- report ----
    from collections import Counter
    print("=" * 60)
    print("BUILD COMPLETE ->", args.out)
    print("conditions kept:", sorted(conditions))
    print("KoBBQ ST+shared rows in scope:", n_total)
    print("items written:", len(items))
    print("unaligned:", len(unaligned))
    per_cat = Counter(it["category"] for it in items)
    print("per-category items:", dict(per_cat))
    tmpl_per_cat = defaultdict(set)
    for it in items:
        tmpl_per_cat[it["category"]].add(it["bbq_id"])
    print("distinct templates (bbq_id) per category:",
          {c: len(s) for c, s in tmpl_per_cat.items()})
    print("total distinct (category,bbq_id) templates:",
          len(set((it["category"], it["bbq_id"]) for it in items)))
    if unaligned:
        print("UNALIGNED examples:", unaligned[:10])


if __name__ == "__main__":
    main()
