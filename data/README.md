# Data

The benchmark datasets are **not redistributed in this repository** (each has its
own license, and the raw files are large). This document explains where to get
them and where to place them so the scripts in this repo can find them.

After obtaining the files, the expected layout is:

```
data/
  bbq/                 <category>.jsonl    (English BBQ)
  jbbq/                <category>.jsonl    (Japanese JBBQ)
  processed/
    paired_templates.json   built EN-JA pairs
    paired_ko.json          built EN-KO pairs (see below)
```

The five shared categories used in the paper are:
`Age`, `Disability_status`, `Gender_identity`, `Physical_appearance`,
`Sexual_orientation`.

## Sources

| Dataset | Owner / source | Notes |
|---------|----------------|-------|
| **BBQ** (English) | NYU MLL, <https://github.com/nyu-mll/BBQ> | Public; redistribution allowed under its license (verify before reuse). Place per-category `*.jsonl` under `data/bbq/`. |
| **JBBQ** (Japanese) | ynklab, <https://github.com/ynklab/JBBQ_data> | **Access-restricted**: JBBQ is released under usage terms that may require agreeing to a license / requesting access. Obtain it directly from the authors; do not redistribute. Place per-category `*.jsonl` under `data/jbbq/`. |
| **KoBBQ** (Korean) | NAVER AI Lab, <https://github.com/naver-ai/KoBBQ> | Public; check its license. Used to build the EN-KO pairs (see below). |

> Always confirm each dataset's current license/terms on its official page. The
> table is a pointer, not a license grant. This repo intentionally ships none of
> the raw data.

## Building the paired files

The EN-JA pairs (`paired_templates.json`) are produced from `data/bbq/` and
`data/jbbq/` by the loaders in `src/data_prep/`.

The EN-KO pairs are built from KoBBQ with:

```bash
python scripts/build_kobbq_pairs.py
```

This restricts KoBBQ to its **Simply-Transferred** templates (clean English BBQ
counterparts), the five shared categories, and the ambiguous condition, then
aligns each Korean item to its English BBQ source via KoBBQ's `bbq_id`. Output is
written to `data/processed/paired_ko.json`.

## Citation

If you use these datasets, cite the original papers (BBQ: Parrish et al. 2022;
JBBQ: Yanaka et al. 2025; KoBBQ: Jin et al. 2024). Full references are in
`paper/main.bib`.
