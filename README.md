# Translation Is Not Representation: English-Hub Routing in Cross-Lingual Bias Benchmarks

Paper for **StereACuLT** (Workshop on Stereotypes Across Cultures in Language Technologies). Paper source in [`paper/`](paper/) (`main.tex` → `main.pdf`); conference poster in [`paper/poster.pdf`](paper/poster.pdf).

## Overview

Cross-lingual bias benchmarks (JBBQ, KoBBQ) translate English bias probes and compare scores across languages, assuming the translated probe measures the same construct. Using 13B models matched on architecture and differing only in language-training regime, we find a **representation–behavior dissociation**:

1. **Geometric convergence masks English-hub routing.** High CKA between languages coexists with the model predicting English-script tokens in middle layers. A matched continual-adaptation pair (Llama-2 → Swallow), identical in architecture and initialization, attributes the drop in this hub routing to target-language adaptation (mid-layer English-script mass 0.77 → 0.56); balanced training minimizes it (0.19); the pattern replicates for Korean (Llama-2 → koen, 0.78 → 0.71). CKA does not reveal this.
2. **The bias asymmetry is language-specific.** English is more stereotype-biased than Japanese (P(stereotyped) higher by 0.13–0.14, all models, 95% CI excludes 0), but in Korean the gap is weak (+0.05 for the base model) and disappears after Korean adaptation, with Korean nearly as stereotype-leaning as English. English is not universally more biased than the translated language.
3. **Hub routing does not transplant bias.** Injecting English hub-layer states into target-language processing (single-layer and across the full hub band) leaves the stereotype preference unchanged vs. a random control, in both Japanese and Korean. The cross-lingual bias gap is genuine language-specific behavior, not an English-pivot artifact, even though the representations are not comparable.

We turn this into a four-step audit protocol for translated bias benchmarks.

## Models

| Role | HuggingFace id |
|------|----------------|
| English-centric base (shared anchor) | `meta-llama/Llama-2-13b-hf` |
| + Japanese continual (matched pair) | `tokyotech-llm/Swallow-13b-hf` |
| + Korean continual (matched pair) | `beomi/llama-2-koen-13b` |
| Balanced bilingual, from scratch | `llm-jp/llm-jp-3-13b` |

## Structure

```
paper/      LaTeX source: main.tex, main.bib, figures/, main.pdf, poster.pdf
src/        extraction + metric modules (CKA, logit lens, BBQ/JBBQ loading)
scripts/    experiment pipeline + analysis
configs/    model_config.yaml
data/       not shipped; see data/README.md for how to obtain BBQ/JBBQ/KoBBQ
results/    per-model metrics.json + patching JSONs
```

The benchmark data is **not redistributed here** (license + size); see
[`data/README.md`](data/README.md) for sources and the expected layout.

## Reproduce

```bash
pip install -r requirements.txt

# First obtain the datasets (not shipped): see data/README.md.
# Then build the Korean EN-KO pairs from KoBBQ:
python scripts/build_kobbq_pairs.py

# Representation metrics (CKA, binned + multi-anchor logit lens), one model at a time:
bash scripts/run_phase1.sh     # Llama-2, Swallow, LLM-jp  (English-Japanese)
bash scripts/run_phase5.sh     # Llama-2, koen             (English-Korean replication)

# Causal hub-patching (bias-transplant test):
bash scripts/run_phase2.sh

# Analysis tables + figures:
python scripts/compare_phase1.py   # CKA / latin-mass comparison
python scripts/compare_phase2.py   # patching transplant + bootstrap CIs
python scripts/baseline_bias.py    # English vs. target stereotype asymmetry
python scripts/figure_gen.py       # paper figures
```

Models are processed one at a time (download → extract → delete weights) so the pipeline fits a single 48 GB GPU.

## License

- **Code** (this repository): [MIT](LICENSE).
- **Data** (BBQ, JBBQ, KoBBQ): not included here; each is owned by its original
  authors under its own license/terms. See [`data/README.md`](data/README.md).
- **Paper** (PDF and LaTeX sources under `paper/`): research output of the
  authors; please cite the paper rather than reusing its text or figures.
