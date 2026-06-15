#!/bin/bash
# KO bias (Llama-2, koen) + round-trip chrF (NLLB). Sequential download->use->delete (disk ~36G).
set -u
cd ~/stereotype-workshop
PY=~/miniconda3/envs/acl2026/bin/python
HUB=~/.cache/huggingface/hub
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_DISABLE_PROGRESS_BARS=1

$PY -m pip install -q sacrebleu 2>&1 | tail -1

echo "=== [llama2_ko] KO bias $(date +%H:%M:%S) ==="
$PY scripts/ko_bias.py --hf-id meta-llama/Llama-2-13b-hf --model-key llama2_ko \
  && rm -rf "$HUB/models--meta-llama--Llama-2-13b-hf" || { echo "ERR llama2_ko"; exit 1; }
df -h / | tail -1

echo "=== [koen] KO bias $(date +%H:%M:%S) ==="
$PY scripts/ko_bias.py --hf-id beomi/llama-2-koen-13b --model-key koen \
  && rm -rf "$HUB/models--beomi--llama-2-koen-13b" || { echo "ERR koen"; exit 1; }
df -h / | tail -1

echo "=== [chrf] back-translation $(date +%H:%M:%S) ==="
$PY scripts/backtranslate_chrf.py \
  && rm -rf "$HUB/models--facebook--nllb-200-distilled-600M" || { echo "ERR chrf"; exit 1; }

echo "ALL_DONE $(date +%H:%M:%S)"
