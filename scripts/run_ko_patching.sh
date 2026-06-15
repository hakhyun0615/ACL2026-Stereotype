#!/bin/bash
# KO hub patching (llama2_ko, koen). Sequential download->patch->delete (disk ~36G).
set -u
cd ~/stereotype-workshop
PY=~/miniconda3/envs/acl2026/bin/python
HUB=~/.cache/huggingface/hub
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_DISABLE_PROGRESS_BARS=1

echo "=== [llama2_ko] patching $(date +%H:%M:%S) ==="
$PY scripts/ko_patching.py --hf-id meta-llama/Llama-2-13b-hf --model-key llama2_ko \
  && rm -rf "$HUB/models--meta-llama--Llama-2-13b-hf" || { echo "ERR llama2_ko"; exit 1; }
df -h / | tail -1

echo "=== [koen] patching $(date +%H:%M:%S) ==="
$PY scripts/ko_patching.py --hf-id beomi/llama-2-koen-13b --model-key koen \
  && rm -rf "$HUB/models--beomi--llama-2-koen-13b" || { echo "ERR koen"; exit 1; }

echo "ALL_DONE $(date +%H:%M:%S)"
