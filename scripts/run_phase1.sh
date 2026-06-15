#!/bin/bash
# Phase 1 순차 추출 (디스크 빠듯 -> 모델마다 실행 후 가중치 삭제).
# Llama-2는 이미 처리됨 -> 먼저 캐시 삭제해 Swallow 공간 확보.
set -u
cd ~/stereotype-workshop
PY=~/miniconda3/envs/acl2026/bin/python
HUB=~/.cache/huggingface/hub
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_DISABLE_PROGRESS_BARS=1

echo "[phase1] freeing Llama-2 weights (results already saved)"
rm -rf "$HUB/models--meta-llama--Llama-2-13b-hf"
df -h / | tail -1

run() {  # key  hf_id  cache_dir
  local key="$1" hf="$2" cache="$3"
  echo "=== [$key] $hf $(date +%H:%M:%S) ==="
  $PY scripts/extract_metrics.py --hf-id "$hf" --model-key "$key" --save-hidden
  rc=$?
  if [ $rc -ne 0 ]; then echo "[phase1] ERROR $key rc=$rc"; exit $rc; fi
  rm -rf "$HUB/$cache"
  echo "=== [$key] done, weights removed $(date +%H:%M:%S) ==="
  df -h / | tail -1
}

run swallow "tokyotech-llm/Swallow-13b-hf" "models--tokyotech-llm--Swallow-13b-hf"
run llm_jp  "llm-jp/llm-jp-3-13b"          "models--llm-jp--llm-jp-3-13b"

echo "ALL_PHASE1_JA_DONE $(date +%H:%M:%S)"
