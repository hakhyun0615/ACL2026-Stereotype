#!/bin/bash
# Phase 2 순차 허브 패칭 (모델 재다운로드 -> 패칭 -> 가중치 삭제). hub_layer는 metrics.json의 CKA peak 사용.
set -u
cd ~/stereotype-workshop
PY=~/miniconda3/envs/acl2026/bin/python
HUB=~/.cache/huggingface/hub
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_DISABLE_PROGRESS_BARS=1
N=300

run(){ local key="$1" hf="$2" cache="$3"
  echo "=== [$key] $hf $(date +%H:%M:%S) ==="
  $PY scripts/hub_patching.py --hf-id "$hf" --model-key "$key" --n-items $N
  if [ $? -ne 0 ]; then echo "[phase2] ERROR $key"; exit 1; fi
  rm -rf "$HUB/$cache"
  echo "=== [$key] done, weights removed $(date +%H:%M:%S) ==="; df -h / | tail -1
}

run llama2  "meta-llama/Llama-2-13b-hf"     "models--meta-llama--Llama-2-13b-hf"
run swallow "tokyotech-llm/Swallow-13b-hf"  "models--tokyotech-llm--Swallow-13b-hf"
run llm_jp  "llm-jp/llm-jp-3-13b"           "models--llm-jp--llm-jp-3-13b"
echo "ALL_PHASE2_DONE $(date +%H:%M:%S)"
