#!/bin/bash
# Phase 5: KO directional replication. EN-KO pairs(paired_ko.json)로 Llama-2 vs koen 추출+메트릭.
# matched 쌍: Llama-2(영어 base) vs koen(Llama-2 + KO continual). JA의 Llama-2 vs Swallow를 KO에서 복제.
set -u
cd ~/stereotype-workshop
PY=~/miniconda3/envs/acl2026/bin/python
HUB=~/.cache/huggingface/hub
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_DISABLE_PROGRESS_BARS=1
PAIRS=data/processed/paired_ko.json

run(){ local key="$1" hf="$2" cache="$3"
  echo "=== [$key] $hf $(date +%H:%M:%S) ==="
  $PY scripts/extract_metrics.py --hf-id "$hf" --model-key "$key" --pairs $PAIRS \
      --src-key context_en --tgt-key context_ko --save-hidden
  if [ $? -ne 0 ]; then echo "[phase5] ERROR $key"; exit 1; fi
  rm -rf "$HUB/$cache"
  echo "=== [$key] done $(date +%H:%M:%S) ==="; df -h / | tail -1
}

run llama2_ko "meta-llama/Llama-2-13b-hf"  "models--meta-llama--Llama-2-13b-hf"
run koen      "beomi/llama-2-koen-13b"      "models--beomi--llama-2-koen-13b"
echo "ALL_PHASE5_DONE $(date +%H:%M:%S)"
