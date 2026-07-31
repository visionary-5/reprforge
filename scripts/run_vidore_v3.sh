#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 || $# -gt 7 ]]; then
  echo "usage: $0 DATASET MODE BASE_MODEL ADAPTER OUTPUT [CANDIDATE_K] [CACHE_CAPACITY]"
  exit 2
fi

dataset=$1
mode=$2
base_model=$3
adapter=$4
output=$5
candidate_k=${6:-20}
cache_capacity=${7:-0}

python -m vidore_benchmark.cli.main pipeline evaluate \
  --dataset-name "$dataset" \
  --module-path reprforge/vidore_pipeline.py \
  --class-name ReprForgeViDoRePipeline \
  --language english \
  --output-file "$output" \
  --pipeline-args "$(
    python -c 'import json,sys; print(json.dumps({
      "mode": sys.argv[1],
      "base_model": sys.argv[2],
      "adapter": sys.argv[3],
      "candidate_k": int(sys.argv[4]),
      "cache_capacity_items": int(sys.argv[5]),
    }))' "$mode" "$base_model" "$adapter" "$candidate_k" "$cache_capacity"
  )"
