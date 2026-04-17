#!/bin/bash
# Aggregate evaluation metrics into per-model CSVs.
#
# Configure via env vars or edit the defaults below:
#   MODEL_NAME    Pretty model name (e.g. gpt5.2)
#   DATA_VERSION  Dataset version suffix (default full_v2)
set -e

export base_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export data_dir="${base_dir}/data"

export model_name="${MODEL_NAME:-gpt5.2}"
export data_version="${DATA_VERSION:-full_v2}"

cd "$base_dir"
mkdir -p "${base_dir}/results"

# Multiple choice results
python model_runs/aggregate_eval_results.py \
    --test_model_name "$model_name" \
    --data_version "$data_version" \
    --format mult \
    --save_path "${base_dir}/results/${model_name}_${data_version}_mult_perf.csv"

# Free generation results
python model_runs/aggregate_eval_results.py \
    --test_model_name "$model_name" \
    --data_version "$data_version" \
    --format free \
    --save_path "${base_dir}/results/${model_name}_${data_version}_free_perf.csv"
