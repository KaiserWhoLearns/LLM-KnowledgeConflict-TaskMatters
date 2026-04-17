#!/bin/bash
# Run free-form prediction + evaluation for a task.
#
# Configure via env vars or edit the defaults below:
#   MODEL_NAME     HuggingFace repo id or API model id
#   TASK_TYPE      One of: KFsummary, KFextract, PCK, CK, PK, RAG
#   DATA_VERSION   Dataset version suffix (default full_v2)
set -e

export base_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export data_dir="${base_dir}/data"

export model_name="${MODEL_NAME:-gpt-5.2}"
export task_type="${TASK_TYPE:-KFextract}"
export data_version="${DATA_VERSION:-full_v2}"

declare -A TASK_TYPE_PRETTY
TASK_TYPE_PRETTY["KFsummary"]="knowledge_free_summary"
TASK_TYPE_PRETTY["KFextract"]="knowledge_free_extract"
TASK_TYPE_PRETTY["PCK"]="parametriccontextual_knowledge"
TASK_TYPE_PRETTY["CK"]="contextual_knowledge"
TASK_TYPE_PRETTY["PK"]="parametric_knowledge"
TASK_TYPE_PRETTY["RAG"]="rag"

declare -A MODEL_NAME_TO_PRETTY
MODEL_NAME_TO_PRETTY["allenai/OLMo-2-1124-7B-Instruct"]="olmo2-7B"
MODEL_NAME_TO_PRETTY["allenai/OLMo-2-1124-13B-Instruct"]="olmo2-13B"
MODEL_NAME_TO_PRETTY["meta-llama/Llama-3.1-8B-Instruct"]="llama-3.1-8B-Instruct"
MODEL_NAME_TO_PRETTY["meta-llama/Llama-3.2-1B-Instruct"]="llama3.2-1B-Instruct"
MODEL_NAME_TO_PRETTY["meta-llama/Llama-3.2-3B-Instruct"]="llama3.2-3B-Instruct"
MODEL_NAME_TO_PRETTY["mistralai/Mistral-7B-Instruct-v0.3"]="mistral7B"
MODEL_NAME_TO_PRETTY["Qwen/Qwen2.5-7B-Instruct-1M"]="qwen7B-instruct"
MODEL_NAME_TO_PRETTY["Qwen/Qwen2.5-14B-Instruct"]="qwen2.5-14B-instruct"
MODEL_NAME_TO_PRETTY["deepseek-ai/DeepSeek-R1-Distill-Llama-8B"]="deepseek-llama8b"
MODEL_NAME_TO_PRETTY["gpt-5.2"]="gpt5.2"

pretty_name="${MODEL_NAME_TO_PRETTY[$model_name]}"

echo "Running ${pretty_name}-pred-${task_type}"
mkdir -p "${base_dir}/logs" "${base_dir}/output"
cd "$base_dir"

python model_runs/predict.py \
    --test_model_name "$pretty_name" \
    --data_version "$data_version" \
    --task_type "$task_type" \
    --data_path "${data_dir}/task_data/${pretty_name}_${TASK_TYPE_PRETTY[$task_type]}_${data_version}.jsonl"

python model_runs/evaluate.py \
    --test_model_name "$pretty_name" \
    --pred_path "${base_dir}/output/${pretty_name}_${task_type}_${data_version}_free.jsonl" \
    --task_type "$task_type"
