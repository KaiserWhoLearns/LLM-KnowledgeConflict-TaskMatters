#!/bin/bash
# Run the context-editing pipeline: classify+edit contexts, filter invalid
# instances, then generate task variants (free-form and multiple-choice).
#
# Configure via env vars or edit the defaults below:
#   MODEL_NAME       HuggingFace repo id or API model id
#   DATA_VERSION     Dataset version suffix (default full_v2)
#   SAMPLE_FRACTION  Fraction of data to process in clas_edit_context.py
set -e

export base_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export data_dir="${base_dir}/data"

export model_name="${MODEL_NAME:-gpt-5.2}"
data_version="${DATA_VERSION:-full_v2}"
sample_fraction="${SAMPLE_FRACTION:-0.1}"

declare -A MODEL_NAME_TO_PRETTY
MODEL_NAME_TO_PRETTY["google/gemma-3-4b-it"]="gemma3-4b"
MODEL_NAME_TO_PRETTY["allenai/OLMo-2-1124-7B-Instruct"]="olmo2-7B"
MODEL_NAME_TO_PRETTY["allenai/OLMo-2-1124-13B-Instruct"]="olmo2-13B"
MODEL_NAME_TO_PRETTY["meta-llama/Llama-3.1-8B-Instruct"]="llama-3.1-8B-Instruct"
MODEL_NAME_TO_PRETTY["meta-llama/Llama-3.2-1B-Instruct"]="llama3.2-1B-Instruct"
MODEL_NAME_TO_PRETTY["meta-llama/Llama-3.2-3B-Instruct"]="llama3.2-3B-Instruct"
MODEL_NAME_TO_PRETTY["mistralai/Mistral-7B-Instruct-v0.3"]="mistral7B"
MODEL_NAME_TO_PRETTY["Qwen/Qwen2.5-14B-Instruct"]="qwen2.5-14B-instruct"
MODEL_NAME_TO_PRETTY["Qwen/Qwen2.5-7B-Instruct-1M"]="qwen7B-instruct"
MODEL_NAME_TO_PRETTY["deepseek-ai/DeepSeek-R1-Distill-Llama-8B"]="deepseek-llama8b"
MODEL_NAME_TO_PRETTY["gpt-5.2"]="gpt5.2"

pretty_name="${MODEL_NAME_TO_PRETTY[$model_name]}"

echo "Running ${pretty_name}-context_edit"
mkdir -p "${base_dir}/logs"
cd "$base_dir"

# Saves to: data/final_data/{model_name}_{data_version}.jsonl
python data_creation/clas_edit_context.py \
    --test_model_name "$pretty_name" \
    --data_version "$data_version" \
    --sample_fraction "$sample_fraction"

# Saves to: data/final_data_filtered/{model_name}_{data_version}.jsonl
python data_creation/remove_invalid_instances.py \
    --test_model_name "$pretty_name" \
    --data_version "$data_version"

# Saves to: data/task_data/{model_name}_knowledge_free_extract_{data_version}.jsonl (and other task variants)
python data_creation/add_instruction.py \
    --test_model_name "$pretty_name" \
    --data_version "$data_version"

# Saves to: data/choice_task/{model_name}_{task_type}_{data_version}.jsonl
python data_creation/add_instruction_choice.py \
    --test_model_name "$pretty_name" \
    --data_version "$data_version"
