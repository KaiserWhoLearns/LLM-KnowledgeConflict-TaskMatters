#!/bin/bash
export base_dir=/scratch4/mdredze1/hsun74/KnowledgeInstruct
export data_dir=/scratch4/mdredze1/hsun74/KnowledgeInstruct/data

export model_name="Qwen/Qwen2.5-7B-Instruct-1M"
export task_type="PK" # Choose from: KFsummary, KFextract, PCK, CK, PK, RAG
export data_version="full_v2"
export length_ablation=true # Set to true for length ablation experiments

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

# Determine data path and save path based on length_ablation flag
if [ "$length_ablation" = true ]; then
    export data_path="${data_dir}/length_ablation_task/${MODEL_NAME_TO_PRETTY[$model_name]}_${TASK_TYPE_PRETTY[$task_type]}_${data_version}.jsonl"
    export save_path="${base_dir}/output/${MODEL_NAME_TO_PRETTY[$model_name]}_${task_type}_${data_version}_choice_len_ablation.jsonl"
else
    export data_path=""  # Use default path in predict.py
    export save_path="${base_dir}/output/${MODEL_NAME_TO_PRETTY[$model_name]}_${task_type}_${data_version}_choice.jsonl"
fi

export exp_name="${MODEL_NAME_TO_PRETTY[$model_name]}-pred-${task_type}"
echo "Running $exp_name"

export BNB_CUDA_VERSION=118

mkdir -p ${base_dir}/logs

cd $base_dir
sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=$exp_name
#SBATCH --mail-user=hsun74@jhu.edu
#SBATCH --mail-type=FAIL,END
#SBATCH -A mdredze80_gpu
#SBATCH --partition=ica100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=50G
#SBATCH --gpus=1
#SBATCH --time=2-15:00:00 # Max runtime in DD-HH:MM:SS format.
#SBATCH --chdir=${BASE_DIR}
#SBATCH --export=all
#SBATCH --output=${base_dir}/logs/output_${exp_name}.log
#SBATCH --error=${base_dir}/logs/error_${exp_name}.log

module load anaconda3
module load cuda/11.8.0
conda activate /scratch4/mdredze1/hsun74/conda_env/kc
# source "/home/hsun74/.bashrc"
cd $base_dir

# Run prediction with optional data_path and save_dir
if [ "$length_ablation" = true ]; then
    python model_runs/predict.py \
        --test_model_name ${MODEL_NAME_TO_PRETTY[$model_name]} \
        --data_version $data_version \
        --task_type $task_type \
        --mult_choice \
        --data_path "$data_path" \
        --save_dir "$save_path"
else
    python model_runs/predict.py \
        --test_model_name ${MODEL_NAME_TO_PRETTY[$model_name]} \
        --data_version $data_version \
        --task_type $task_type \
        --mult_choice
fi

python model_runs/evaluate_choice.py \
    --test_model_name ${MODEL_NAME_TO_PRETTY[$model_name]} \
    --pred_path "$save_path" \
    --task_type $task_type

# python model_runs/aggregate_eval_results.py
#     --test_model_name ${MODEL_NAME_TO_PRETTY[$model_name]}

EOT