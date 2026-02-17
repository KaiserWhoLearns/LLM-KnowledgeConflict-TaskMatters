#!/bin/bash
export base_dir=/scratch4/mdredze1/hsun74/KnowledgeInstruct
export data_dir=/scratch4/mdredze1/hsun74/KnowledgeInstruct/data

export model_name="gpt-5.2"
export task_type="KFextract"
export data_version="full_v2"

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

# Determine if this is an API model (no GPU needed)
is_api_model=false
if [[ "$model_name" == gpt-* ]]; then
    is_api_model=true
fi

export exp_name="${MODEL_NAME_TO_PRETTY[$model_name]}-pred-${task_type}"
echo "Running $exp_name"

export BNB_CUDA_VERSION=118

mkdir -p ${base_dir}/logs

cd $base_dir

if [ "$is_api_model" = true ]; then
sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=$exp_name
#SBATCH --mail-user=hsun74@jhu.edu
#SBATCH --mail-type=FAIL,END
#SBATCH --partition=parallel
#SBATCH -A mdredze1
#SBATCH --gpus=0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=8G
#SBATCH --time=2-24:00:00 # Max runtime in DD-HH:MM:SS format.
#SBATCH --chdir=${BASE_DIR}
#SBATCH --export=all
#SBATCH --output=${base_dir}/logs/output_${exp_name}.log
#SBATCH --error=${base_dir}/logs/error_${exp_name}.log

module load anaconda3
conda activate /scratch4/mdredze1/hsun74/conda_env/kc
cd $base_dir

python model_runs/predict.py \
    --test_model_name ${MODEL_NAME_TO_PRETTY[$model_name]} \
    --data_version $data_version \
    --task_type $task_type \
    --data_path ${data_dir}/task_data/${MODEL_NAME_TO_PRETTY[$model_name]}_${TASK_TYPE_PRETTY[$task_type]}_${data_version}.jsonl

python model_runs/evaluate.py \
    --test_model_name ${MODEL_NAME_TO_PRETTY[$model_name]} \
    --pred_path ${base_dir}/output/${MODEL_NAME_TO_PRETTY[$model_name]}_${task_type}_${data_version}_free.jsonl \
    --task_type $task_type

# python model_runs/aggregate_eval_results.py
#     --test_model_name ${MODEL_NAME_TO_PRETTY[$model_name]}

EOT
else
sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=$exp_name
#SBATCH --mail-user=hsun74@jhu.edu
#SBATCH --mail-type=FAIL,END
#SBATCH --partition=a100
#SBATCH -A mdredze1_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=50G
#SBATCH --gpus=2
#SBATCH --time=1-00:00:00 # Max runtime in DD-HH:MM:SS format.
#SBATCH --chdir=${BASE_DIR}
#SBATCH --export=all
#SBATCH --output=${base_dir}/logs/output_${exp_name}.log
#SBATCH --error=${base_dir}/logs/error_${exp_name}.log

module load anaconda3
module load cuda/11.8.0
conda activate /scratch4/mdredze1/hsun74/conda_env/kc
# source "/home/hsun74/.bashrc"
cd $base_dir

python model_runs/predict.py \
    --test_model_name ${MODEL_NAME_TO_PRETTY[$model_name]} \
    --data_version $data_version \
    --task_type $task_type \
    --data_path ${data_dir}/task_data/${MODEL_NAME_TO_PRETTY[$model_name]}_${TASK_TYPE_PRETTY[$task_type]}_${data_version}.jsonl
   # --pilot_run \

python model_runs/evaluate.py \
    --test_model_name ${MODEL_NAME_TO_PRETTY[$model_name]} \
    --pred_path ${base_dir}/output/${MODEL_NAME_TO_PRETTY[$model_name]}_${task_type}_${data_version}_free.jsonl \
    --task_type $task_type

# python model_runs/aggregate_eval_results.py
#     --test_model_name ${MODEL_NAME_TO_PRETTY[$model_name]}

EOT
fi