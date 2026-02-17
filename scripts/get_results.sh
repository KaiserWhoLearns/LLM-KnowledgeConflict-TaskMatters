#!/bin/bash
export base_dir=/scratch4/mdredze1/hsun74/KnowledgeInstruct
export data_dir=/scratch4/mdredze1/hsun74/KnowledgeInstruct/data

export model_name="gpt5.2"
export data_version="full_v2"

cd $base_dir

# Multiple choice results
python model_runs/aggregate_eval_results.py \
    --test_model_name $model_name \
    --data_version $data_version \
    --format mult \
    --save_path ${base_dir}/results/${model_name}_${data_version}_mult_perf.csv

# Free generation results
python model_runs/aggregate_eval_results.py \
    --test_model_name $model_name \
    --data_version $data_version \
    --format free \
    --save_path ${base_dir}/results/${model_name}_${data_version}_free_perf.csv
