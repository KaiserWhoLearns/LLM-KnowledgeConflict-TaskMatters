import os

ENV = {
    "data_dir": "/scratch4/mdredze1/hsun74/KnowledgeInstruct/data",
    "base_dir": "/scratch4/mdredze1/hsun74/KnowledgeInstruct",
    "HF_HOME": "/scratch4/mdredze1/hsun74/huggingface_cache",
    "TRANSFORMERS_CACHE": "/scratch4/mdredze1/hsun74/huggingface_cache/transformers",
    "HF_DATASETS_CACHE": "/scratch4/mdredze1/hsun74/huggingface_cache/datasets"
}

def get_constant():
    for var_name in ENV:
        os.environ[var_name] = ENV[var_name]