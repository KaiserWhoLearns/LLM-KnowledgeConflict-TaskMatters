import os

ENV = {
    "data_dir": "/scratch4/mdredze1/hsun74/KnowledgeInstruct/data",
    "base_dir": "/scratch4/mdredze1/hsun74/KnowledgeInstruct"
}

def get_constant():
    for var_name in ENV:
        os.environ[var_name] = ENV[var_name]