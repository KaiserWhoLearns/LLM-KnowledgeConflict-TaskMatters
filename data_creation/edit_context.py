import os
import sys
import pdb
sys.path.append(os.getcwd())
import json
import time
import requests
import openai
from openai import OpenAI
from tqdm import tqdm
from datasets import load_from_disk
from dotenv import load_dotenv

from utils.constant import get_constant
load_dotenv()

# Load API keys from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
client = OpenAI(api_key=OPENAI_API_KEY)

# Constants
EDITOR_MODEL_NAME = "gpt-4o"


def create_edit_prompts(dataset, model_name, context_type):
    """
    Dataset = an output from "get_parametric_knowledge.py"
    context_type = HPC, LPC, High Plausibiliy Contradiction without Explanation, Low plausibility Contradiction
    Edit the context
    """
    # Context type: No contradiction (no need to edit) - NC, Contradiction with Explanation (Existing) - HPCE, 
    # High Plausibiliy Contradiction without Explanation - HPC, Low plausibility Contradiction - LPC
    inputs = []
    input_context = []
    # Remove the ones that the model does not have parametirc knowledge
    dataset = dataset.filter(lambda example: example[model_name] != 0)
    # TODO: Test
    dataset = dataset.select([i for i in range(10)])

    for instance in tqdm(dataset):
        answer = instance["answer1"] if instance[model_name] == 1 else instance["answer2"]
        alt_answer = instance["answer1"] if instance[model_name] == 2 else instance["answer2"]
        alt_context = instance["context1"] if instance[model_name] == 2 else instance["context2"]
        context = instance["context1"] if instance[model_name] == 1 else instance["context2"]

        if context_type == "HPC":
            # High Plausibiliy Contradiction without Explanation
            inputs.append(f"You are a smart editor that remove the explanation in the given passage, such that the answer to the question {instance['question']} is '{answer}'. \n You should only output the edited passage.\n")
            input_context.append(context)
        elif context_type == "LPC":
            # Low plausibility Contradiction
            inputs.append(f"You are a smart editor that creates inplausible texts. Your job is to edit the given evidence to the question {instance['question']}. You should change the content of the given passage, remove any explanation given in the passage, and make the passage as inplausible as possible such that the answer of the given passage become {alt_answer}. Inplausible passages include passages that disobey real-world knowledge or violets logical constraints. You should only output the edited passage.\n")
            input_context.append(alt_context)
        else:
            raise Exception("Unsupported context type")

    # Formats input data into JSONL format required by OpenAI's batch API.
    with open(os.path.join(os.environ["data_dir"], "temp", f"{model_name}_edit_input.jsonl"), "w") as f:
        i = 0
        for prompt, context in zip(inputs, input_context):
            json.dump({"custom_id": f"request-{i}", "method": "POST", "url": "/v1/chat/completions", "body": {"model": EDITOR_MODEL_NAME, "messages": [{"role": "system", "content": prompt},{"role": "user", "content": context}],"max_tokens": 10000}}, f)
            i += 1
            f.write("\n")
    print(f'Formatted dataset saved to {os.path.join(os.environ["data_dir"], "temp", f"{model_name}_edit_input.jsonl")}')

def submit_batch_job(input_file_path):
    """
    Submits the batch job to OpenAI API.
    """
    
    print("Uploading intput file...")
    batch_input_file = client.files.create(
        file=open(input_file_path, "rb"),
        purpose="batch"
    )

    print(batch_input_file)
    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "Context Edit Job"
        }
    )
    return batch.id

def check_batch_status(batch_id):
    """
    Checks the status of a batch job.
    """
    while True:
        batch = client.batches.retrieve(batch_id)

        print(f"Batch job status: {batch.status}...")
        if batch.status in ["completed", "failed", "cancelled", "expired"]:
            print(f"batchID = {batch_id}")
            return batch
        time.sleep(60)  # Wait before checking again

def download_results(batch, output_file_path):
    """
    Downloads the results of a batch job.
    """
    results = client.files.content(batch.output_file_id)
    # The output context is automatically a jsonl file
    with open(output_file_path, "w") as f:
        f.write(results.text)
    

if __name__ == "__main__":
    get_constant()
    # pdb.set_trace()
    model_name = "llama3.2-3B-Instruct"
    context_type = "HPC"
    dataset = load_from_disk(os.path.join(os.environ["data_dir"], "model_knowledge", model_name))

    create_edit_prompts(dataset=dataset, model_name=model_name, context_type=context_type)

    os.makedirs(os.path.join(os.environ["data_dir"], "intermediate_processing", context_type), exist_ok=True)
    output_file_path = os.path.join(os.environ["data_dir"], "intermediate_processing", context_type, f"{model_name}.jsonl")
    # Step 2: Submit batch job
    batch_id = submit_batch_job(os.path.join(os.environ["data_dir"], "temp", f"{model_name}_edit_input.jsonl"))
    if batch_id:
        # Step 3: Monitor job status
        batch = check_batch_status(batch_id)

        if batch.status == "completed":
            # Step 4: Download results
            download_results(batch, output_file_path)
        else:
            print(f"Batch job did not complete successfully. Final status: {batch.status}")

