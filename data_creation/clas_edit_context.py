import os
import sys
import re
import pdb
sys.path.append(os.getcwd())
import json
import time
import requests
import openai
from openai import OpenAI
from tqdm import tqdm
from datasets import load_from_disk, Dataset
from dotenv import load_dotenv

from utils.constant import get_constant
load_dotenv()

# Load API keys from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
client = OpenAI(api_key=OPENAI_API_KEY)

# Constants
EDITOR_MODEL_NAME = "gpt-4o"

def classify_context(dataset):
    """
    Dataset = an output from "get_parametric_knowledge.py"
    Classify the context into NC, HPC, HPCE, LPC
    We already know what is NC. Existing evidence cannot be LPC. Thus, we just need to put the conflicting context into HPC v.s. HPCE
    """
    dataset = dataset.filter(lambda example: example[model_name] != 0)
    # TODO: Test
    dataset = dataset.select([i for i in range(10)])
    
    classified_data = []
    for instance in tqdm(dataset):
        model_answer = instance["answer1"] if instance[model_name] == 1 else instance["answer2"]
        model_context = instance["context1"] if instance[model_name] == 1 else instance["context2"]

        alt_answer = instance["answer1"] if instance[model_name] == 2 else instance["answer2"]
        alt_context = instance["context1"] if instance[model_name] == 2 else instance["context2"]

        # Determine if it is HPC or HPCE
        # Decide whether there include enough explanation (rate 1-5)
        prompt = "You are a smart reader, your job is to rate from 1 to 5 whether the given text has enough **explanation** to answer the given question. The lower the score, the less explanation the given passage contains to answer the question. Your job is to determine the logical coherence of the given passage, and you should not be affected by the factual presence of the given passage. You should only output the integer scores. Below are some examples:\n"
        "Passage 1: Apomorphine is said to be main psychoactive compound present. \n" + \
        "Question 1: Which of the following are present in Nymphaea nouchali var. caerulea: apomorphine, aporphine, or neither? \n" + \
        "Rating 1: 1 \n" + \
        "Passage 2: Past research claim that aporphine are present in Nymphaea nouchali var. However, a recent study provide solid evidence that Apomorphine is the main psychoactive compound present. \n" + \
        "Question 2: Which of the following are present in Nymphaea nouchali var. caerulea: apomorphine, aporphine, or neither? \n" + \
        "Rating 2: 5 \n"
        content = f"Passage: {alt_context}\nQuestion: {instance['question']}\nRating:"

        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "developer", "content": prompt},
                {
                    "role": "user",
                    "content": content
                }
            ]
        )
        response = completion.choices[0].message.content
        # Add to dictionary of processed
        match = re.search(r'\d+', response)
        if match:
            rating = int(match.group())
            print(response)
            # If Rating < 3, HPC, elif rating >= 3: HPCE
            if rating < 3:
                curr_key = "HPC"
                spare_key = "HPCE"
            else:
                curr_key = "HPCE"
                spare_key = "HPC"
            classified_data.append({
                "question": instance["question"],
                "NC_context": model_context,
                "NC_answer": model_answer,
                "alt_answer": alt_answer,
                "alt_context": alt_context,
                f"{curr_key}_answer": alt_answer,
                f"{curr_key}_context": alt_context,
                f"{spare_key}_answer": "",
                f"{spare_key}_context": "",
            })
        else:
            raise Exception(f"The output does not contain a rating. Model Output: {response}")
    return Dataset.from_list(classified_data)
        

def create_edit_prompts(dataset, model_name, context_type):
    """
    Dataset = an output from "get_parametric_knowledge.py"
    HPC/HPCE
    context_type = HPC, LPC, High Plausibiliy Contradiction without Explanation, Low plausibility Contradiction
    Edit the context
    """
    # Context type: No contradiction (no need to edit) - NC, Contradiction with Explanation (Existing) - HPCE, 
    # High Plausibiliy Contradiction without Explanation - HPC, Low plausibility Contradiction - LPC
    inputs = []
    input_context = []
    # Remove the ones that the model does not have parametirc knowledge
    # dataset = dataset.filter(lambda example: example[model_name] != 0)
    # TODO: Test
    dataset = dataset.select([i for i in range(10)])

    for instance in tqdm(dataset):
        alt_answer = instance["alt_answer"]
        alt_context = instance["alt_context"]

        # Determine context type for generation
        if context_type == "HPC/HPCE":
            if instance["HPC_context"] == "":
                # High Plausibiliy Contradiction without Explanation
                inputs.append(f"You are a smart editor that remove the explanation in the given passage, such that the answer to the question {instance['question']} is '{alt_answer}'. \n You should only output the edited passage.\n")
            else:
                # High Plausibiliy Contradiction with Explanation
                inputs.append(f"You are a smart editor that add an explanation that is logically coherent in the given passage, such that the answer to the question {instance['question']} is '{alt_answer}'. \n You should only output the edited passage.\n")
            input_context.append(alt_context)
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

def map_back_to_dataset():
    """
    Map the output result back to the original dataset
    """

    pass
    

if __name__ == "__main__":
    get_constant()
    # pdb.set_trace()
    model_name = "llama3.2-3B-Instruct"
    context_type = "HPC/HPCE"
    dataset = load_from_disk(os.path.join(os.environ["data_dir"], "model_knowledge", model_name))

    classified_dataset = classify_context(dataset)

    create_edit_prompts(dataset=classified_dataset, model_name=model_name, context_type=context_type)

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

