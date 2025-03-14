import os
import sys
import re
import pdb
sys.path.append(os.getcwd())
import json
import time
import requests
import openai
import argparse
from openai import OpenAI
from tqdm import tqdm
from datasets import load_from_disk, load_dataset, Dataset
from dotenv import load_dotenv

from utils.constant import get_constant
load_dotenv()

# Load API keys from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

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
    dataset = dataset.select([i for i in range(20)])
    
    classified_data = []
    for instance in tqdm(dataset):
        model_answer = instance["answer1"] if instance[model_name] == 1 else instance["answer2"]
        model_context = instance["context1"] if instance[model_name] == 1 else instance["context2"]

        alt_answer = instance["answer1"] if instance[model_name] == 2 else instance["answer2"]
        alt_context = instance["context1"] if instance[model_name] == 2 else instance["context2"]

        # Determine if it is HPC or HPCE
        # Note: TODO: Maybe no need to classify here
        # Decide whether there include enough explanation (rate 1-5)
        prompt = "You are a smart reader, your job is to rate from 1 to 5 whether the given text has enough **explanation** to select one answer over another. The lower the score, the less explanation the given passage contains to answer the question. Your job is to determine the logical coherence of the given passage, and you should not be affected by the factual presence of the given passage. You should only output the integer scores. Below are some examples:\n"
        "Passage 1: Apomorphine is said to be main psychoactive compound present. \n" + \
        "Question 1: Which of the following are present in Nymphaea nouchali var. caerulea: apomorphine, aporphine, or neither? \n" + \
        "Answer 1a: apomorphine \n Answer 1b: aporphine \n" + \
        "Rating 1: 1 \n" + \
        "Passage 2: Past research claim that aporphine are present in Nymphaea nouchali var. However, a recent study provide solid evidence that Apomorphine is the main psychoactive compound present, instad of aporphine.\n" + \
        "Question 2: Which of the following are present in Nymphaea nouchali var. caerulea: apomorphine, aporphine, or neither? \n" + \
        "Answer 2a: apomorphine \n Answer 2b: aporphine \n" + \
        "Rating 2: 5 \n"
        content = f"Passage: {alt_context}\nQuestion: {instance['question']} \n" + \
            f"Answer a: {instance['answer1']} \n Answer b: {instance['answer2']} \nRating:"

        # Temporary Workaround: Avoid Rating, directly ask GPT-4o to write Explanations
        # completion = client.chat.completions.create(
        #     model="gpt-4o",
        #     messages=[
        #         {"role": "developer", "content": prompt},
        #         {
        #             "role": "user",
        #             "content": content
        #         }
        #     ]
        # )
        # response = completion.choices[0].message.content
        response = "1"
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
            if alt_context != model_context:
                # Avoid including invalid instances where there is only one context
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
                    "LPC_context": "",
                    "LPC_answer": "",
                    "conflict_explanation_rating": rating
                })
        else:
            raise Exception(f"The output does not contain a rating. Model Output: {response}")
    classified_data = Dataset.from_list(classified_data)
    os.makedirs(os.path.join(os.environ["data_dir"], "intermediate_processing", "classified_context"), exist_ok=True)
    save_path = os.path.join(os.environ["data_dir"], "intermediate_processing", "classified_context", f"{model_name}.jsonl")
    # Write to directory
    classified_data.save_to_disk(save_path)
    return classified_data

def format_LPC_prompt(question, context, answer):
    """
    Code to format the prompt for LPC generation
    """
    # Load the prompt
    prompt_file = os.path.join(os.environ["base_dir"], "prompts", "LPC.txt")
    curr_prompt_file = open(prompt_file, "r")
    prompt_template = curr_prompt_file.read()
    curr_prompt_file.close()
    return prompt_template.format(question=question, answer=answer, context=context)
        

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
    dataset = dataset.select([i for i in range(15)])

    for instance in tqdm(dataset):
        alt_answer = instance["alt_answer"]
        alt_context = instance["alt_context"]

        # Determine context type for generation
        if context_type == "HPCHPCE":
            if instance["HPC_context"] == "":
                # High Plausibiliy Contradiction without Explanation
                inputs.append(f"You are a smart editor that removes the explanation in the given passage, such that the answer to the question {instance['question']} is '{alt_answer}'. It should not contain any reasoning of why the answer should not be {instance['NC_answer']}. \n You should only output the edited passage.")
            else:
                # High Plausibiliy Contradiction with Explanation
                inputs.append(f"You are a smart editor that adds a contrastive explanation that is logically coherent in the given passage. Your edit should explain why the answer to the question {instance['question']} is the answer '{alt_answer}' instead of {instance['NC_answer']}. \n However, you should write it as a statement, instead of explicitly call out the given answer. You should only output the edited passage.")
            input_context.append(alt_context)
        elif context_type == "LPC":
            # Low plausibility Contradiction
            inputs.append(format_LPC_prompt(question = instance['question'], context = alt_context, answer = alt_answer))
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
    return dataset, inputs, input_context

def query_whole_dataset(dataset, prompts, context, context_type):
    output_contexts = []
    output_answers = []
    if context_type == "LPC":
        key_field = "LPC"
        dataset = dataset.remove_columns(["LPC_context", "LPC_answer"])
    else:
        # Avoid checking Exp score, By default generating explanations for all instance
        key_field = "HPCE"
        dataset = dataset.remove_columns(["HPCE_context", "HPCE_answer"])
            
    for instance, prompt, context in zip(dataset, prompts, input_context):
        completion = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "developer", "content": ""},
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        output = completion.choices[0].message.content

        if key_field == "LPC":
            # Parse for EditedPassage and NewAnswer
            match = re.search(r'EditedPassage:\s*(.*?)\s*\n\s*NewAnswer:\s*(.*)', output, re.DOTALL)
            if match:
                edited_passage = match.group(1).strip()
                new_answer = match.group(2).strip()
            else:
                raise Exception(f"Failed to strip the Edited passage and new answer from the output. The output = {output}")
        else:
            edited_passage = output
            new_answer = instance["alt_answer"]
        output_contexts.append(edited_passage)
        output_answers.append(new_answer)
    dataset = dataset.add_column(f"{key_field}_context", output_contexts)
    dataset = dataset.add_column(f"{key_field}_answer", output_answers)
    return dataset
    

def submit_batch_job(input_file_path):
    """
    Submits the batch job to OpenAI API.
    """
    
    print("Uploading intput file...")
    batch_input_file = openai_client.files.create(
        file=open(input_file_path, "rb"),
        purpose="batch"
    )

    print(batch_input_file)
    batch = openai_client.batches.create(
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
        batch = openai_client.batches.retrieve(batch_id)

        print(f"Batch job status: {batch.status}...")
        if batch.status in ["completed", "failed", "cancelled", "expired"]:
            print(f"batchID = {batch_id}")
            return batch
        time.sleep(60)  # Wait before checking again

def download_results(batch, output_file_path):
    """
    Downloads the results of a batch job.
    """
    results = openai_client.files.content(batch.output_file_id)
    # The output context is automatically a jsonl file
    with open(output_file_path, "w") as f:
        f.write(results.text)

def map_back_to_dataset(classified_data, context_type, openai_input_file_path, output_prediction_path):
    """
    Map the output result back to the original dataset
    classified_data: Output of the classified data
    context_type = "HPCHPCE", "LPC"
    """
    # Load the input file and output file
    input_data = load_dataset("json", data_files=openai_input_file_path)["train"]
    output_data = load_dataset("json", data_files=output_prediction_path)["train"]
    
    classified_data = classified_data.to_list()
    # Important: This assumes that the dataset is never shuffled
    for idx, input_prompt in enumerate(input_data):
        instance = classified_data[idx]
        if context_type == "LPC":
            key_field = "LPC"
        else:
            # Check whether it is asking for HPC or HPCE
            if instance["HPC_context"] == "" and instance["HPCE_context"] != "":
                key_field = "HPC"
            elif instance["HPCE_context"] == "":
                key_field = "HPCE"
            else:
                raise Exception("Both fields of HPC/HPCE are filled. Did you pass the wrong context type?")
        # Find corresponding output and Map the input output instance back
        # pdb.set_trace()
        output = output_data.filter(lambda example: example["custom_id"] == input_prompt["custom_id"])[0]["response"]["body"]["choices"][0]["message"]["content"]
        # print(instance[f"{key_field}_context"])
        # Update the corresponding field
        if key_field == "LPC":
            # Parse for EditedPassage and NewAnswer
            match = re.search(r'EditedPassage:\s*(.*?)\s*\n\s*NewAnswer:\s*(.*)', output, re.DOTALL)
            if match:
                edited_passage = match.group(1).strip()
                new_answer = match.group(2).strip()
                classified_data[idx][f"{key_field}_context"] = edited_passage
                classified_data[idx][f"{key_field}_answer"] = new_answer
            else:
                raise Exception(f"Failed to strip the Edited passage and new answer from the output. The output = {output}")
        else:
            classified_data[idx][f"{key_field}_context"] = output
            classified_data[idx][f"{key_field}_answer"] = instance["alt_answer"]
        # print(instance[f"{key_field}_context"])
        # print(instance)
    return Dataset.from_list(classified_data)
    

if __name__ == "__main__":
    get_constant()

    parser = argparse.ArgumentParser()
    # Required positional argument
    parser.add_argument('--test_model_name', type=str, default="llama3.2-3B-Instruct",
                            help='name of a dataset')
    parser.add_argument('--use_batch', action="store_true",
                            help='whether to use the batching feature in OpenAI. Enable it could be slow.')
    parser.add_argument('--classified_path', type=str, default=None,
                            help='the path to classified contexts. If it is not none, then we do not classify the context (into w/ and wo/ explanations) and will load from the given path. \n If "x" is passed, then it load the default classified contexts for each model (classified_context/model_name.jsonl).')
    args = parser.parse_args()

    model_name = args.test_model_name
    # # context_type = args.context_type

    # Load from derived model knowledge
    dataset = load_from_disk(os.path.join(os.environ["data_dir"], "model_knowledge", model_name))

    if args.classified_path is not None:
        if args.classified_path == "x":
            dataset = load_from_disk(os.path.join(os.environ["data_dir"], "intermediate_processing", "classified_context", f"{model_name}.jsonl"))
        else:
            dataset = load_from_disk(args.classified_path)
    else:
        dataset = classify_context(dataset)
    for context_type in ["HPCHPCE", "LPC"]:

        print(f"Creating edit prompts for {context_type}")
        # Create the prompt for edits
        dataset, inputs, input_context = create_edit_prompts(dataset=dataset, model_name=model_name, context_type=context_type)

        if args.use_batch:
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
        else:
            dataset = query_whole_dataset(dataset, prompts=inputs, context=input_context, context_type=context_type)
            print(dataset)

    output_file_path = os.path.join(os.environ["data_dir"], "intermediate_processing", "HPCHPCE", f"{model_name}.jsonl")
    input_file_path = os.path.join(os.environ["data_dir"], "temp", f"{model_name}_edit_input.jsonl")

    if args.use_batch:
        # Map the OpenAI edits back to the datasets
        dataset = map_back_to_dataset(classified_data = classified_dataset, context_type = "HPCHPCE", openai_input_file_path=input_file_path, output_prediction_path=output_file_path)

        output_file_path = os.path.join(os.environ["data_dir"], "intermediate_processing", "LPC", f"{model_name}.jsonl")
        dataset = map_back_to_dataset(classified_data = dataset, context_type = "LPC", openai_input_file_path=input_file_path, output_prediction_path=output_file_path)

    # Write to directory Save to jsonl
    dataset.to_json(os.path.join(os.environ["data_dir"], "final_data", f"{model_name}_v4.jsonl"))
