# Used to extract answers using GPT-4
import json
import os
import time
import re
import openai

openai.api_key = ""

def load_data_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

def ask_gpt(question):
    response = openai.ChatCompletion.create(
        model="gpt-4", 
        messages=[{"role":"user", "content":question}],
        max_tokens=1000,
        temperature=0,
        stop=None
    )

    return response["choices"][0]["message"]["content"]

def ask_gpt_with_retry(question, max_retries=5, initial_delay=15, backoff_factor=2):
    retries = 0

    while retries <= max_retries:
        try:
            return ask_gpt(question)
        except (openai.error.APIError, openai.error.APIConnectionError, openai.error.RateLimitError) as e:
            if retries == max_retries:
                print(f"Request failed after {max_retries} retries. Error: {e}")
                return None  # or raise an exception, or return a default value
            else:
                delay = initial_delay * (backoff_factor ** retries)
                print(f"Request failed with error: {e}. Retrying in {delay} seconds...")
                time.sleep(delay)
                retries += 1
    
def save_to_json(data, file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def process_dataset(file_path, original_data):
    data = load_data_from_file(file_path)
    results = []

    for question in data:
        answer = data[question]
        ground_truth = original_data[question]["correct_answer"]
        options = "A: " + original_data[question]["choices"]["A"] + "\n" + "A: " + original_data[question]["choices"]["B"] + "\n" + "A: " + original_data[question]["choices"]["C"] + "\n" + "A: " + original_data[question]["choices"]["D"] + "\n"

        question_string = "Only provide one letter answer. The following statement contains the full answer for an unknown multiple-choice math problem. We do not care about the problem itself. Somewhere in the string provided, is the final answer choice, ranging from [A, B, C, D]. We want to know the answer choice only. Given this string, write the final answer choice presented there. Do not provide any other explanations or descriptions. If no answer letter provided, answer with \"N/A\". Full answer:\n " 
        question_string += answer
        question_string += "\nPossible choices:\n"
        question_string += options
        question_string += 'Do not explain your answer. Only provide the answer choice in squiggly brackets in the following format {answer}:' 

        extracted = ask_gpt_with_retry(question_string)
        #print(question_string)
        if extracted is not None:
            result = {
                "Q": question,
                "R": answer,
                "O": original_data[question]["choices"],
                "A": extracted,
                "GT": ground_truth
            }
            results.append(result)
            print("Answer: ", extracted)
    return results

#data = load_data_from_file("./generated/gpt35-turbo/zero-shot-answer-first.json")
original_data = load_data_from_file("./formatted/mmlu_formatted/high_school_mathematics_combined.json")


results = []

# Folder containing your json datasets
folder_path = './generated/gpt4/hs_responses/'

# Collect all json files
json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    

# Process each json file
for json_file in json_files:
    print(json_file)
    results = process_dataset(os.path.join(folder_path, json_file), original_data)
    save_directory = "./generated/gpt4/2phase/"
    os.makedirs(save_directory, exist_ok=True)  # Create the directory if it doesn't exist
    output_filename = os.path.join(save_directory, ("extracted_"+json_file))
    save_to_json(results, output_filename)
    
#save_to_json(results, "./generated/gpt35-turbo/af-extract-exclusive.json")