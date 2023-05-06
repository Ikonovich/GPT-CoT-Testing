# Re-formats queries from cleaned dataset.
import json
import os
import time
import re

# Loads data from a json file
def load_data_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

def get_question_string(question, choices):
    choices_str = "\n".join(f"{key}: {value}" for key, value in choices.items())
    return f"{question}\n{choices_str}\n"

# Saves data to a json file
def save_to_json(data, file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

data_lists = ["zero-shot-answer-first", "zero-shot-explanation-first", "zero-shot-supressed", "zero-shot-cot", "zero-shot"]
default_prompt = "\nPlace your final answer choice in squiggly brackets. "
prompts = ["Provide the answer followed by the explanation.", "Provide the explanation followed by the answer.", "Provide only the answer", "Let’s think step by step.", ""]
original_data = load_data_from_file("./formatted/mmlu_formatted/high_school_mathematics_combined.json")


for iterator, datas in enumerate(data_lists):
    data_path = "./generated/gpt4/" + datas + ".json"
    data = load_data_from_file(data_path)
    results = []
    for question in data:
        question_string = get_question_string(question, original_data[question]["choices"])
        response = data[question]
        options = original_data[question]["choices"]
        ground_truth = original_data[question]["correct_answer"]
        question_string += default_prompt + prompts[iterator]
        result = {"Q":question_string, "R":response, "O":options, "GT":ground_truth}
        results.append(result)
    save_path = "./generated/gpt4/" + "cleaned/" + datas + "_formatted.json"
    save_to_json(results, save_path)