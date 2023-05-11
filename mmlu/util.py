import json
import re
import time
from json import JSONDecodeError
from os import path

import openai


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

# Answer cleaning patterns
float_pattern = re.compile(r"[^\s0-9.-]")
brackets_pattern = re.compile(r"{(.*?)}")

# Stores the time of the last query
last_query_time = 0


# Timer function that pauses operation until the time since last query exceeds config.WAIT_TIME
def timer(wait_time: float):
    global last_query_time

    cur_time = time.time()
    diff = cur_time - last_query_time
    if diff < wait_time:
        time.sleep(wait_time - diff)

    last_query_time = time.time()



def clean_numeric_answer(answer: str, extraction_type: str) -> float | str:
    # Remove commas
    pred = answer.replace(",", "")

    if extraction_type == "in-brackets":
        match = re.search(brackets_pattern, pred)
        if match != None:
            #raise IndexError("No bracketed answer could be located.")
            pred = match.group(0)
            return pred[1]

    # Citation: This basic method and some of this code is taken from
    # DOI 10.48550/arXiv.2205.11916
    # Find numerical answers
    pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]
    # Get the first word of the response
    if len(pred) == 0:
        return ""
    pred = pred[0]

    # Remove trailing periods
    if pred[-1] == ".":
        pred = pred[:-1]

    # Try to return it as a float, otherwise return an empty string
    try:
        final = float(pred)
        return final
    except Exception as e:
        return e

data_lists = ["zero-shot-answer-first", "zero-shot-explanation-first", "zero-shot-supressed", "zero-shot-cot", "zero-shot"]
default_prompt = "\nPlace your final answer choice in squiggly brackets. "
prompts = ["Provide the answer followed by the explanation.", "Provide the explanation followed by the answer.", "Provide only the answer", "Let’s think step by step.", ""]

for iterator, datas in enumerate(data_lists):
    data_path = "./generated/gpt4/cleaned/combined" + datas + "_formatted.json"
    data = load_data_from_file(data_path)
    results = []
    for question in data:
        #print(question["R"])
        #print("Brackets: ",clean_numeric_answer(question["R"], "in-brackets"))
        #print("_______________________________________")
        gt = question["GT"]
        extracted = clean_numeric_answer(question["R"], "in-brackets")
        result = {"GT":gt, "A":extracted}
        results.append(result)
    save_path = "./generated/gpt4/" + "cleaned/answers/" + datas + "_a.json"
    save_to_json(results, save_path)