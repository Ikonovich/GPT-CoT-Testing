# Parses and asks questions to gpt
import json
import os
import time
import re
import openai

openai.api_key = "" # ENTER API KEY HERE

def load_data_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

def get_question_string(question, choices):
    choices_str = "\n".join(f"{key}: {value}" for key, value in choices.items())
    return f"{question}\n{choices_str}\n"

def ask_gpt(question):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=[{"role":"user", "content":question}],
        max_tokens=1000,
        temperature=0,
        stop=None
    )

    return response["choices"][0]["message"]["content"]

def ask_gpt_with_retry(question, max_retries=5, initial_delay=3, backoff_factor=2):
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

data = load_data_from_file("./formatted/mmlu_formatted/high_school_mathematics_combined.json")

results = {}
for question_text, question_data in data.items():
    question_string = get_question_string(question_text, question_data["choices"])
    question_string += "\nPlace your final answer choice in squiggly brackets. Provide the answer followed by the explanation."

    answer = ask_gpt_with_retry(question_string)
    if answer is not None:
        results[question_text] = answer
        print(f"Question: {question_text}")
        print(f"Answer: {answer}\n")
    

save_to_json(results, "./generated/gpt35-turbo/zero-shot-answer-first.json")
