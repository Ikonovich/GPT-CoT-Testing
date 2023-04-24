import json
import re
import time
from json import JSONDecodeError
from os import path

import openai

from config import MAX_TOKENS, RESULTS_FOLDER, DATASET_FOLDER, datasets, \
    completion, chat, extract_prompt, suppression_prompt, cot_prompt, MAX_SAMPLES, WAIT_TIME, \
    answer_first_prompt, explanation_first_prompt, USE_SIMPLE_PROMPT
from data_utils import load_dataset

# Answer cleaning pattern
pattern = re.compile(r"[^\s0-9.-]")

# Stores the time of the last query
last_query_time = 0


# Timer function that pauses operation until the time since last query exceeds config.WAIT_TIME
def timer():
    global last_query_time

    cur_time = time.time()
    diff = cur_time - last_query_time
    if diff < WAIT_TIME:
        time.sleep(WAIT_TIME - diff)

    last_query_time = time.time()


def query(model: str, prompt: str) -> str:

    if model in completion:
        response = openai.Completion.create(
            engine=model,
            prompt=prompt,
            max_tokens=MAX_TOKENS,
            temperature=0,
            stop=None
        )

        return response["choices"][0]["text"]

    elif model in chat:
        response = openai.ChatCompletion.create(
            model=model,
            messages = [{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS,
            temperature=0,
            stop=None
        )

        return response["choices"][0]["message"]["content"]

    else:
        raise ValueError("The provided model has not been defined.")


def build_prompt(question: str, modality: str) -> str:
    # Constructs the question query based on the test type and the prompt settings in config.
    # If simple is false, appends Q to the start of each question and A at certain points in the question

    if USE_SIMPLE_PROMPT:
        Q = ""
        A = ""
    else:
        Q = "Q: "
        A = "A: "

    match modality:
        case "zero_shot_cot":
            output = f"{Q}{question} \n{A}{cot_prompt}"
        case "zero_shot":
            output = f"{Q}{question}"
        case "the_answer_is":
            output = f"{Q}{question}. \n{A}The answer is "
        case "zero_shot_no_extract":
            output = f"{Q}{question} \n{A}The answer (arabic numerals) is "
        case "suppressed_cot":
            output = f"{Q}{question} \n{A}{suppression_prompt} "
        case "explanation_first":
            output = f"{Q}{question} \n{A}{explanation_first_prompt} "
        case "answer_first":
            output = f"{Q}{question} \n{A}{answer_first_prompt} "
        case _:
            raise ValueError("The provided test type has not been defined.")

    return output


def clean_numeric_answer(answer: str) -> float | str:
    # Citation: This method and some of this code is taken from
    # DOI 10.48550/arXiv.2205.11916
    # Remove commas and find all numerical answers
    pred = answer.replace(",", "")
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
        return ""


def run_test(model: str, modality: str, dataset: str, save: bool = True, cont: bool = False):
    # Runs a test on a given model, test modality, and dataset.
    # If num samples is 0, runs the whole dataset, otherwise stops at index num_samples
    # Keeps a running accuracy and saves the results as they come in if desired.
    # If cont is true, will attempt to load the first line from the file
    # with the name Model-Modality-Dataset-Results.jsonl and begin iterating from the value stored there at
    # last_index

    # Set the dataset folder, because we don't split individual step runs into separate folders
    if 'step' in dataset:
        dataset_sub = "stepwise"
    else:
        dataset_sub = dataset

    # Results file: Stores last index ran, total count, correct counts, and accuracy.
    metadata_file = "SimplePrompt-" + model + "-" + modality + "-" + dataset + "-Metadata.jsonl"
    metadata_path = path.join(RESULTS_FOLDER, model, modality, dataset_sub, metadata_file)
    output_file = "SimplePrompt-" + model + "-" + modality + "-" + dataset + ".jsonl"
    output_path = path.join(RESULTS_FOLDER, model, modality, dataset_sub, output_file)
    dataset_path = path.join(DATASET_FOLDER, datasets[dataset])

    # Store the start index, total number of questions asked so far, the number answered correctly,
    # a rolling accuracy, and the query / response / answer / ground truth.
    # If cont is true, this information will be loaded from the file Model-Modality-Dataset.jsonl.
    start_index = 0
    total = 0
    correct = 0

    if cont and path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as saved:
                line = saved.readline()
                vals = json.loads(line)

                start_index = vals["Last Sample Index"] + 1
                total = vals["Total"]
                correct = vals["Correct"]

        except JSONDecodeError as e:
            print(f"There was an error decoding the first line of {metadata_path} into json at index {e.pos}")
        except KeyError as e:
            print(f"Expected key {e.args[0]} was not found in the first line of {metadata_path} when decoded into json.")

    # Load the dataset
    data = load_dataset(dataset_path)
    # Set the end index
    # This lets us continue from where we left off if the model is overloaded
    if MAX_SAMPLES == 0:
        end_index = len(data)
    else:
        end_index = min(MAX_SAMPLES, len(data))

    for i in range(start_index, end_index):
        # Run the timer to keep from querying too quickly
        timer()

        x, y = data[i]
        # Build the prompt out of our question
        prompt = build_prompt(question=x, modality=modality)
        # Get the initial response from the model
        response = query(model=model, prompt=prompt)

        #If replicating  DOI 10.48550/arXiv.2205.11916, zero shot has no answer extraction prompt.
        if modality == "zero_shot_no_extract":
            extraction_response = response
        else:
            # Run the timer to keep from querying too quickly, then resubmit the response for answer extraction
            timer()
            extraction = prompt + " " + response + "\n" + extract_prompt
            extraction_response = query(model, extraction)
        answer = clean_numeric_answer(extraction_response)

        total += 1
        # Clean y if it's a string
        if type(y) == str:
            y = y.replace(",", "")
        if answer != "" and answer == float(y):
            correct += 1
        accuracy = correct / total * 100

        result = {"Q": prompt, "R": response, "Extract-Response": extraction_response, "A": answer, "GT": y}

        #print(f"Question: {prompt}\nResponse: {response}\nAnswer: {answer}\nGT: {y}")

        if save:
            # Save the run results
            with open(metadata_path, 'w') as metadata_file:
                processed = json.dumps({"Total": total, "Correct": correct,
                                        "Accuracy": accuracy, "Last Sample Index": i,
                                        "Extraction Prompt": extract_prompt, "Modality": modality, "Model": model})
                metadata_file.write(processed)
            with open(output_path, 'a+') as output_file:
                processed = json.dumps(result) + '\n'
                output_file.write(processed)


    print("Test " + model + "-" + modality + "-" + dataset + " completed.")
