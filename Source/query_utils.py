import json
import re
import time
from json import JSONDecodeError
from os import path

import openai

from config import RESULTS_FOLDER, DATASET_FOLDER, datasets, \
    completion, chat, two_stage_extract_prompt, suppression_prompt, cot_prompt, \
    answer_first_prompt, explanation_first_prompt
from data_utils import load_dataset

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


def query(model: str, prompt: str, max_tokens: int) -> str:

    if model in completion:
        response = openai.Completion.create(
            engine=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0,
            stop=None
        )

        return response["choices"][0]["text"]

    elif model in chat:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0,
            stop=None
        )

        return response["choices"][0]["message"]["content"]

    else:
        raise ValueError("The provided model has not been defined.")


def build_prompt(question: str, modality: str, use_simple_prompt: bool, bracket_extract: bool) -> str:
    # Constructs the question query based on the test type and the prompt settings in config.
    # If simple is false, appends Q to the start of each question and A at certain points in the question

    # Generate universal insertions
    Q = ""
    A = ""
    if bracket_extract:
        A = ""
    if not use_simple_prompt:
        Q += "Q: "
        A += "A: "

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
            output = f"{Q}{question} \n{suppression_prompt}. \n{A} "
        case "explanation_first":
            output = f"{Q}{question} \n{explanation_first_prompt}. \n{A} "
        case "answer_first":
            output = f"{Q}{question} \n{answer_first_prompt}. \n{A} "
        case _:
            raise ValueError("The provided test type has not been defined.")

    return output


def clean_numeric_answer(answer: str, extraction_type: str) -> float | str:
    # Remove commas
    pred = answer.replace(",", "")

    if extraction_type == "in-brackets":
        match = re.search(brackets_pattern, pred)
        if match is None:
            raise IndexError("No bracketed answer could be located.")
        pred = match.group(0)

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
        return ""


def run_test(model: str, modality: str, dataset: str, args):
    # Runs a test on a given model, test modality, and dataset.
    # If num samples is 0, runs the whole dataset, otherwise stops at index num_samples
    # Keeps a running accuracy and saves the results as they come in if desired.
    # If cont is true, will attempt to load the first line from the file
    # with the name Model-Modality-Dataset-Results.jsonl and begin iterating from the value stored there at
    # last_index
    save = args.save
    cont = args.continuation
    use_simple_prompt = args.use_simple_prompt
    extraction_type = args.extraction_type
    num_samples = args.num_samples
    wait_time = args.wait_time
    max_tokens = args.max_tokens

    if extraction_type == "in-brackets":
        bracket_extract = True
    else:
        bracket_extract = False

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
            print(f"Expected key {e.args[0]} was not found in the first line of {metadata_path} when "
                  f"decoded into json.")

    # Load the dataset
    data = load_dataset(dataset_path)
    # Set the end index
    # This lets us continue from where we left off if the model is overloaded
    if num_samples == 0:
        end_index = len(data)
    else:
        end_index = min(num_samples, len(data))

    for i in range(start_index, end_index):
        # Run the timer to keep from querying too quickly
        timer(wait_time)

        x, y = data[i]
        # Build the prompt out of our question
        prompt = build_prompt(question=x, modality=modality, use_simple_prompt=use_simple_prompt,
                              bracket_extract=bracket_extract)
        # Get the initial response from the model
        response = query(model=model, prompt=prompt, max_tokens=max_tokens)

        #  If replicating  DOI 10.48550/arXiv.2205.11916, zero shot has no answer extraction prompt.
        if extraction_type == "none" or extraction_type == "in-brackets":
            extraction_response = response
        elif extraction_type == "two-stage":
            # Run the timer to keep from querying too quickly, then resubmit the response for answer extraction
            timer(wait_time)
            extraction = prompt + " " + response + "\n" + two_stage_extract_prompt
            extraction_response = query(model, extraction, max_tokens=max_tokens)
        else:
            raise ValueError("The provided extraction type is not valid.")

        answer = clean_numeric_answer(answer=extraction_response, extraction_type=extraction_type)

        total += 1
        # Clean y if it's a string
        if type(y) == str:
            y = y.replace(",", "")
        if answer != "" and answer == float(y):
            correct += 1
        accuracy = correct / total * 100

        result = {"Q": prompt, "R": response, "Extract-Response": extraction_response, "A": answer, "GT": y}

        # print(f"Question: {prompt}\nResponse: {response}\nAnswer: {answer}\nGT: {y}")

        if save:
            # Save the run results
            with open(metadata_path, 'w') as metadata_file:
                processed = json.dumps({"Total": total, "Correct": correct,
                                        "Accuracy": accuracy, "Last Sample Index": i,
                                        "Extraction Prompt": two_stage_extract_prompt, "Modality": modality,
                                        "Model": model})
                metadata_file.write(processed)
            with open(output_path, 'a+') as output_file:
                processed = json.dumps(result) + '\n'
                output_file.write(processed)

    print("Test " + model + "-" + modality + "-" + dataset + " completed.")
