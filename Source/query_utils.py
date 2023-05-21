import time
from json import JSONDecodeError
from os import path

import openai
from config import RESULTS_FOLDER, DATASET_FOLDER, DATASETS, \
    completion, chat, two_stage_extract_prompt, suppression_prompt, cot_prompt, \
    answer_first_prompt, explanation_first_prompt, in_bracket_prompt
from file_utils import load_dataset, write_json, read_json

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
    B = ""
    if bracket_extract:
        B = in_bracket_prompt
    if not use_simple_prompt:
        Q += "Q: "
        A += "A: "

    match modality:
        case "zero_shot_cot":
            output = f"{Q}{question} {B} {A}{cot_prompt}"
        case "zero_shot":
            output = f"{Q}{question} {B} {A}"
        case "the_answer_is":
            output = f"{Q}{question}. {B} {A}The answer is "
        case "zero_shot_no_extract":
            output = f"{Q}{question} {B} {A}The answer (arabic numerals) is "
        case "suppressed_cot":
            output = f"{Q}{question} {B} {suppression_prompt}. {A}"
        case "explanation_first":
            output = f"{Q}{question} {B} {explanation_first_prompt}. {A}"
        case "answer_first":
            output = f"{Q}{question} {B} {answer_first_prompt}. {A}"
        case _:
            raise ValueError("The provided test type has not been defined.")

    return output


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

    if use_simple_prompt:
        prompt = "Simple"
    else:
        prompt = "Initial"

    # Set the dataset folder, because we don't split individual step runs into separate folders
    if 'step' in dataset:
        dataset_sub = "stepwise"
    else:
        dataset_sub = dataset

    save_directory = path.join(RESULTS_FOLDER, model, modality, dataset_sub)

    # Results file: Stores last index ran, total count, correct counts, and accuracy.
    output_file = prompt + "-" + extraction_type + "-" + model + "-" + modality + "-" + dataset + ".json"
    output_path = path.join(save_directory, output_file)

    dataset_path = path.join(DATASET_FOLDER, DATASETS[dataset])

    start_index = 0
    if cont and path.exists(output_path):
        try:
            previous = read_json(filepath=output_path)
            start_index = previous[-1]["Index"] + 1

        except JSONDecodeError as e:
            print(f"There was an error decoding the prior test results at {output_path} into json at index {e.pos}")
        except KeyError as e:
            print(f"Expected key {e.args[0]} was not found in the last index of {output_path} when "
                  f"decoded into json.")

    # Load the dataset
    data = load_dataset(dataset_path)
    # Set the end index
    # This lets us continue from where we left off if the model is overloaded or the test has to restart.
    if num_samples == 0:
        end_index = len(data)
    else:
        end_index = min(num_samples, len(data))

    for i in range(start_index, end_index):
        # Run the timer to keep from querying too quickly
        timer(wait_time)

        x, y, test_entry = data[i]
        # Build the prompt out of our question
        prompt = build_prompt(question=x, modality=modality, use_simple_prompt=use_simple_prompt,
                              bracket_extract=bracket_extract)
        # Get the initial response from the model
        response = query(model=model, prompt=prompt, max_tokens=max_tokens)

        if dataset == "aqua" or "mmlu" in dataset:
            options = test_entry["Options"]
        else:
            options = None


        # Validate the extraction type and perform two-stage extraction, if necessary.
        if extraction_type == "in-brackets":
            extraction_response = "None"
        elif extraction_type == "two-stage":
            # Run the timer to keep from querying too quickly, then resubmit the response for answer extraction
            timer(wait_time)
            extraction_prompt = prompt + " " + response + "\n" + two_stage_extract_prompt
            extraction_response = query(model, extraction_prompt, max_tokens=max_tokens)
        elif extraction_type == "two-stage-style-two":
            extraction_prompt = two_stage_style_two_generation(answer=response, options=options)
            extraction_response = query(model, extraction_prompt, max_tokens=max_tokens)
        else:
            raise ValueError("The provided extraction type is not valid.")

        result = {"Index": test_entry["Index"], "Query": prompt, "Response": response, "Extract-Response": extraction_response,
                  "GT": y}

        if dataset == "aqua" or "mmlu" in dataset:
            result["Options"] = options
        if save:
            if path.exists(output_path):
                test_results = read_json(filepath=output_path)
            else:
                test_results = list()
            test_results.append(result)
            write_json(filepath=output_path, data=test_results)

    print("Test " + model + "-" + modality + "-" + dataset + " completed.")


# Used to format the second two-stage-second-style prompt.
def two_stage_style_two_generation(answer: str, options: dict) -> str:
    prompt = "Only provide one letter answer. The following statement contains the full answer for " \
             "an unknown multiple-choice math problem. Somewhere in the string provided, " \
             "is the final answer choice, ranging from [A, B, C, D]. We want to know the answer " \
             "choice only. Given this string, write the final answer choice presented there. " \
             "Do not provide any other explanations or descriptions. If no answer letter provided, " \
             "answer with \"N/A\". Full answer:\n"
    prompt += answer
    prompt += "\nPossible choices:\n"
    for key in options:
        prompt += f"{key}: {options[key]}"
    prompt += "\nDo not explain your answer. Only provide the answer choice in squiggly brackets in the following format {answer}: "

    return prompt