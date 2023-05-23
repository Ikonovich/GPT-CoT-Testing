from json import JSONDecodeError
from os import path

import openai
from torch.utils.data import dataset

from config import DATASET_FOLDER, DATASETS, chat, RESULTS_FOLDER
from file_utils import load_dataset, read_json, write_json
from query_utils import build_prompt, timer

roles = {
    "Internal": {"role": "system", "content": "You are engaging in internal reasoning. "
                                              "None of your output from this cycle will be shown to the user. "
                                              "All of your output for this cycle will be available in the next "
                                              "cycle to allow you to produce relevant output for the user."},
    "External": {"role": "system", "content": "Use the provided reasoning to answer the user's latest message. "
                                              "Follow any instructions provided by the user."}
}


def multi_message_query(model: str, messages: list[dict[str, str]], max_tokens: int):
    # Run the timer to keep from querying too quickly
    timer()

    if model not in chat:
        raise ValueError("The provided model is not a Chat-equipped model.")

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0,
        stop=None
    )

    return response["choices"][0]["message"]


def iterate(model: str, modality: str, dataset: str, args):
    num_samples = args.num_samples
    wait_time = args.wait_time
    max_tokens = args.max_tokens

    # Set the results folder, because we don't split individual step runs into separate folders
    if 'step' in dataset:
        dataset_sub = "stepwise"
    else:
        dataset_sub = dataset

    save_directory = path.join(RESULTS_FOLDER, "ScratchPad", model, modality, dataset_sub)
    # Results file: Stores last index ran, total count, correct counts, and accuracy.
    output_file = "SP" "-" + model + "-" + modality + "-" + dataset + ".json"
    output_path = path.join(save_directory, output_file)

    # Set the start index
    # This lets us continue from where we left off if the model is overloaded or the test has to restart.
    start_index = 0
    if path.exists(output_path):
        try:
            previous = read_json(filepath=output_path)
            start_index = previous["Trials"][-1]["Index"] + 1

        except JSONDecodeError as e:
            print(f"There was an error decoding the prior test results at {output_path} into json at index {e.pos}")
        except KeyError as e:
            print(f"Expected key {e.args[0]} was not found in the last index of {output_path} when "
                  f"decoded into json.")

    # Load the dataset and set the iteration range
    dataset_path = path.join(DATASET_FOLDER, DATASETS[dataset])
    data = load_dataset(dataset_path)

    if num_samples == 0:
        end_index = len(data)
    else:
        end_index = min(num_samples, len(data))

    for i in range(start_index, end_index):
        datum = data[i]
        if args.mode == "scratchpad":
            result = scratchpad_test(datum=datum, model=model, modality=modality, max_tokens=max_tokens)

        else:
            raise ValueError("The provided mode is not valid here.")

        if path.exists(output_path):
            test_results = read_json(filepath=output_path)
        else:
            test_results = {"Model": model,
                            "Modality": modality,
                            "Dataset": dataset,
                            "Interal Role": roles["Internal"],
                            "External Role": roles["External"],
                            "Trials": list()
                            }

        result["Index"] = i
        if dataset == "aqua" or "mmlu" in dataset:
            result["Options"] = datum[2]["Options"]
        test_results["Trials"].append(result)
        write_json(filepath=output_path, data=test_results)


def scratchpad_test(datum: dict, model: str, modality: str, max_tokens: int):

    x, y, test_entry = datum

    prompt = build_prompt(question=x, modality=modality, use_simple_prompt=True,
                          bracket_extract=True)

    messages = [
        roles["Internal"],
        {"role": "user", "content": prompt}
    ]

    internal_message = multi_message_query(model=model, messages=messages, max_tokens=max_tokens)
    reasoning = internal_message["content"]
    print(f"Internal reasoning: {reasoning}")
    messages = [
        roles["External"],
        {"role": "user", "content": prompt},
        internal_message
    ]

    external_message = multi_message_query(model="gpt-4", messages=messages, max_tokens=max_tokens)
    response = external_message["content"]

    result = {"Index": test_entry["Index"], "Query": prompt, "Reasoning": reasoning, "Response": response,
              "GT": y}

    return result
