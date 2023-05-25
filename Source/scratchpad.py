from json import JSONDecodeError
from os import path

from config import DATASET_FOLDER, DATASETS, RESULTS_FOLDER
from file_utils import load_dataset, read_json, write_json
from query_utils import build_prompt, multi_message_query
from step_generation import step_generation_test

roles = {
    "Reasoning": {"role": "system", "content": "You are engaging in internal reasoning. "
                                               "None of your output from this cycle will be shown to the user. "
                                               "All of your output for this cycle will be available in the next "
                                               "cycle to allow you to produce relevant output for the user."},
    "Answering": {"role": "system", "content": "Use the provided reasoning to answer the user's latest message. "
                                               "Follow any instructions provided by the user."},
    "Step-Making": {"role": "system", "content": "You will be provided with a simple mathematical formula. Solve it "
                                                 "step by step. Do not use any words in your steps. An example of the desired "
                                                 "formatting is as follows:"
                                                 "2 * 4 = 8"
                                                 "2: 3 + 8 = {11} "},
    "Step-Extracting": {"role": "system", "content": "Use the provided reasoning to provide only "
                                                     "the steps utilized to solve the problem."},
}


def multi_query_test(model_one: str, model_two: str, modality: str, dataset: str, args):
    num_samples = args.num_samples
    max_tokens = args.max_tokens
    mode = args.mode

    # Set the results folder, because we don't split individual step runs into separate folders
    if 'step' in dataset:
        dataset_sub = "stepwise"
    else:
        dataset_sub = dataset

    save_directory = path.join(RESULTS_FOLDER, mode, model_one, modality, dataset_sub)
    # Results file: Stores last index ran, total count, correct counts, and accuracy.
    output_file = mode + "-" + model_one + "-" + model_two + "-" + modality + "-" + dataset + ".json"
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
            result = scratchpad_test(datum=datum, model_one=model_one, model_two=model_two, modality=modality,
                                     max_tokens=max_tokens)
        elif args.mode == "step-generation":
            result = step_generation_test(datum=datum, model_one=model_one, model_two=model_two, modality=modality,
                                          max_tokens=max_tokens)
        else:
            raise ValueError("The provided mode is not valid here.")

        if path.exists(output_path):
            test_results = read_json(filepath=output_path)
        else:
            test_results = {"Internal Model": model_one,
                            "External Model": model_two,
                            "Modality": modality,
                            "Dataset": dataset,
                            "Interal Role": roles["Reasoning"],
                            "External Role": roles["Answering"],
                            "Trials": list()
                            }

        result["Index"] = i
        if dataset == "aqua" or "mmlu" in dataset:
            result["Options"] = datum[2]["Options"]
        test_results["Trials"].append(result)
        write_json(filepath=output_path, data=test_results)


def scratchpad_test(datum: dict, model_one: str, model_two: str, modality: str, max_tokens: int):
    x, y, test_entry = datum

    prompt = build_prompt(question=x, modality=modality, use_simple_prompt=True,
                          bracket_extract=True)

    messages = [
        roles["Reasoning"],
        {"role": "user", "content": prompt}
    ]

    internal_message = multi_message_query(model=model_one, messages=messages, max_tokens=max_tokens)
    reasoning = internal_message["content"]
    print(f"Internal reasoning: {reasoning}")
    messages = [
        roles["Answering"],
        {"role": "user", "content": prompt},
        internal_message
    ]

    external_message = multi_message_query(model=model_two, messages=messages, max_tokens=max_tokens)
    response = external_message["content"]

    result = {"Index": test_entry["Index"], "Query": prompt, "Reasoning": reasoning, "Response": response,
              "GT": y}

    return result


