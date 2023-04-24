import copy
import json
import os

import config
from Datasets.JsonlDataset import JsonlDataset


def format_name(name: str, is_model: bool = False) -> str:
    # Formats a name to fit the format
    # Item1 Item2 Item3, etc
    # Basically, capitalizes the first letter of each word and removes spaces.
    # If it's a model, capitalizes "GPT" separately if found.
    if "-" in name:
        name = name.split("-")
    elif "_" in name:
        name = name.split("_")
    else:
        name = name.split()

    new_name = list()
    for word in name:
        if is_model and word == "gpt":
            new_word = word[:3].upper() + word[3:]
        else:
            new_word = word[:1].upper() + word[1:]
        new_name.append(new_word)

    new_name = " ".join(new_name)
    return new_name


def collate_results(model: str) -> tuple[dict, dict]:
    # Collates all results for a specific model
    # Sorts into a dictionary by dataset
    start = os.path.join(config.RESULTS_FOLDER, model)
    contains = ["Metadata"]
    paths = get_filepaths(start, contains)

    metadata = list()
    for path in paths:
        with open(path) as file:
            entry = json.load(file)
            metadata.append((path, entry))

    # Store results by dataset
    simple_prompt = dict()
    initial_prompt = dict()

    initial_prompt["multiarith"] = dict()
    initial_prompt["gsm8k"] = dict()
    initial_prompt["stepwise"] = dict()

    simple_prompt["multiarith"] = dict()
    simple_prompt["gsm8k"] = dict()
    simple_prompt["stepwise"] = dict()

    # Create list for stepwise data
    for modality in config.modalities:
        initial_prompt["stepwise"][modality] = [None for i in range(0, 9)]
        simple_prompt["stepwise"][modality] = [None for i in range(0, 9)]

    for entry in metadata:
        path = entry[0]
        modality = entry[1]["Modality"]
        dataset = get_dataset(path)
        if "SimplePrompt" in path:
            if "step" in dataset:
                index = int(dataset[:1]) - 1
                step_data = entry[1]
                step_data["Step Count"] = index + 1
                simple_prompt["stepwise"][modality][index] = step_data
            else:
                simple_prompt[dataset][modality] = entry[1]
        else:
            if "step" in dataset:
                index = int(dataset[:1]) - 1
                step_data = entry[1]
                step_data["Step Count"] = index + 1
                initial_prompt["stepwise"][modality][index] = step_data
            else:
                initial_prompt[dataset][modality] = entry[1]

    return initial_prompt, simple_prompt


def get_cross_modality_results(data: dict[str, dict]) -> tuple[list[str], list[float]]:
    # Used to generate cross-modality results for different datasets
    # Data should contain a dictionary of keys corresponding to modality names, with values
    # being dictionaries containing an "Accuracy" key consisting of float values.
    labels = list()
    accuracies = list()

    for modality in data:
        mod_name = format_name(name=modality)
        labels.append(modality)
        accuracies.append(data[modality]["Accuracy"])

    return labels, accuracies


def get_dataset(path: str) -> str:
    # Returns the dataset a file path contains information about
    for key in config.datasets:
        if key in path:
            return key


def get_filepaths(start_path: str, contains: list[str]) -> list:
    # Recursively walks from the start directory and returns a list of all file paths
    # containing every string in contains.
    results = list()
    for path in os.listdir(start_path):
        full_path = os.path.join(start_path, path)
        if os.path.isdir(full_path):
            results.extend(get_filepaths(full_path, contains))
        elif check_path(path=full_path, contains=contains):
            results.append(full_path)

    return results


def check_path(path: str, contains: list[str]) -> bool:
    for substr in contains:
        if substr not in path:
            return False
    return True


def sort_by_term_count(dataset: JsonlDataset, discrim_key, base_path: str):
    # Takes a jsonldataset and sorts it by how many times
    # mathematical operators (+, -, *. /" are in << >> annotated sections in the answer.

    # Store discriminators
    discriminators = {"%", "+", "-", "*", "/"}

    data = dataset.data

    # Store items by term count by term: sample
    by_num_terms = dict()

    for entry in data:
        entry["Operators"] = ""
        count = 0
        field = entry[discrim_key]

        # Get the indices of the annotated areas
        # + 2 to skip over the first < or ( and because the first index after should never be an operator
        indices = [i + 2 for i in range(len(field)) if field.startswith("<<", i)]
        indices += [i + 2 for i in range(len(field)) if field.startswith("(", i)]

        for index in indices:
            count = 0
            # Go letter by letter until we hit a > bracket.
            i = index
            char = field[i]
            while char != ">" and char != ")" and i < len(field):
                char = field[i]
                if char in discriminators:
                    count += 1
                    entry["Operators"] += char
                i += 1

        # Add the sample to the appropriate term count dataset
        count = len(entry["Operators"])
        if count in by_num_terms:
            by_num_terms[count].append(entry)
        else:
            by_num_terms[count] = list()
            by_num_terms[count].append(entry)

    # Finally, save the datasets
    for num in by_num_terms:
        filepath = base_path + "-" + str(num) + "-Terms.jsonl"
        write_json(filepath=filepath, data=by_num_terms[num], append=False)


def process_gsm8k(read_path: str = "Datasets/GSM8K/test.jsonl", save_path: str = "Datasets/GSM8K/GSM8K-Processed.jsonl"):

    output = list()
    with open(read_path) as file:
        lines = file.readlines()

    for line in lines:
        j_str = json.loads(line)
        question = j_str["question"]
        answer = j_str["answer"]
        gt = answer[answer.index("####") + 4:].strip()

        output.append({"Question": question, "Answer": answer, "Ground Truth": gt})

    write_json(filepath=save_path, data=output)


def process_multiarith(read_path: str = "Datasets/MultiArith/MultiArith.json",
                       save_path: str = "Datasets/MultiArith/MultiArith-Processed.jsonl"):
    output = list()
    with open(read_path) as file:
        entries = json.load(file)

    for entry in entries:
        index = entry["iIndex"]
        question = entry["sQuestion"]
        answer = entry["lSolutions"][0]

        output.append({"Index": index, "Question": question, "Ground Truth": answer})

    write_json(filepath=save_path, data=output)


def write_json(filepath: str, data: list[dict] | dict, append: bool = False):
    # Writes json list data to the filepath. If append is true, will attempt to write it to
    # an existing file, and will create a new one otherwise.
    if append:
        write_mode = "a"
    else:
        write_mode = "w"
    with open(filepath, write_mode) as save_file:
        if type(data) is list:
            for line in data:
                processed = json.dumps(line) + '\n'
                save_file.write(processed)
        else:
            processed = json.dumps(data, indent=4)
            save_file.write(processed)


def load_dataset(path: str) -> JsonlDataset:
    data = list()
    filepath = path
    with open(filepath) as file:
        lines = file.readlines()

    for line in lines:
        j_dict = json.loads(line)
        data.append(j_dict)

    dataset = JsonlDataset(data=data)
    return dataset
