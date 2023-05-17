import json
import math
import os

import pandas as pd
from regex import regex

from config import RESULTS_FOLDER, z_val, METADATA_FOLDER

from file_utils import get_filepaths, read_json, write_json


def search_metadata(model: str = None, modalities: list[str] = None, datasets: list[str] = None):
    metapath = os.path.join(METADATA_FOLDER, "Test Results.csv")
    frame = pd.DataFrame.from_csv(metapath)

    if model is not None:
        frame = frame[frame.Model == model]

    if modalities is not None:
        frame = frame[frame.Modality.isin(modalities)]

    if datasets is not None:
        frame = frame[frame.Dataset.isin(datasets)]
    return frame


def generate_metadata(root: str = None):
    if root is None:
        root = RESULTS_FOLDER

    data_paths = get_filepaths(root=root, contains=["json"])

    results = list()
    for i in range(len(data_paths)):
        test_path = data_paths[i]
        metadata = path_to_metadata(test_path=test_path)
        stats = test_quantification(test_path=test_path)
        metadata.update(stats)
        results.append(metadata)

    # Update the metadata file
    frame = pd.DataFrame.from_records(data=results)
    metapath = os.path.join(METADATA_FOLDER, "Test Results.csv")
    frame.to_csv(metapath)


def path_to_metadata(test_path: str) -> dict[str, str]:
    # Generates metadata files from provided test results and test path.

    steps = None
    metadata = dict()
    # get prompt style
    if "simple" in test_path:
        simple_prompt = "True"
    else:
        simple_prompt = False

    # Get the model
    if "gpt-4-32k" in test_path:
        model = "gpt-4-32k"
    elif "gpt-4" in test_path:
        model = "gpt-4"
    elif "gpt-3.5" in test_path:
        model = "gpt-3.5"
    elif "davinci-002" in test_path:
        model = "text-davinci-002"
    else:
        raise ValueError

    # Get the extraction type
    if "in-brackets" in test_path:
        extraction_type = "in-brackets"
    elif "two-stage" in test_path:
        extraction_type = "two-stage"
    else:
        raise ValueError

    # Get the modality
    if "zero_shot_cot" in test_path:
        modality = "zero_shot_cot"
    elif "answer_first" in test_path:
        modality = "answer_first"
    elif "explanation_first" in test_path:
        modality = "explanation_first"
    elif "suppressed" in test_path:
        modality = "suppressed_cot"
    elif "zero_shot" in test_path:
        modality = "zero_shot"
    else:
        raise ValueError

    # Get the dataset
    if "aqua" in test_path:
        dataset = "aqua"
    elif "gsm8k" in test_path:
        dataset = "gsm8k"
    elif "multiarith" in test_path:
        dataset = "multiarith"
    elif "mmlu" in test_path:
        dataset = "mmlu"
    elif "coin_flip" in test_path:
        dataset = "coin_flip"
    elif "step" in test_path:
        dataset = "stepwise"
        steps = regex.findall(r"\d{1,2}step", test_path)[0]
        steps = regex.findall(r"\d{1,2}", steps)[0]
        steps = int(steps)
        metadata["Steps"] = steps
    else:
        raise ValueError

    metadata.update({"Dataset": dataset,
                     "Modality": modality,
                     "Model": model,
                     "Extraction Type": extraction_type,
                     "Simple Prompt": simple_prompt,
                     "Test Path": test_path})

    return metadata


def count_cot(data: list[dict], dataset: str) -> tuple[int, int, int, int, int, int]:
    total = 0
    total_accurate = 0
    cot_total = 0
    cot_accurate = 0
    non_cot_total = 0
    non_cot_accurate = 0

    for entry in data:
        total += 1
        is_accurate = 0
        response = entry["Response"].lower()
        answer = entry["Answer"]
        gt = entry["GT"]

        # Get the accuracy value
        try:
            if dataset == "aqua" or dataset == "coin_flip" or dataset == "mmlu":
                if answer.lower() == gt.lower():
                    is_accurate = 1
                    total_accurate += 1
            elif float(answer) == float(gt):
                is_accurate = 1
                total_accurate += 1
        except ValueError:
            is_accurate = 0

        # Track CoT presence
        if "step" in dataset:
            # For stepwise datasets, we define CoT to be the length of the original question + 20 chars.
            question = entry["Query"][0: entry["Query"].index("=")]
            cutoff = len(question) + 20

            if len(response) > cutoff:
                cot_total += 1
                cot_accurate += is_accurate
            else:
                non_cot_total += 1
                non_cot_accurate += is_accurate

        elif dataset == "coin_flip":
            cutoff = 60
            if len(response) > cutoff and "no one" not in response and "none of" not in response \
                    and "unknown" not in response:
                cot_total += 1
                cot_accurate += is_accurate
            else:
                non_cot_total += 1
                non_cot_accurate += is_accurate
        else:
            cutoff = 60
            if len(response) > cutoff:
                is_cot = True
                cot_total += 1
                cot_accurate += is_accurate
            else:
                non_cot_total += 1
                non_cot_accurate += is_accurate

    return total, total_accurate, cot_accurate, cot_total, non_cot_accurate, non_cot_total


def test_quantification(test_path: str = None):
    # Quantifies results for all tests.

    data = read_json(test_path)

    # Initialize metadata
    metadata = path_to_metadata(test_path=test_path)
    # Calculate total accuracy, % of answers containing Chain-of-Thought reasoning,
    # the accuracy of CoT answers, and the accuracy of Non-CoT answers,
    # along with over all counts of each item type.
    total, total_accurate, cot_accurate, cot_total, non_cot_accurate, non_cot_total = count_cot(
        data=data, dataset=metadata["Dataset"])

    if cot_total == 0:
        cot_accuracy = 0
    else:
        cot_accuracy = (cot_accurate / cot_total)

    if non_cot_total == 0:
        non_cot_accuracy = 0
    else:
        non_cot_accuracy = (non_cot_accurate / non_cot_total)

    cot_percent = (cot_total / total)

    total_accuracy = (total_accurate / total)
    ci_radius = (z_val * math.sqrt((total_accuracy * (1 - total_accuracy)) / total))

    return {"Total": total,
            "Total Accurate": total_accurate,
            "Accuracy": total_accuracy,
            "Percent of Answers Containing CoT": cot_percent,
            "CoT Accuracy": cot_accuracy,
            "Non-CoT Accuracy": non_cot_accuracy,
            "ci_radius": ci_radius,
            "ci_upper": total_accuracy + ci_radius,
            "ci_lower": total_accuracy - ci_radius}


def process_mmlu():
    gpt35_path = r"C:\Users\evanh\Documents\GPT-CoT-Testing\mmlu\generated\gpt35-turbo\2phase"
    gpt4_path = r"C:\Users\evanh\Documents\GPT-CoT-Testing\mmlu\generated\gpt4\2phase"

    for path in os.listdir(gpt4_path):

        if "college" in path:
            discriminator = "college"
        elif "combined" in path:
            discriminator = "combined"
        else:
            discriminator = "high-school"

        if "zero-shot-cot" in path:
            modality = "zero_shot_cot"
        elif "answer-first" in path:
            modality = "answer_first"
        elif "explanation-first" in path:
            modality = "explanation_first"
        elif "supressed" in path:
            modality = "suppressed_cot"
        else:
            modality = "zero_shot"

        results = list()
        filepath = os.path.join(gpt4_path, path)
        with open(filepath, encoding='utf-8') as file:
            data = file.read()
            data = json.loads(data)

        for item in data:
            options = item["O"]
            options = dict((v, k) for k, v in options.items())

            new_item = {"Query": item["Q"],
                        "Response": item["R"],
                        "Extract-Response": item["A"],
                        "Options": options,
                        "GT": item["GT"]}

            results.append(new_item)

        file_name = f"Simple-two-stage-gpt-4-{modality}-mmlu-{discriminator}.json"
        write_json(data=results, filepath=os.path.join(RESULTS_FOLDER, "gpt-4", modality,
                                                       "mmlu", file_name))

    for path in os.listdir(gpt35_path):

        if "college" in path:
            discriminator = "college"
        elif "combined" in path:
            discriminator = "combined"
        else:
            discriminator = "high-school"

        if "zero-shot-cot" in path:
            modality = "zero_shot_cot"
        elif "answer-first" in path:
            modality = "answer_first"
        elif "explanation-first" in path:
            modality = "explanation_first"
        elif "supressed" in path:
            modality = "suppressed_cot"
        else:
            modality = "zero_shot"

        results = list()
        filepath = os.path.join(gpt35_path, path)
        with open(filepath, encoding='utf-8') as file:
            data = file.read()
            data = json.loads(data)

        for item in data:
            options = item["O"]
            options = dict((v, k) for k, v in options.items())

            new_item = {"Query": item["Q"],
                        "Response": item["R"],
                        "Extract-Response": item["A"],
                        "Options": options,
                        "GT": item["GT"]}

            results.append(new_item)
        file_name = f"Simple-two-stage-gpt-35-{modality}-mmlu-{discriminator}.json"

        write_json(data=results, filepath=os.path.join(RESULTS_FOLDER, "gpt-3.5-turbo",
                                                       modality, "mmlu", file_name))
