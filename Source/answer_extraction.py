from regex import regex

from config import RESULTS_FOLDER, dataset_index_map
from utils.file_utils import get_filepaths, read_json, write_json


def extract_answers(root: str | None = None):
    # Extracts answers from all test data
    if root is None:
        root = RESULTS_FOLDER

    data_paths = get_filepaths(root=root, contains=["json"], excludes=["Metadata"])

    # Iterate over every set of test result json files found under the root folder
    for i in range(len(data_paths)):
        # Get the data
        results = list()
        data_path = data_paths[i]
        data = read_json(data_path)

        modality = data["Modality"]
        dataset = data["Dataset"]
        extraction_type = data["Extraction Type"]

        data["Dataset Index"] = dataset_index_map[data["Dataset"]]

        trials = data["Trials"]
        # Iterate over every test and run answer extraction.
        i = 0
        for test in trials:
            # Get options if needed to match responses to letters
            if dataset == "aqua" or "mmlu" in dataset:
                options = test["Options"]
            else:
                options = None

            # Special extraction for Answer-First modality with two-stage answer extraction.
            #
            # We extract the first answer from the initial response and the final answer
            # from the extraction response, to allow distinguishing the model changing answers.
            if modality == "answer_first" and "two-stage" in extraction_type:
                response = test["Response"]
                final_response = test["Extract-Response"]

                _, front_pred = clean_answer(response=response, dataset=dataset,
                                             extraction_type=extraction_type,
                                             options=options)

                _, end_pred = clean_answer(response=final_response, dataset=dataset,
                                           extraction_type=extraction_type,
                                           options=options)

            else:
                if "two-stage" in extraction_type:
                    extraction_response = test["Extract-Response"]
                    end_pred, front_pred = clean_answer(response=extraction_response, dataset=dataset,
                                                        extraction_type=extraction_type,
                                                        options=options)
                    if end_pred == "":
                        response = test["Response"]
                        end_pred, front_pred = clean_answer(response=response, dataset=dataset,
                                                            extraction_type=extraction_type,
                                                            options=options)
                else:
                    response = test["Response"]
                    end_pred, front_pred = clean_answer(response=response, dataset=dataset,
                                                        extraction_type=extraction_type,
                                                        options=options)

            if modality == "answer_first":
                answer = front_pred
                final_answer = end_pred
            elif "two-stage" in extraction_type:
                answer = front_pred
                final_answer = front_pred
            else:
                answer = end_pred
                final_answer = end_pred

            test["Answer"] = answer
            test["Final Answer"] = final_answer
            results.append(test)
            i += 1
        data["Trials"] = results
        write_json(filepath=data_path, data=data)


def clean_answer(response: str, dataset: str, extraction_type: str, options: dict = None):
    # Returns two answers from the submitted response in a tuple formatted
    # (prediction with the highest index, prediction with the lowest index)
    # The answers are allowed to represent the same object if only one is detected in the item.

    # Look for anything between brackets in the response and
    # attempt to extract the answer.
    # If the modality is answer_first, attempt to extract it from the first and last match.
    # Else, extract it from the last match.
    if extraction_type == 'in-brackets' or extraction_type == "two-stage-style-two":
        brackets = [s for s in regex.findall(r'\{(?:[^{}]|(?R))*}', response)]

        if len(brackets) > 0:
            # Remove the brackets
            cleaned = list()
            for item in brackets:
                item = item.replace("{", "")
                item = item.replace("}", "")
                cleaned.append(item)
            brackets = cleaned
            # Extract two predictions from the brackets.
            # Takes the first and last items extracted from the brackets.
            # These items are allowed to be the same.
            # Then, runs them through answer extraction.
            # Then concatenates the results.
            results = _extract(data=brackets, dataset=dataset, options=options)
            if len(results) == 0:
                print('Bracket-based extraction failed. Falling back on full-response extraction.')
                results = _extract(data=response, dataset=dataset, options=options)
        else:
            print('Bracket-based extraction failed. Falling back on full-response extraction.')
            results = _extract(data=response, dataset=dataset, options=options)

    else:
        results = _extract(data=response, dataset=dataset, options=options)

    if len(results) > 0:
        # Get the first and last items from the results or a set of empty strings
        front_pred = results[0]
        end_prediction = results[-1]
    else:
        front_pred = ""
        end_prediction = ""

    return end_prediction, front_pred


def _extract(data: list[str] | str, dataset: str, options: dict = None) -> list:
    dataset = dataset.lower()
    # This function performs final extraction of possible answers from strings provided by the
    # answer_extraction function.
    # It runs linearly along each list index and string, such that answers
    # will be ordered by [list index][str index]

    results = list()

    if type(data) is str:
        data = [data]

    for sample in data:
        # Citation: Most of this basic method, except for the option-answer choice replacement
        # and returning of lists of possible predictions instead of only one,
        # as well as some of this code, is taken from
        # citation:
        # "Large Language Models are Zero-Shot Reasoners"
        # Kojima et al., 2022
        # Advances in Neural Information Processing Systems, pages 22199--22213
        # DOI 10.48550/arXiv.2205.11916

        if dataset == "aqua" or "mmlu" in dataset:
            # Look for a response of "N/A"
            pred = regex.findall(r"N/A", sample, regex.IGNORECASE)
            if len(pred) == 0:
                # If no N/A was found, look for a single ABCDE.
                pred = regex.findall('[ABCDE]', sample)
            if len(pred) == 0:
                # Finally, try to map an option to a result
                sample = sample.split(" ")
                for substr in sample:
                    for key, value in options.items():
                        if value in substr:
                            pred.append(options[key])

        elif dataset in ['multiarith', 'gsm8k'] or 'step' in dataset or "modified" in dataset:
            # Remove commas and find an arbitrary length integers or float
            pred = sample.replace(',', '')
            pred = [s for s in regex.findall(r'-?\d+\.?\d*', pred)]
            pred = [float(x) for x in pred]

        # Find yes or no in the extracted sections. If none is found,
        # look for the last yes or no in the full response sentence.
        elif dataset in ["coin_flip"]:
            pred = sample.lower()
            pred = regex.sub(r"\"|\'|\n|\.|\s|\:|\,", " ", pred)  # Remove unnecessary chars
            pred = pred.split()
            # Find the last index of the words "yes" and "no" in the string. Take whichever is higher as the answer.
            pred = [i for i in pred if i in ("yes", "no")]

        else:
            raise ValueError("The given dataset has not been defined.")

        # Remove trailing periods
        for i in range(len(pred)):
            if type(pred[i]) is str and pred[i][-1] == ".":
                pred[i] = pred[i][:-1]

        results += pred

    return results
