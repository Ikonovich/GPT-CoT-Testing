import json
import random

import os

from regex import regex

from config import DATASETS, DATASET_FOLDER
from utils.file_utils import read_json, write_json, load_dataset
from utils.query_utils import multi_message_query, timer


def generate_steps():

    for i in range(1, 10):
        tests_35_path = f"Results/Primary_Test_Results/gpt-3.5-turbo/zero_shot_cot/stepwise/Simple-in-brackets-gpt-3.5-turbo-zero_shot_cot-{i}step.json"
        tests_4_path = f"Results/Primary_Test_Results/gpt-4/zero_shot_cot/stepwise/Simple-in-brackets-gpt-4-zero_shot_cot-{i}step.json"
        output_path = f"Datasets/Stepwise_Extracted/Unmodified/steps_unmodified_{i}step.json"
        print(f"Starting step generation for step {i}")

        # Prepare a list to store the responses and set the starting index
        responses = []
        start_index = 0
        if os.path.exists(output_path):
            responses = read_json(output_path)
            start_index = responses[-1]["Index"]
        # Load the questions
        dataset = str(i) + "step"
        questions = load_dataset(os.path.join(DATASET_FOLDER, DATASETS[dataset]))

        # Load the trials
        tests_4 = read_json(filepath=tests_4_path)
        tests_35 = read_json(filepath=tests_35_path)
        trials_4 = tests_4["Trials"]
        trials_35 = tests_35["Trials"]
        # Filter the data for accurately answered questions
        filtered_data = list()
        for j in range(len(questions)):
            # Get the test question and its results from zero_shot_cot modality on both models.
            _, _, test = questions[j]
            trial_4 = trials_4[j]
            trial_35 = trials_35[j]

            # Validate that the queries match the questions
            if test["Index"] != j or test["Index"] != trial_4["Index"] or test["Index"] != trial_35["Index"]:
                raise ValueError("Trials out of order.")
            if test["Question"][:5] != trial_4["Query"][:5] or test["Question"][:5] != trial_35["Query"][:5]:
                raise ValueError("Trial mismatch detected.")

            # Check accuracy. If either trial on either model is inaccurate, throw it out.
            gt = float(test["GT"])
            if gt == float(trial_4["Answer"]) and gt == float(trial_35["Answer"]):
                filtered_data.append(test)

        # This is a formatting example to provide to the model
        s = """
        {2 * 4 = 8}\n
        {3 + 8 = 11}\n
        {11 + 5 = 16}\n
        {Answer = 16}\n
        """

        # Get the index to start on
        # Send the queries to the GPT-3 model and store the responses
        delay = 10 # Time to wait after a failure
        for j in range(start_index, len(filtered_data)):
            item = filtered_data[j]
            index = item["Index"]
            query = item['Question']
            gt = item['GT']
            while True:
                try:
                    messages = [
                        {"role": "system", "content": (
                                    "You are a math problem-solving machine. You take in simple arithmetic problems, "
                                    "and output each step required to solve this arithmetic problem in order of "
                                    "operations. Show each mathematical step one by one, and place each step in "
                                    "squiggly brackets. Do not provide any additional explanations. For example, if "
                                    "the problem is 3 + 2 * 4 + 5, the output should be: " + s)},
                        {"role": "user", "content": str(query)}
                    ]
                    response = multi_message_query(model="gpt-3.5-turbo", messages=messages, max_tokens=2000)
                    entry = dict()
                    entry["Index"] = index
                    entry["Query"] = query
                    entry["Response"] = response
                    entry["GT"] = gt
                    responses.append(entry)
                    break
                except Exception as e:
                    print(str(e))
                    timer(delay)

            # Write responses to a JSON file
            write_json(filepath=output_path, data=responses)


def modify_and_remove_final_step():
    # Modify one and both numbers in each of the final steps, then remove the final step
    # (if more than 1 step) or the final answer to the step (if only 1 step)
    for i in range(1, 10):
        path = os.path.join("Datasets", "Stepwise_Extracted", "Unmodified",
                            f"responses_Simple-in-brackets-gpt-3.5-turbo-zero_shot_cot-{i}step.json")
        data = read_json(path)
        results = list()

        for entry in data:
            index = entry["Index"]
            query = entry['Query']
            steps = entry['Steps']
            answer = entry['Answer']
            gt = entry['GT']

            new_steps = modify(steps, -1, 1)

            # Cut off the query after the first equal sign
            stop_index = query.index("=")
            query = query[:stop_index + 1]
            results.append({
                "Index": index,
                "Question": query,
                "GT": gt,
                "New Steps": new_steps,
                "Original Steps": steps,
                "Steps Length": steps
            })

        write_json(filepath=f"Datasets\\Stepwise_Extracted\\Modified-Single-Val-Final\\{i}-step.json", data=results)

    # Double val modification with final step removal
    for i in range(1, 10):
        path = os.path.join("Datasets", "Stepwise_Extracted", "Unmodified", f"responses_Simple-in-brackets-gpt-3.5-turbo-zero_shot_cot-{i}step.json")
        data = read_json(path)
        results = list()

        for entry in data:
            index = entry["Index"]
            query = entry['Query']
            steps = entry['Steps']
            gt = entry["GT"]
            new_steps = modify(steps, -1, 2)

            # Cut off the query after the first equal sign
            stop_index = query.index("=")
            query = query[:stop_index + 1]
            results.append({
                "Index": index,
                "Question": query,
                "GT": gt,
                "New Steps": new_steps,
                "Original Steps": steps,
                "Steps Length": steps
            })

        write_json(filepath=f"Datasets\\Stepwise_Extracted\\Modified-Off-by-One-Val-Final\\{i}-step.json", data=results)


def modify_and_remove_off_by_one():
    # Modify one and both numbers in each of the final steps, then remove the final step
    # (if more than 1 step) or the final answer to the step (if only 1 step)
    for i in range(1, 10):
        path = os.path.join("Datasets", "Stepwise_Extracted", "Unmodified",
                            f"responses_Simple-in-brackets-gpt-3.5-turbo-zero_shot_cot-{i}step.json")
        data = read_json(path)
        results = list()

        for entry in data:
            index = entry["Index"]
            query = entry['Query']
            steps = entry['Steps']
            answer = entry['Answer']
            gt = entry['GT']

            new_steps = modify(steps, -1, 1, off_by_one=True)

            # Cut off the query after the first equal sign
            stop_index = query.index("=")
            query = query[:stop_index + 1]
            results.append({
                "Index": index,
                "Question": query,
                "GT": gt,
                "New Steps": new_steps,
                "Original Steps": steps,
                "Steps Length": steps
            })

        write_json(filepath=f"Datasets\\Stepwise_Extracted\\Modified-Single-Val-Final\\{i}-step.json", data=results)

    # Double val modification with final step removal
    for i in range(1, 10):
        path = os.path.join("Datasets", "Stepwise_Extracted", "Unmodified", f"responses_Simple-in-brackets-gpt-3.5-turbo-zero_shot_cot-{i}step.json")
        data = read_json(path)
        results = list()

        for entry in data:
            index = entry["Index"]
            query = entry['Query']
            steps = entry['Steps']
            gt = entry["GT"]
            new_steps = modify(steps, -1, 2, off_by_one=True)

            # Cut off the query after the first equal sign
            stop_index = query.index("=")
            query = query[:stop_index + 1]
            results.append({
                "Index": index,
                "Question": query,
                "GT": gt,
                "New Steps": new_steps,
                "Original Steps": steps,
                "Steps Length": steps
            })

        write_json(filepath=f"Datasets\\Stepwise_Extracted\\Modified-Off-by-One-Double-Val-Final\\{i}-step.json", data=results)


def modify_and_keep_final_step():
    for i in range(1, 10):
        path = os.path.join("Datasets", "Stepwise_Extracted", "Unmodified",
                            f"responses_Simple-in-brackets-gpt-3.5-turbo-zero_shot_cot-{i}step.json")
        data = read_json(path)
        results = list()

        for entry in data:
            index = entry["Index"]
            query = entry['Query']
            steps = entry['Steps']
            gt = entry["GT"]
            new_steps = modify(steps, -1, 2, remove_last=False)

            # Cut off the query after the first equal sign
            stop_index = query.index("=")
            query = query[:stop_index + 1]
            results.append({
                "Index": index,
                "Question": query,
                "GT": gt,
                "New Steps": new_steps,
                "Original Steps": steps,
                "Steps Length": steps
            })

        write_json(filepath=f"Datasets\\Stepwise_Extracted\\Modified-Final-Step-Double\\{i}-step.json", data=results)


def modify(steps: list[str], step_index: int, num_modifications: int, remove_last: bool = True, off_by_one: bool = False) -> list[str]:
    # Modifies the provided index at the provided step.
    # If off_by_one is true, randomly selects -1 or 1 and adds that to the selected value.
    # Otherwise, sets a limit between the 2 * the original number and zero and selects a random value,
    # excluding the original value as a possibility.
    if num_modifications > 2 or num_modifications < 1:
        raise ValueError("Number of values to be modified must be either 1 or 2.")

    step = steps[step_index]
    vals = [s for s in regex.findall(r'-?\d+\.?\d*', step)]
    vals = [int(x) for x in vals]

    for i in range(num_modifications):
        # If there's only one modification to be made, randomize its location.
        if num_modifications == 1:
            index = random.randint(0, 1)
        else:
            index = i
        num = vals[index]
        # Replace the chosen number with a value between 0 and 2 * the number inclusive.
        # If the number is zero, choose a number between 0 and 9 inclusive.
        # Exclude the original number.
        if off_by_one:
            offset = random.choice([-1, 1])
            new_num = num + offset
        else:
            if num == 0:
                lim = 9
            else:
                lim = 2 * num
            new_num = None
            while new_num is None:
                if lim < 0:
                    new_val = random.randint(lim, 0)
                else:
                    new_val = random.randint(0, lim)
                if new_val != num:
                    new_num = new_val

        new_steps = list()
        pattern = f"\\b({num})\\b"
        for entry in steps:
            new_step = regex.sub(pattern=pattern, repl=f"{str(new_num)}", string=entry)
            new_steps.append(new_step)
        steps = new_steps

    # If the list of steps is longer than 1 and remove last is true, remove the final step. Otherwise, remove the
    # final answer to the final step.
    if remove_last and len(steps) > 1:
        steps = steps[:-1]
    else:
        stop_index = steps[-1].index("=")
        steps[-1] = steps[-1][:stop_index + 1]

    return steps


def extract_steps():
    for i in range(1, 10):
        path = f"Datasets/Stepwise_Extracted/Unmodified/steps_unmodified_{i}step.json"
        results = list()

        data = read_json(path)

        # Iterate over each item in the data
        print(f"Step count: {i} Length: {len(data)}")
        for item in data:
            response = item['Response']

            # Use a regular expression to find all substrings within squiggly brackets
            steps = regex.findall(r'\{([^}]*)\}', response)

            # Remove the "Answer =" step, and any steps containing only one value
            steps = [step for step in steps if "Answer" not in step]
            steps = [s for s in steps if len(regex.findall(r'-?\d+\.?\d*', s)) > 1]

            # Bypass this entry if no steps were successfully extracted
            if len(steps) == 0:
                continue

            item["Steps"] = steps

            results.append(item)

        write_json(filepath=path, data=results)


if __name__ == "__main__":
    # filter_questions()
    # generate_steps()
    extract_steps()
    # modify_and_keep_final_step()
