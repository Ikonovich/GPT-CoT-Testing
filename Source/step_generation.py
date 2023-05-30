import json
import random

import os

from regex import regex

from utils.file_utils import read_json, write_json
from utils.query_utils import multi_message_query, timer


def generate_steps():
    directory_path = "Results/Primary_Test_Results/gpt-3.5-turbo/zero_shot_cot/stepwise"

    files = os.listdir(directory_path)
    # Skip 1step
    files = [path for path in files if "1step" not in path]

    for file in files:
        print("Starting step generation for path: " + file)
        output_path = f"Datasets/Stepwise_Extracted/Unmodified/responses_{file}"
        # Prepare a list to store the responses and set the starting index
        responses = []
        start_index = 0
        if os.path.exists(output_path):
            responses = read_json(output_path)
            start_index = responses[-1]["Index"]
        # Load the data
        with open(os.path.join(directory_path, file), 'r') as f:
            data = json.load(f)

        data = data["Trials"]
        # Filter the data for accurately answered questions
        filtered_data = [item for item in data if item['Final Answer'] == item['GT']]


        s = """
        {2 * 4 = 8}\n
        {3 + 8 = 11}\n
        {11 + 5 = 16}\n
        {Answer = 16}\n
        """

        # Get the index to start on
        # Send the queries to the GPT-3 model and store the responses
        delay = 1  # start delay at 1 second
        for i in range(len(filtered_data)):
            item = filtered_data[i]
            index = item["Index"]
            query = item['Query']
            answer = item['Final Answer']
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
                    entry["Answer"] = answer
                    entry["GT"] = gt
                    break
                except Exception as e:
                    print(str(e))
                    timer(delay)

        # Write responses to a JSON file
        with open(output_path, "w") as f:
            json.dump(responses, f, indent=4, sort_keys=True)

        print(f"Responses saved to responses_{file}.json")


def modify_steps():
    # Modify one and both numbers in each of the final steps, then remove the final step
    # (if more than 1 step) or the final answer to the step (if only 1 step)
    for i in range(1, 10):
        path = os.path.join("Datasets", "Stepwise_Extracted", "Unmodified", f"responses_Simple-in-brackets-gpt-3.5-turbo-zero_shot_cot-{i}step.json")
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
            query = query[:stop_index]
            results.append({
                "Index": index,
                "Question": query,
                "GT": gt,
                "New Steps": new_steps,
                "Original Steps": steps,
                "Steps Length": steps
            })

        write_json(filepath=f"Datasets\\Stepwise_Extracted\\Modified-Double-Val-Final\\{i}-step.json", data=results)


def modify(steps: list[str], step_index: int, num_modifications: int, remove_last: bool=True) -> list[str]:

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

    # If the list of steps is longer than 1 and remove last is true, remove the final step. Otherwise, remove the final answer
    # to the final step.
    if remove_last and len(steps) > 1:
        steps = steps[:-1]
    else:
        stop_index = steps[-1].index("=")
        steps[-1] = steps[-1][:stop_index + 1]


def extract_steps():

    for i in range(1, 10):
        path = r"Datasets\Stepwise_Extracted\Unmodified\responses_Simple-in-brackets-gpt-3.5-turbo-zero_shot_cot-" + str(i) + "step.json"
        results = list()

        data = read_json(path)

        # Iterate over each item in the data
        for item in data:

            index = item["Index"]
            query = item['Query']
            gt = item['GT']
            answer = item['Answer']
            response = item['Response']

            # Use a regular expression to find all substrings within squiggly brackets
            steps = regex.findall(r'\{([^}]*)\}', response)

            # Remove the "Answer =" step, and any steps containing only one value
            steps = [step for step in steps if "Answer" not in step]
            steps = [s for s in steps if len(regex.findall(r'-?\d+\.?\d*', s)) > 1]

            # Bypass this entry if no steps were successfully extracted
            if len(steps) == 0:
                continue
            entry = dict()
            entry["Index"] = index
            entry["Query"] = query
            entry["Response"] = response
            entry["Answer"] = answer
            entry["GT"] = gt
            entry["Steps"] = steps

            results.append(entry)

        write_json(filepath=path, data=results)


def map_indices():
    # Maps indices of modified questions to their original dataset question
    for i in range(1, 10):
        data_path = r"Datasets\Stepwise_Extracted\Unmodified\responses_Simple-in-brackets-gpt-3.5-turbo-zero_shot_cot-" + str(i) + "step.json"
        original_path = "Datasets\\Stepwise\\" + str(i) + r"-Step-Int-Formulae.json"
        results = list()

        data = read_json(data_path)
        questions = read_json(original_path)

        for trial in data:
            if "response" in trial:
                item = trial["response"]
                query = item['query']
                gt = item['gt']
                answer = item['answer']
                response = item['response']
                trial = {
                    "Query": query,
                    "Response": response,
                    "Answer": answer,
                    "GT": gt
                }

            query = trial["Query"]
            stop_index = query.index("=")
            query = query[:stop_index]

            found = False
            for entry in questions:
                question = entry["Question"]
                if query in question:
                    trial["Index"] = entry["Index"]
                    results.append(trial)
                    found = True
                    break
            if not found:
                raise ValueError("Didn't find the index!")

        write_json(filepath=data_path, data=results)


if __name__ == "__main__":
    generate_steps()
    # extract_steps()
    #modify_steps()