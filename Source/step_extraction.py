import json
import re
import os
import random

directory_path = './Datasets/Stepwise_Extracted/'

files = os.listdir(directory_path)

datasets = []
# Create the two new datasets
dataset_inconsequential = []
dataset_consequential = []

for file in files:
    if(".json" in file):
        # Load the data
        with open(os.path.join(directory_path, file), 'r') as f:
            data = json.load(f)

        # Iterate over each item in the data
        for item in data:
            # Get the response string
            response_str = item['response']['response']
            query_str = item['response']['query']
            gt = item['response']['gt']
            answer = item['response']['answer']

            # Use a regular expression to find all substrings within squiggly brackets
            steps_in_item = re.findall(r'\{([^}]*)\}', response_str)

            # Remove the "Answer =" step
            steps_in_item = [step for step in steps_in_item if "Answer" not in step]

            # Randomize inconsequential and consequential numbers for each step
            steps_inconsequential = []
            steps_consequential = []

            for index, step in enumerate(steps_in_item):
                # Find all numbers in the step
                numbers = re.findall(r'\b\d+\b', step)

                if numbers:
                    # For consequential steps, remove the right-hand side of the equation
                    # in the step before the final one and randomize values only on the final step
                    if index == len(steps_in_item) - 1:
                        # Remove the right-hand side of the equation
                        step = re.sub(r'=(.*)', '=', step)
                        # Randomize the left hand side of the equation
                        lhs_numbers = re.findall(r'^(.*?)=', step)
                        if lhs_numbers:
                            lhs_randomized = re.sub(r'\b\d+\b', str(random.randint(1, 10)), lhs_numbers[0], 1)
                            step_consequential = lhs_randomized + "="
                        else:
                            step_consequential = step
                    else:
                        step_consequential = step
                    steps_consequential.append(step_consequential)

                    # For inconsequential steps, remove the right-hand side of the equation
                    # in the step before the final one and randomize one number on any step
                    # For inconsequential steps, remove the right-hand side of the equation only in the final step
                    # and randomize one number on any step
                    if index != len(steps_in_item) - 1:
                        lhs_numbers = re.findall(r'^(.*?)=', step)
                        if lhs_numbers:
                            lhs_randomized = re.sub(r'\b\d+\b', str(random.randint(1, 10)), lhs_numbers[0], 1)
                            step_inconsequential = lhs_randomized + "=" + re.findall(r'=(.*)', step)[0]
                        else:
                            step_inconsequential = step
                    else:
                        # Remove the right-hand side of the equation in the final step
                        step_inconsequential = re.sub(r'=(.*)', '=', step)

                    steps_inconsequential.append(step_inconsequential)

            # Create dictionaries to add to the datasets
            item_dataset = {"query":query_str, "response": response_str, "steps": steps_in_item, "answer": answer, "gt": gt}
            item_dataset_inconsequential = {"query":query_str, "response": response_str, "steps": steps_inconsequential, "answer": answer, "gt": gt}
            item_dataset_consequential = {"query":query_str, "response": response_str, "steps": steps_consequential, "answer": answer, "gt": gt}

            # Add these dictionaries to the datasets
            datasets.append(item_dataset)
            dataset_inconsequential.append(item_dataset_inconsequential)
            dataset_consequential.append(item_dataset_consequential)

# Write the new datasets to JSON files
with open('./Datasets/Stepwise_Extracted/Consequential/dataset_consequential.json', 'w') as f:
    json.dump(dataset_consequential, f, indent=4)

with open('./Datasets/Stepwise_Extracted/Inconsequential/dataset_inconsequential.json', 'w') as f:
    json.dump(dataset_inconsequential, f, indent=4)

with open('./Datasets/Stepwise_Extracted/datasets.json', 'w') as f:
    json.dump(datasets, f, indent=4)
