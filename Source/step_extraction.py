import json
import re
import os
import time
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
            gt = item['response']['gt']
            answer = item['response']['answer']

            # Use a regular expression to find all substrings within squiggly brackets
            steps_in_item = re.findall(r'\{([^}]*)\}', response_str)

            # Randomize inconsequential and consequential numbers for each step
            steps_inconsequential = []
            steps_consequential = []

            for step in steps_in_item[:-1]:
                # Find all numbers in the step
                numbers = re.findall(r'\b\d+\b', step)

                if numbers:
                    # Select a random number from the step
                    inconsequential_number = random.choice(numbers)

                    # Replace the selected number with a random number
                    step_inconsequential = step.replace(inconsequential_number, str(random.randint(1, 10)), 1)
                    steps_inconsequential.append(step_inconsequential)

                    # Replace the result (right-hand side) of the step with a random number
                    step_consequential = re.sub(r'=(.*)', '= ' + str(random.randint(1, 10)), step)
                    steps_consequential.append(step_consequential)

            # Create dictionaries to add to the datasets
            item_dataset = {"response": response_str, "steps": steps_in_item[:-1], "answer": answer, "gt": gt}
            item_dataset_inconsequential = {"response": response_str, "steps": steps_inconsequential, "answer": answer, "gt": gt}
            item_dataset_consequential = {"response": response_str, "steps": steps_consequential, "answer": answer, "gt": gt}

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
