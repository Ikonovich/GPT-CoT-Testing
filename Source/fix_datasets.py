import json
import os

directory_path = './Datasets/Stepwise_Extracted/Inconsequential/dataset_inconsequential.json'

# Load the first JSON file
with open(directory_path, 'r') as f:
    data1 = json.load(f)

# Create a dictionary from the first dataset, using "response" as the key
data1_dict = {entry['response']: entry for entry in data1}

# Get a list of all the JSON files in the second dataset's directory
second_dataset_files = os.listdir('./Datasets/Stepwise_Extracted/')

# For each file in the second dataset...
for filename in second_dataset_files:
    if(".json" in filename):
        # Load the file
        with open(f'./Datasets/Stepwise_Extracted/{filename}', 'r') as f:
            data2 = json.load(f)

        # For each entry in the second dataset...
        for entry in data2:
            response = entry['response']['response']

            # If the "response" exists in the first dataset...
            if response in data1_dict:
                # Add the "query" field to the corresponding entry in the first dataset
                data1_dict[response]['query'] = entry['response']['query']

# Convert the updated first dataset back to a list
updated_data1 = list(data1_dict.values())

# Save the updated first dataset
with open('./Datasets/Stepwise_Extracted/Inconsequential/inconsequential_fixed.json', 'w') as f:
    json.dump(updated_data1, f, indent=4, sort_keys=True)