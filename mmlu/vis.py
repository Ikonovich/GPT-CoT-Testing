import json
import os
import matplotlib.pyplot as plt

# Folder containing your json datasets
folder_path = './generated/gpt4/cleaned/answers/'
save_folder_path = './figures2/'
# Collect all json files
json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]

# Store the accuracies for each dataset
accuracies = {}

# Process each json file
for json_file in json_files:
    with open(os.path.join(folder_path, json_file), 'r') as f:
        data = json.load(f)
        total = len(data)
        correct = sum(1 for item in data if item['A'] == item['GT'])
        accuracy = correct / total
        accuracies[json_file[:-7]] = accuracy

# Visualize each accuracy separately
for dataset, accuracy in accuracies.items():
    plt.figure()
    plt.bar(dataset, accuracy)
    plt.title(f'Accuracy for {dataset}')
    plt.xlabel('Dataset')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)  # set y-axis limits to 0-1 for consistency
    #plt.show()
    plt.savefig(os.path.join(save_folder_path, f'{dataset}_accuracy.png'))  # save the figure
    plt.close()  # close the figure to free up memory

# Visualize all accuracies together
plt.figure()
plt.bar(accuracies.keys(), accuracies.values())
plt.title('Accuracies for all datasets')
plt.xlabel('Dataset')
plt.ylabel('Accuracy')
plt.ylim(0, 1)  # set y-axis limits to 0-1 for consistency
#plt.show()
plt.savefig(os.path.join(save_folder_path, 'all_accuracies.png'))  # save the figure
plt.close()  # close the figure to free up memory