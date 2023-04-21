import json

from Datasets.JsonlDataset import JsonlDataset


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
        write_jsonl(filepath=filepath, data=by_num_terms[num], append=False)


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

    write_jsonl(filepath=save_path, data=output)


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

    write_jsonl(filepath=save_path, data=output)


def write_jsonl(filepath: str, data: list[dict], append: bool = False):
    # Writes jsonl data to the filepath. If append is true, will attempt to write it to
    # an existing file, and will create a new one otherwise.
    if append:
        write_mode = "a"
    else:
        write_mode = "w"
    with open(filepath, write_mode) as save_file:
        for line in data:
            processed = json.dumps(line) + '\n'
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
