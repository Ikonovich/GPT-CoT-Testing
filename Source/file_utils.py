import json
import os
import pathlib

from Datasets.JsonlDataset import JsonlDataset
from config import METADATA_FOLDER


def generate_metadata_path(model: str, modality: str, extraction_type: str, dataset: str, steps: int | None = None,
                           meta_folder: str = None):

    if meta_folder is None:
        meta_folder = METADATA_FOLDER

    if dataset == "stepwise":
        if steps is not None:
            filename = f"{modality}-stepwise-{steps}steps-Metadata.json"
            meta_path = os.path.join(meta_folder, model, extraction_type, modality, "stepwise", filename)
        else:
            raise ValueError("Number of steps must be supplied when saving metadata for a stepwise dataset test.")
    else:
        filename = f"{modality}-{dataset}-Metadata.json"
        meta_path = os.path.join(meta_folder, model, extraction_type, modality, filename)

    return meta_path


def get_filepaths(root: str, contains: list[str] | None = None, excludes: list[str] | None = None) -> list:
    # Recursively walks from the start directory and returns a list of all file paths
    # containing every string in contains and without any string in excludes
    if contains is None:
        contains = list()
    if excludes is None:
        excludes = list()

    results = list()
    for path in os.listdir(root):
        full_path = os.path.join(root, path)
        if os.path.isdir(full_path):
            filepaths = get_filepaths(root=full_path, contains=contains, excludes=excludes)
            results.extend(filepaths)

        elif check_path(path=full_path, contains=contains, excludes=excludes):
            results.append(full_path)

    return results


def check_path(path: str, contains: list[str] | None = None, excludes: list[str] | None = None) -> bool:
    if excludes is None:
        excludes = []
    if contains is None:
        contains = []

    for substr in contains:
        if substr not in path:
            return False

    for substr in excludes:
        if substr in path:
            return False

    return True


def write_json(filepath: str, data: dict | list[dict], append: bool = False):
    # Writes json or json list data to the filepath. If append is true, will attempt to write it to
    # an existing file, and will create a new one otherwise.
    pathlib.Path(filepath).parents[0].mkdir(parents=True, exist_ok=True)

    if append:
        write_mode = "a"
    else:
        write_mode = "w"
    with open(filepath, write_mode) as save_file:
        processed = json.dumps(data, indent=4)
        save_file.write(processed)


def write_lines(filepath: str, data: list[str], append: bool = False):
    # Writes json list data to the filepath. If append is true, will attempt to write it to
    # an existing file, and will create a new one otherwise.
    pathlib.Path(filepath).parents[0].mkdir(parents=True, exist_ok=True)

    if append:
        write_mode = "a"
    else:
        write_mode = "w"
    with open(filepath, write_mode) as save_file:
        save_file.writelines(data)


def read_json(filepath: str) -> dict | list[dict]:
    # Reads json data from the filepath
    # If the first char in the file is a "[", reads in as jsonl.
    # Otherwise, reads in as json
    with open(filepath, encoding="utf8") as file:
        results = json.load(file)

    return results


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
