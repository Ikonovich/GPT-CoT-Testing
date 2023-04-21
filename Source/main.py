import os
import config
import data_utils
from config import datasets, selected_datasets, selected_modalities, selected_models
from query_utils import run_test, clean_numeric_answer
from data_utils import load_dataset
from visualizer import generate_table

if __name__ == '__main__':
    # If true, saves query output to a file with the format Model-TestType-.jsonl
    # in the folder Model/Test
    save = True

    # Runs a loop over all models stored in config.selected_models
    for model in selected_models:
        # Runs a loop over all test modalities stored in config.selected_modalities
        for modality in selected_modalities:
            # Run a loop over all datasets listed in config.selected_datasets with the provided config parameters
            for entry in selected_datasets:
                run_test(model=model, modality=modality, dataset=entry, save=True, cont=True)











