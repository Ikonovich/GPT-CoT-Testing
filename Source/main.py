import time

import openai

from config import selected_datasets, selected_modalities, selected_models, ERROR_WAIT_TIME
from query_utils import run_test


if __name__ == '__main__':

    # If true, saves query output to a file with the format Model-TestType-.jsonl
    # in the folder Model/Test
    save = True

    # Tracks how many models we've ran through. Once we've ran through them all, the
    # operation won't continue after a failed response.
    model_index = 0
    # Outer loop ensures operations will repeat on a failed response.
    while model_index < len(selected_models):
        try:
            model = selected_models[model_index]
            # Runs a loop over all test modalities stored in config.selected_modalities
            for modality in selected_modalities:
                # Run a loop over all datasets listed in config.selected_datasets with the provided config parameters
                for entry in selected_datasets:
                    run_test(model=model, modality=modality, dataset=entry, save=True, cont=True)
            # Increment and carry on to the next model
            model_index += 1
        except Exception as e:
            print(e)
            # Wait and try again
            time.sleep(ERROR_WAIT_TIME)











