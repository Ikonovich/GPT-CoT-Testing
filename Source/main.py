import argparse
import ast
import time

from openai.error import OpenAIError

from create_graphs import create_graphs
from data_utils import generate_metadata, process_mmlu
from answer_extraction import extract_answers
from config import DATASETS, ERROR_WAIT_TIME, chat, completion, modalities
from query_utils import run_test


def main():
    args = parse_args()

    mode = args.mode
    if mode == "extract":
        extract_answers()
    elif mode == "metadata":
        generate_metadata()
    elif mode == "graph":
        create_graphs()
    elif mode == "test":
        test(args)


def test(args):
    selected_models = args.models
    selected_modalities = args.modalities
    selected_datasets = args.datasets
    # Tracks how many models we've run through. Once we've run through them all, the
    # operation won't continue after a failed response.
    model_index = 0
    # # Outer loop ensures operations will repeat on a failed response.
    while model_index < len(selected_models):
        try:
            model = selected_models[model_index]
            # Runs a loop over all test modalities stored in config.selected_modalities
            for modality in selected_modalities:
                # Run a loop over all datasets listed in config.selected_datasets with the provided config parameters
                for entry in selected_datasets:
                    run_test(model=model, modality=modality, dataset=entry, args=args)
            # Increment and carry on to the next model
            model_index += 1
        except OpenAIError as e:
            print(f"An OpenAI API error has occurred: \n{e}\nAttempting to retry after {ERROR_WAIT_TIME} seconds.")
            # Wait and try again
            time.sleep(ERROR_WAIT_TIME)


def parse_args():
    parser = argparse.ArgumentParser(description='Platform for running series of tests on series of OpenAI models.')

    # Takes a run mode. Default is to perform tests.
    # Extract: runs data_utils.extract_answers to extract final answers from all test queries
    # Metadata: Generates metadate files for all tests in config.RESULTS_FOLDER, saving them in config.METADATA_FOLDER.
    # This includes calculating accuracy and quantifying CoT reasoning.
    # Graph: Runs create_graphs.create_graphs to create all graphs defined there.
    parser.add_argument(
        "--mode", type=str, choices=["test", "metadata", "graph", "extract"], default="test",
        help="Choose whether to run tests, extract answers, collate metadata, or graph results."
    )

    # Takes a list of open AI models to iteratively run through the provided modalities and datasets.
    parser.add_argument(
        "--models", type=str, nargs='+', choices=chat + completion, help="Models to test."
    )

    parser.add_argument(
        "--modalities", type=str, nargs='+', choices=modalities, help="Test modalities to run."
    )

    # Takes a list of datasets to be run through each model with each modality.
    parser.add_argument(
        "--datasets", type=str, nargs='+', choices=DATASETS, help="Datasets to be tested on."
    )

    # Takes a list of datasets to be run through each model with each modality.
    parser.add_argument(
        "--num_samples", type=int, help="Defines the max number of samples to be ran for each dataset."
    )

    parser.add_argument(
        "--use_simple_prompt", type=ast.literal_eval, default=True,
        help="If true, uses a simplified prompt format (Does not append Q: to the "
             "beginning of each question or A: after each question.)"
    )

    parser.add_argument(
        "--extraction_type", type=str, choices=["two-stage", "in-brackets", "none"],
        help="Select the answer extraction method to use. Two-stage sends an extra query appended with "
             + "\"The answer is\". In-brackets appends \"Place the final answer in squiggly brackets.\" to the "
             + "question. None does neither of these and only attempts to extract the answer directly - this is not "
             + "recommended for most cases."
    )

    parser.add_argument(
        "--save", type=ast.literal_eval, default=True,
        help="If true, saves the test results to a file and the associated metadata"
    )

    parser.add_argument(
        "--continuation", type=ast.literal_eval, default=True,
        help="If true, will look for a file matching the parameters "
             + "provided. If this file exists, will append all new results "
             + "to the file located. Only applies if save==True."
    )

    parser.add_argument(
        "--wait_time", type=int, default=0, help="Sets the minimum weight time between individual queries."
    )

    parser.add_argument(
        "--max_tokens", type=int, default=1000, help="Sets the maximum number of tokens that a model can use."
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # process_mmlu()
    main()
