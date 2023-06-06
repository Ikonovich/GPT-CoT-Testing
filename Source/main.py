import argparse
import ast
import time

from openai.error import OpenAIError

import config
from create_graphs import create_graphs
from utils.data_utils import generate_metadata
from answer_extraction import extract_answers
from config import *
from utils.query_utils import run_test
from scratchpad import multi_query_test
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
def main():
    args = parse_args()
    config.__dict__['GPU_ID'] = args.gpu
    config.__dict__['WAIT_TIME'] = args.wait_time
    mode = args.mode
    if mode == "extract":
        extract_answers(root=RESULTS_FOLDER)
    elif mode == "metadata":
        generate_metadata(root=RESULTS_FOLDER, test_file="Test_Results.csv",
                          scratchpad_file="Scratchpad_Results.csv")
    elif mode == "graph":
        create_graphs()
    elif mode == "test" or mode == "scratchpad" or mode == "modified_cot":
        test(args=args)


def test(args):
    config.WAIT_TIME = args.wait_time
    selected_models = args.models
    selected_modalities = args.modalities
    selected_datasets = args.datasets
    # Tracks how many models we've run through. Once we've run through them all, the
    # operation will stop.
    model_index = 0
    # # Outer loop ensures operations will repeat on a failed response.
    while model_index < len(selected_models):
        try:
            # Runs a loop over all test modalities stored in config.selected_modalities
            for modality in selected_modalities:
                # Run a loop over all datasets listed in config.selected_datasets with the provided config parameters
                for entry in selected_datasets:
                    if args.mode == "test" or args.mode == "modified_cot":
                        model = selected_models[model_index]
                        run_test(model=model, modality=modality, dataset=entry, args=args)
                    elif args.mode == "scratchpad":
                        model_one = selected_models[model_index]
                        model_two = selected_models[model_index + 1]
                        multi_query_test(model_one=model_one, model_two=model_two, modality=modality, dataset=entry,
                                         args=args)

            # Increment and carry on to the next model or models:
            if args.mode == "scratchpad":
                model_index += 2
            else:
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
        "--mode", type=str,
        choices=["test", "metadata", "graph", "extract", "scratchpad"],
        default="test",
        help="Choose whether to run standard single-query tests, extract answers, collate metadata, graph results, "
             "or to run scratchpad test mode."
    )

    # Takes a list of open AI models to iteratively run through the provided modalities and datasets.
    parser.add_argument(
        "--models", type=str, nargs='+', choices=CHAT + COMPLETION + [key for key in LOCAL_AUTO]
        + [key for key in LOCAL_LLAMA], help="Models to test."
    )

    parser.add_argument(
        "--modalities", type=str, nargs='+', choices=MODALITIES, help="Test modalities to run."
    )

    # Takes a list of datasets to be run through each model with each modality.
    parser.add_argument(
        "--datasets", type=str, nargs='+', choices=DATASETS,
        help="Datasets to be tested on."
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
        "--extraction_type", type=str, choices=["two-stage", "in-brackets", "none", "two-stage-style-two",
                                                "two-stage-multi-choice"],
        help="Select the answer extraction method to use. Two-stage sends an extra query appended with "
             "\"The answer is\". In-brackets appends \"Place the final answer in squiggly brackets.\" to the "
             "question. Style two uses the query_utils.two_stage_style_two_generation function. Multi choice uses the prompt "
             "\"Therefore, among A through (last option letter), the answer is\". None does none of these and only "
             "attempts to extract the answer directly - this is not recommended for most cases."
    )

    parser.add_argument(
        "--continuation", type=ast.literal_eval, default=True,
        help="If true, will look for a file matching the parameters "
             "provided. If this file exists, will append all new results "
             "to the file located. Only applies if save==True."
    )

    parser.add_argument(
        "--wait_time", type=int, default=0, help="Sets the minimum weight time between individual queries."
    )

    parser.add_argument(
        "--max_tokens", type=int, default=1000, help="Sets the maximum number of tokens that a model can use."
    )

    parser.add_argument(
        "--gpu", type=int, default=0, help="Sets the GPU that locally ran models will utilize."
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
