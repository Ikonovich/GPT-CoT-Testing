"""
This module contains settings for each test run.
Each run of main.py iterates through every combination of
selected_models, selected_modalities, and selected_datasets.
"""

# Sets the max number of tokens for each query
from Datasets.Enumerations import stepwise

# If true, uses a simplified prompt format
# (Does not append Q: to the beginning of each question or A: after each question.
USE_SIMPLE_PROMPT = True

# Sets max tokens for the model responses
MAX_TOKENS = 1000
# Sets the maximum of samples to use from each dataset
MAX_SAMPLES = 600

# Stores the number of seconds to wait between queries by default
WAIT_TIME = 0
# Stores the number of seconds to wait after a query failure
ERROR_WAIT_TIME = 60

# Store the base folder for dataset files
DATASET_FOLDER = "Datasets"

# Store the base folder for test results
RESULTS_FOLDER = "Results"

# Store all available test modalities
modalities = ["zero_shot_no_extract", "zero_shot", "zero_shot_cot", "suppressed_cot", "explanation_first",
              "answer_first", "the_answer_is"]

# Datasets. Stored as a list of json-formatted dictionaries with "Question" and "Ground Truth" keys.
datasets = {"multiarith": "MultiArith/MultiArith-Processed.jsonl", "gsm8k": "GSM8K/GSM8K-Processed.jsonl",
            "1step": stepwise[0], "2step": stepwise[1], "3step": stepwise[2],
            "4step": stepwise[3], "5step": stepwise[4], "6step": stepwise[5], "7step": stepwise[6],
            "8step": stepwise[7], "9step": stepwise[8]}

# Models. Models in chat will use the OpenAI ChatCompletion endpoint.
# Models in completion will use the Completion endpoint.
chat = ["gpt-4", "gpt-4-32k", "gpt-3.5-turbo"]
completion = ["da-vinci-002"]

# Set the model to be used.
# Options are in config.chat and config.completion
selected_models = ["gpt-3.5-turbo", "da-vinci-002"]

# Set the test modalities to be tested on.
# All except for zero_shot_no_extract use two-stage prompting, with the second prompt appending the answer extraction
# prompt stored in config.extract_prompt.
# Zero shot uses the question only, beginning with "Q:". Suppressed cot appends the question with
# the prompt stored in config.suppress_cot_prompt, by default "Provide only the answer."
# Options: "zero_shot", "the_answer_is", "suppressed_cot", "zero_shot_cot", "zero_shot_no_extract", "explanation_first",
# "answer_first"


# Store the modalities to be tested on
selected_modalities = ["explanation_first", "answer_first"]
# Set the datasets to be tested against.
# Options are in config.datasets
selected_datasets = ["gsm8k", "multiarith", "1step", "2step", "3step", "4step", "5step", "6step", "7step", "8step",
                     "9step"]

# ---- PROMPTS ----

# Used to extract the answer after each question.
extract_prompt = "The answer (arabic numerals) is "
# Prompt to extract chain-of-thought reasoning
cot_prompt = "Let's think step by step."
# Used for the suppressed_cot modality
suppression_prompt = "Provide only the answer."
# Used for the answer_first modality
answer_first_prompt = "Provide the answer followed by the explanation."
# Used for the explanation_first modality
explanation_first_prompt = "Provide the explanation followed by the answer."
