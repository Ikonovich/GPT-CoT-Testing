"""
This module contains settings for each test run.
Each run of main.py iterates through every combination of
selected_models, selected_modalities, and selected_datasets.
"""

# Sets the max number of tokens for each query
from Datasets.Enumerations import stepwise


# Stores the number of seconds to wait after a query failure
ERROR_WAIT_TIME = 60

# Store the base folder for dataset files
DATASET_FOLDER = "Datasets"

# Store the base folder for test results
RESULTS_FOLDER = "Results"

# Store all available test modalities
modalities = ["zero_shot", "zero_shot_cot", "suppressed_cot", "explanation_first",
              "answer_first", "the_answer_is"]

# Datasets. Stored as a list of json-formatted dictionaries with "Question" and "Ground Truth" keys.
datasets = {"multiarith": "MultiArith/MultiArith-Processed.jsonl", "gsm8k": "GSM8K/GSM8K-Processed.jsonl",
            "aqua": "AQUA/Aqua-Processed.jsonl", "coin_flip": "Coin Flip/coin_flip-processed.jsonl",
            "1step": stepwise[0], "2step": stepwise[1], "3step": stepwise[2],
            "4step": stepwise[3], "5step": stepwise[4], "6step": stepwise[5], "7step": stepwise[6],
            "8step": stepwise[7], "9step": stepwise[8], "10step": stepwise[9], "11step": stepwise[10],
            "12step": stepwise[11], "13step": stepwise[12], "14step": stepwise[13], "15step": stepwise[14]}

# Models. Models in chat will use the OpenAI ChatCompletion endpoint.
# Models in completion will use the Completion endpoint.
chat = ["gpt-4", "gpt-4-32k", "gpt-3.5-turbo"]
completion = ["text-davinci-002"]


# ---- PROMPTS ----

# Appended to tell the model to place its final answer into squiggly brackets.
in_bracket_prompt = "Place your final answer in squiggly brackets.\n"
# Used to extract the answer during the two stage extraction query.
two_stage_extract_prompt = "The answer is "
# Prompt to extract chain-of-thought reasoning
cot_prompt = "Let's think step by step."
# Used for the suppressed_cot modality
suppression_prompt = "Provide only the answer."
# Used for the answer_first modality
answer_first_prompt = "Provide the answer followed by the explanation."
# Used for the explanation_first modality
explanation_first_prompt = "Provide the explanation followed by the answer."
