"""
This module contains various configurations that aren't individual to each test.
"""
import os

# Stores the ID of the GPU to utilize
GPU_ID = 4

# Stores the minimum wait time between queries, in seconds
WAIT_TIME = 0

# Sets the number of steps available in the stepwise dataset
STEPWISE_MAX_SIZE = 55

# Stores the number of seconds to wait after a query failure
ERROR_WAIT_TIME = 60

# Store the base folder for dataset files
DATASET_FOLDER = "Datasets"

# Store the base folder for test results
RESULTS_FOLDER = os.path.join("Results", "Primary_Test_Results")
METADATA_FOLDER = os.path.join("Results", "Metadata")
# Store the base folder for generated graphs

GRAPHS_FOLDER = os.path.join("Results", "Graphs")

# Store all available test modalities
MODALITIES = ["zero_shot", "zero_shot_cot", "suppressed_cot", "explanation_first",
              "answer_first", "the_answer_is"]

# Datasets. Stored as a list of json-formatted dictionaries with "Question" and "Ground Truth" keys.
DATASETS = {"multiarith": os.path.join("MultiArith", "MultiArith-Processed.json"),
            "gsm8k": os.path.join("GSM8K", "GSM8K-Processed.json"),
            "aqua": os.path.join("AQUA", "Aqua-Processed.json"),
            "coin_flip": os.path.join("Coin Flip", "coin_flip-processed.json"),
            "mmlu-high-school": os.path.join("MMLU", "Processed", "mmlu_high_school.json"),
            "mmlu-college": os.path.join("MMLU", "Processed", "mmlu_college.json")}

# Add all possible stepwise datasets to the mapping
# Some of these may not actually exist in the data
for i in range(0, STEPWISE_MAX_SIZE):
    DATASETS[f"{i + 1}step"] = os.path.join("Stepwise", f"{i + 1}-Step-Int-Formulae.json")

# Create a mapping of modified CoT datasets
for i in range(0, STEPWISE_MAX_SIZE):
    DATASETS[f"Modified-Single-Val-Final-{i + 1}-step"] = os.path.join("Stepwise_Extracted",
                                                                                    "Modified-Single-Val-Final",
                                                                                    f"{i + 1}-step.json")
    DATASETS[f"Modified-Double-Val-Final-{i + 1}-step"] = os.path.join("Stepwise_Extracted",
                                                                                    "Modified-Double-Val-Final",
                                                                                    f"{i + 1}-step.json")

# Models. Models in chat will use the OpenAI ChatCompletion endpoint.
# Models in completion will use the Completion endpoint
# Local models will use a huggingface API.
CHAT = ["gpt-4", "gpt-4-32k", "gpt-3.5-turbo"]
COMPLETION = ["text-davinci-002"]
# Stores models that have to use transformers.AutoTokenizer and AutoModelForCausalLM
LOCAL_AUTO = {"alpaca": "chavinlo/alpaca-native"}
# Stores models that have to use transformers.LlamaTokenizer and LlamaForCausalLM
LOCAL_LLAMA = {"goat": "tiedong/Goat"}

# ---- PROMPTS ----

# Appended to tell the model to place its final answer into squiggly brackets.
in_bracket_prompt = "Place your final answer in squiggly brackets.\n"
# Used to extract the answer during the two stage extraction query for style one.
# For style two, see query_utils.two_stage_style_two_generation
two_stage_extract_prompt = "The answer is "
# Prompt to extract chain-of-thought reasoning
cot_prompt = "Let's think step by step."
# Used for the suppressed_cot modality
suppression_prompt = "Provide only the answer."
# Used for the answer_first modality
answer_first_prompt = "Provide the answer followed by the explanation."
# Used for the explanation_first modality
explanation_first_prompt = "Provide the explanation followed by the answer."

# ---- DATA UTIL ITEMS ----

# Confidence interval 95% z-value
z_val = 1.96

# Mapping used for sorting dataframes.

# Maps testing modalities to indices
modality_index_map = {"zero_shot": 0, "zero_shot_cot": 1, "suppressed_cot": 2,
                      "explanation_first": 3, "answer_first": 4, "the_answer_is": 5}
# Maps non-stepwise datasets to indices
dataset_index_map = {"multiarith": 0, "gsm8k": 1, "aqua": 2, "coin_flip": 3, "mmlu-combined": 4, "stepwise": 5,
                     "mmlu-high-school": 6, "mmlu-college": 7}
# Maps models to indices
model_index_map = {"text-davinci-002": 0, "gpt-3.5-turbo": 1, "gpt-4": 2, "gpt-4-32k": 3, "goat": 4}
