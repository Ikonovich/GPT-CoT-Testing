# Sets the max number of tokens for each query
from Datasets.Enumerations import stepwise

# Sets max tokens for the model responses
MAX_TOKENS = 1000
# Sets the maximum of samples to use from each dataset
MAX_SAMPLES = 600

# Store the base folder for dataset files
DATASET_FOLDER = "Datasets"

# Store the base folder for test results
RESULTS_FOLDER = "Results"

# Datasets. Stored as a list of json-formatted dictionaries with "Question" and "Ground Truth" keys.
datasets = {"multiarith": "MultiArith/MultiArith-Processed.jsonl", "gsm8k": "GSM8K/GSM8K-Processed.jsonl", "1step": stepwise[0], "2step": stepwise[1], "3step": stepwise[2],
            "4step": stepwise[3], "5step": stepwise[4], "6step": stepwise[5], "7step": stepwise[6],
            "8step": stepwise[7], "9step": stepwise[8]}

# Models. Models in chat will use the OpenAI ChatCompletion endpoint.
# Models in completion will use the Completion endpoint.
chat = ["gpt-4", "gpt-4-32k", "gpt-3.5-turbo"]
completion = ["text-davinci-002"]


# Set the model to be used.
# Options are in config.chat and config.completion
selected_models = ["gpt-3.5-turbo"]


# Set the test modalities to be tested on.
# All use two-stage prompting, with the second prompt appending the answer extraction prompt
# stored in config.extract_prompt.
# Zero shot uses the question only, beginning with "Q:". Suppressed cot appends the question with the prompt stored in
# config.suppress_cot_prompt, by default "Provide only the answer."
# Options: "zero_shot", "suppressed_cot", "zero_shot_cot"

selected_modalities = ["zero_shot", "zero_shot_cot", "suppressed_cot"]

# Used to extract the answer after each question.
extract_prompt = "The answer (arabic numerals) is "

# Prompt to extract chain-of-thought reasoning
cot_prompt = "Let's think step by step."

# The prompts available for the suppressed_cot modality
suppression_prompts = ["Provide only the answer.", "Provide the answer followed by the explanation.",
                       "Provide the explanation followed by the answer."]

suppression_prompt = suppression_prompts[0]


# Set the datasets to be tested against.
# Options are in config.datasets
stepwise_datasets = ["gsm8k", "multiarith", "1step", "2step", "3step", "4step"]  # "5step", "6step", "7step", "8step", "9step"]
selected_datasets = stepwise_datasets
