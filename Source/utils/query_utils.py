import time
from json import JSONDecodeError
from os import path

import openai
import torch
import transformers
from regex import regex

from transformers import (
    LlamaTokenizer,
    LlamaForCausalLM,
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig
)
from peft import PeftModel

from config import *
from utils.file_utils import load_dataset, write_json, read_json

# Stores the time of the last query
last_query_time = 0

# Stores tuples of (local model, tokenizer),
# by the same key values as in config.LOCAL.
local_models = dict()


# Timer function that pauses operation until the time since last query exceeds config.WAIT_TIME
def timer(delay: float | int = None):
    global last_query_time

    if delay is None:
        delay = WAIT_TIME
    cur_time = time.time()
    diff = cur_time - last_query_time
    if diff < delay:
        time.sleep(delay - diff)

    last_query_time = time.time()


def query(model: str, prompt: str, max_tokens: int) -> str:
    if model in COMPLETION or model in CHAT:
        return openai_query(
            model_name=model,
            prompt=prompt,
            max_tokens=max_tokens)
    else:
        return local_query(
            model_name=model,
            prompt=prompt,
            max_tokens=max_tokens)


def local_query(model_name: str, prompt: str, max_tokens: int) -> str:
    if model_name not in local_models:
        load_local_model(model_name=model_name)

    generation_config = GenerationConfig(
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=512,
        stream_output=True,
        return_dict_in_generate=True,
        output_scores=True,
    )
    with torch.inference_mode():
        model, tokenizer = local_models[model_name]

        input_ids = tokenizer(
            prompt,
            return_tensors="pt").input_ids
        # Replace the starting zero with BOS token and convert to torch-cuda
        input_ids[0][0] = 1
        input_ids = input_ids.to("cuda")
        # Generate the response and decode
        generated = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            stopping_criteria=transformers.StoppingCriteriaList())
        response = generated.sequences[0]
        response = tokenizer.decode(response, skip_special_tokens=True).strip()
        response = response.split(prompt)[1].strip()
    return response


def openai_query(model_name: str, prompt: str, max_tokens: int) -> str:
    # Run the timer to keep from querying too quickly
    timer()

    if model_name in COMPLETION:
        response = openai.Completion.create(
            engine=model_name,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0,
            stop=None
        )

        return response["choices"][0]["text"]

    elif model_name in CHAT:
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0,
            stop=None
        )

        return response["choices"][0]["message"]["content"]

    else:
        raise ValueError("The provided model has not been defined.")


def load_local_model(model_name: str):
    torch.cuda.set_device(GPU_ID)

    if model_name == "goat":
        tokenizer = LlamaTokenizer.from_pretrained('decapoda-research/llama-7b-hf')
        model = LlamaForCausalLM.from_pretrained(
            'decapoda-research/llama-7b-hf',
            torch_dtype=torch.float16,
            device_map="auto")
        model = PeftModel.from_pretrained(
            model,
            "tiedong/goat-lora-7b",
            torch_dtype=torch.float16,
            device_map={'': 0}
        )
        local_models[model_name] = (model, tokenizer)

    elif model_name in LOCAL_AUTO:
        model_path = LOCAL_AUTO[model_name]
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto", )
        local_models[model_name] = (model, tokenizer)
    elif model_name in LOCAL_LLAMA:
        model_path = LOCAL_LLAMA[model_name]
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto")
        local_models[model_name] = (model, tokenizer)
    else:
        raise ValueError("An invalid model has been provided to local_query.")


def build_prompt(question: str, modality: str, use_simple_prompt: bool, bracket_extract: bool) -> str:
    # Constructs the question query based on the test type and the prompt settings in config.
    # If simple is false, appends Q to the start of each question and A at certain points in the question

    # Generate universal insertions
    Q = ""
    A = ""
    B = ""
    if bracket_extract:
        B = in_bracket_prompt
    if not use_simple_prompt:
        Q += "Q: "
        A += "A: "

    match modality:
        case "zero_shot_cot":
            output = f"{Q}{question} {B} {A}{cot_prompt}"
        case "zero_shot":
            output = f"{Q}{question} {B} {A}"
        case "the_answer_is":
            output = f"{Q}{question}. {B} {A}The answer is "
        case "zero_shot_no_extract":
            output = f"{Q}{question} {B} {A}The answer (arabic numerals) is "
        case "suppressed_cot":
            output = f"{Q}{question} {B} {suppression_prompt}. {A}"
        case "explanation_first":
            output = f"{Q}{question} {B} {explanation_first_prompt}. {A}"
        case "answer_first":
            output = f"{Q}{question} {B} {answer_first_prompt}. {A}"
        case _:
            raise ValueError("The provided test type has not been defined.")

    return output


def run_test(model: str, modality: str, dataset: str, args):
    # Runs a test on a given model, test modality, and dataset.
    # If num samples is 0, runs the whole dataset, otherwise stops at index num_samples
    # Keeps a running accuracy and saves the results as they come in if desired.
    # If cont is true, will attempt to load the first line from the file
    # with the name Model-Modality-Dataset-Results.jsonl and begin iterating from the value stored there at
    # last_index
    cont = args.continuation
    use_simple_prompt = args.use_simple_prompt
    extraction_type = args.extraction_type
    num_samples = args.num_samples
    max_tokens = args.max_tokens
    mode = args.mode

    if extraction_type == "in-brackets":
        bracket_extract = True
    else:
        bracket_extract = False

    if use_simple_prompt:
        prompt = "Simple"
    else:
        prompt = "Initial"

    # Set the results folder, because we don't split individual step runs into separate folders
    if dataset in MODIFIED_COT_DATASETS:
        stop_val = regex.findall(r"\d{1,2}", dataset)[0]
        stop_index = dataset.index(stop_val)
        dataset_sub = dataset[:stop_index - 1]
    elif 'step' in dataset:
        dataset_sub = "stepwise"
    else:
        dataset_sub = dataset

    # Results file: Stores last index ran, total count, correct counts, and accuracy.
    if dataset in MODIFIED_COT_DATASETS:
        output_file = dataset + "-" + mode + "-" + model + ".json"
        save_directory = path.join(RESULTS_FOLDER, mode, model, modality, dataset_sub)
    else:
        output_file = prompt + "-" + extraction_type + "-" + model + "-" + modality + "-" + dataset + ".json"
        save_directory = path.join(RESULTS_FOLDER, model, modality, dataset_sub)

    output_path = path.join(save_directory, output_file)
    dataset_path = path.join(DATASET_FOLDER, DATASETS[dataset])

    # Set the start index
    # This lets us continue from where we left off if the model is overloaded or the test has to restart.
    start_index = 0
    if cont and path.exists(output_path):
        try:
            previous = read_json(filepath=output_path)
            start_index = len(previous["Trials"])

        except JSONDecodeError as e:
            print(f"There was an error decoding the prior test results at {output_path} into json at index {e.pos}")
        except KeyError as e:
            print(f"Expected key {e.args[0]} was not found in the last index of {output_path} when "
                  f"decoded into json.")

    # Load the dataset
    data = load_dataset(dataset_path)
    # Set the end index
    if num_samples == 0:
        end_index = len(data)
    else:
        end_index = min(num_samples, len(data))

    for j in range(start_index, end_index):
        # Get the question, ground truth, and full dataset entry
        x, y, test_entry = data[j]
        # Initialize the results, which will be updated as we go
        results = {"Index": test_entry["Index"], "GT": y}
        # Build the prompt out of our question
        prompt = build_prompt(
            question=x,
            modality=modality,
            use_simple_prompt=use_simple_prompt,
            bracket_extract=bracket_extract)
        results["Query"] = prompt
        # if model == "goat":
        #     prompt = prompt.replace(" =", "?")
        #     prompt = "What is " + prompt + "Answer: "
        # Get the initial response from the model

        if dataset not in MODIFIED_COT_DATASETS:
            # Run a normal test prompt
            response = query(
                model=model,
                prompt=prompt,
                max_tokens=max_tokens)
        else:
            # Inject the modified CoT into the test context
            cot = "\n".join(test_entry["New Steps"])
            results["Injected CoT"] = cot
            # If the model is local, concatenate the question and steps.
            if model == 'goat':
                response = local_query(
                    model_name=model,
                    prompt=prompt + " \n" + cot,
                    max_tokens=max_tokens)
            # Otherwise, send the steps as GPT assistant message and the query as a user message.
            else:
                messages = [{"role": "user", "content": prompt},
                            {"role": "assistant", "content": cot}]
                response = multi_message_query(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens)

        results["Response"] = response

        if dataset == "aqua" or "mmlu" in dataset:
            options = test_entry["Options"]
            results["Options"] = options
        else:
            options = None

        # Validate the extraction type and perform two-stage extraction, if necessary.
        if extraction_type == "in-brackets":
            extraction_response = "None"
        else:
            extraction_response = extraction_query(
                prompt=prompt,
                response=response,
                options=options,
                args=args)
        results["Extract-Response"] = extraction_response

        if path.exists(output_path):
            test_results = read_json(filepath=output_path)
        else:
            # Generate new test metadata
            test_results = {"Mode": args.mode,
                            "Model": model,
                            "Model Index": model_index_map[model],
                            "Modality": modality,
                            "Modality Index": modality_index_map[modality],
                            "Dataset": dataset_sub,
                            }
            if "step" in dataset:
                test_results["Steps"] = int(regex.findall(r"\d{1,2}", dataset)[0])

        test_results.update({"Extraction Type": extraction_type,
                             "Simple Prompt": use_simple_prompt,
                             "Test Path": output_path,
                             "Trials": list()
                             })

        test_results["Trials"].append(results)
        write_json(filepath=output_path, data=test_results)
        trial_index = test_entry["Index"]
        # print(f"Model: {model} Dataset: {dataset} Index: {trial_index} Iteration: {j}"
        #       f"\nPrompt: {prompt} "
        #       f"\nResponse: {response} "
        #       f"\nExtraction Response: {extraction_response},"
        #       f"\nGT: {y}")
    print("Test " + model + "-" + modality + "-" + dataset + " completed.")


def extraction_query(prompt: str, response: str, options: dict[str, str] | None, args) -> str:
    extraction_type = args.extraction_type
    model = args.model
    max_tokens = args.max_tokens

    if extraction_type == "two-stage":
        # Resubmit the response for answer extraction
        extraction_prompt = prompt + " " + response + "\n" + two_stage_extract_prompt
        extraction_response = query(
            model=model,
            prompt=extraction_prompt,
            max_tokens=max_tokens)

    elif extraction_type == "two-stage-style-two":
        extraction_prompt = two_stage_style_two_generation(
            answer=response,
            options=options)
        extraction_response = query(
            model=model,
            prompt=extraction_prompt,
            max_tokens=max_tokens)
    else:
        raise ValueError("The provided extraction type is not valid.")

    return extraction_response


# Used to format the second two-stage-second-style prompt.
def two_stage_style_two_generation(answer: str, options: dict) -> str:
    prompt = "Only provide one letter answer. The following statement contains the full answer for " \
             "an unknown multiple-choice math problem. Somewhere in the string provided, " \
             "is the final answer choice, ranging from [A, B, C, D]. We want to know the answer " \
             "choice only. Given this string, write the final answer choice presented there. " \
             "Do not provide any other explanations or descriptions. If no answer letter provided, " \
             "answer with \"N/A\". Full answer:\n"
    prompt += answer
    prompt += "\nPossible choices:\n"
    for key in options:
        prompt += f"{key}: {options[key]}"
    prompt += "\nDo not explain your answer. Only provide the answer choice in squiggly brackets in the following " \
              "format {answer}: "

    return prompt


def multi_message_query(model: str, messages: list[dict[str, str]], max_tokens: int):
    # Run the timer to keep from querying too quickly
    timer()

    if model not in CHAT:
        raise ValueError("The provided model is not a Chat-equipped model.")

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0,
        stop=None
    )

    return response["choices"][0]["message"]["content"]
