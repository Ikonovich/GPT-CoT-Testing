import os

import pandas as pd

from config import GRAPHS_FOLDER
from data_utils import search_metadata
from graph_utils import graph_generic, graph_cot_data, graph_stepwise_comparison, graph_dataset_comparison


model_to_label_map = {"gpt-3.5-turbo": "GPT-3.5 Turbo", "gpt-4": "GPT-4", "text-davinci-002": "GPT-3",
                      "gpt-4-32k": "GPT-4-32k"}
model_to_index_map = {"text-davinci-002": 0, "gpt-3.5-turbo": 1, "gpt-4": 2, "gpt-4-32k": 3}


def create_graphs():
    non_stepwise_full(model="gpt-4")
    non_stepwise_full(model="gpt-3.5-turbo")
    non_stepwise_full(model="text-davinci-002")

    stepwise(model="gpt-4")
    stepwise(model="gpt-3.5-turbo")
    stepwise(model="text-davinci-002")

    stepwise_long(model="gpt-4")


def non_stepwise_full(model: str):
    modalities = ["zero_shot", "zero_shot_cot", "suppressed_cot", "explanation_first", "answer_first"]
    label = model_to_label_map[model]
    # Graph GPT-4 All dataset results
    results = search_metadata(model=model, modalities=modalities)

    data = pd.melt(results, id_vars=["Model", "Modality", "Dataset Index", "Modality Index", "Dataset"],
                   value_vars=["Total Accuracy"], value_name="Accuracy")

    graph_dataset_comparison(title=f"{label} Performance, All Modalities, All Datasets",
                             data=data,
                             modalities=modalities,
                             figsize=(10, 3),
                             plot_size=(1, 5),
                             sort_by="Modality Index",
                             add_legend=True,
                             output_path=os.path.join(GRAPHS_FOLDER, model, f"{model}-Full-Results"),
                             )

    # CoT Quant results
    data = pd.melt(results, id_vars=["Model", "Modality", "Dataset Index", "Modality Index", "Dataset"],
                   value_vars=["Total Accuracy", "Answers Containing CoT", "CoT Accuracy", "Non-CoT Accuracy"],
                   var_name="Metric",
                   value_name="Percentage").sort_values("Modality Index")

    graph_cot_data(title="GPT-4 CoT Quantification, All Modalities, All Datasets",
                   data=data,
                   figsize=(10, 10),
                   plot_size=(2, 3),
                   hue="Metric",
                   x="Modality",
                   groupby="Dataset Index",
                   output_path=os.path.join(GRAPHS_FOLDER, model, f"{model}-CoT-Quant"))


def stepwise(model: str):

    label = model_to_label_map[model]
    modalities = ["zero_shot", "zero_shot_cot", "suppressed_cot", "explanation_first", "answer_first"]

    results = search_metadata(model="gpt-4", modalities=modalities, datasets=["stepwise"])
    results = results.loc[results['Step'] <= 9]

    data = pd.melt(results,
                   id_vars=["Model", "Modality", "Modality Index", "Dataset", "Step", "ci_upper", "ci_lower"],
                   value_vars=["Total Accuracy"], value_name="Accuracy").sort_values(by=["Step", "Modality Index"])

    graph_stepwise_comparison(
        title=f"{label} Performance on Stepwise Dataset",
        data=data, modalities=modalities,
        output_path=os.path.join(GRAPHS_FOLDER, model, f"{model}-9-step-Results"),
    )

    # CoT Quantification
    data = pd.melt(results,
                   id_vars=["Step", "Modality", "Modality Index", "Model", "Model Index", "ci_upper", "ci_lower"],
                   value_vars=["Total Accuracy", "Answers Containing CoT", "CoT Accuracy", "Non-CoT Accuracy"],
                   var_name="Metric",
                   value_name="Percentage")

    # graph_cot_data(title="GPT-4 CoT Quantification, Stepwise Dataset",
    #                data=data,
    #                x="Step",
    #                xtick_labels=[str(i) + " Step" for i in range(1, 21)],
    #                figsize=(15, 5),
    #                plot_size=(1, 3),
    #                output_path="GPT4/Stepwise/gpt4-step-supp-cot-quant")


def stepwise_long(model: str):

    label = model_to_label_map[model]
    modalities = ["zero_shot", "zero_shot_cot", "suppressed_cot", "explanation_first", "answer_first"]

    results = search_metadata(model="gpt-4", datasets=["stepwise"], modalities=modalities)
    results = results.loc[results['Step'] <= 23]

    data = pd.melt(results,
                   id_vars=["Model", "Modality", "Modality Index", "Dataset", "Step", "ci_upper", "ci_lower"],
                   value_vars=["Total Accuracy"], value_name="Accuracy").sort_values(by=["Step", "Modality Index"])

    graph_stepwise_comparison(
        title=f"{label} Performance on Stepwise Dataset",
        data=data, modalities=modalities,
        output_path=os.path.join(GRAPHS_FOLDER, model, f"{model}-23-Step-Results"),
    )

    # CoT Quantification
    data = pd.melt(results,
                   id_vars=["Step", "Modality", "Modality Index", "Model", "Model Index", "ci_upper", "ci_lower"],
                   value_vars=["Total Accuracy", "Answers Containing CoT", "CoT Accuracy", "Non-CoT Accuracy"],
                   var_name="Metric",
                   value_name="Percentage")

    # graph_cot_data(title="GPT-4 CoT Quantification, Stepwise Dataset",
    #                data=data,
    #                x="Step",
    #                xtick_labels=[str(i) + " Step" for i in range(1, 21)],
    #                figsize=(15, 5),
    #                plot_size=(1, 3),
    #                output_path="GPT4/Stepwise/gpt4-step-supp-cot-quant")


def prompt_comparison():
    frames = list()

    frames.append(pd.DataFrame({
        "Model": "text-davinci-002",
        "Model Index": model_to_index_map["text-davinci-002"],
        "Extraction": "Initial",
        "Extraction Index": 1,
        "Dataset": "gsm8k",
        "Total Accuracy": [15.0]
    }))

    frames.append(pd.DataFrame({
        "Model": "text-davinci-002",
        "Model Index": model_to_index_map["text-davinci-002"],
        "Extraction": "Simplified",
        "Extraction Index": 2,
        "Dataset": "gsm8k",
        "Total Accuracy": [12.17]
    }))

    frames.append(pd.DataFrame({
        "Model": "text-davinci-002",
        "Model Index": model_to_index_map["text-davinci-002"],
        "Extraction": "Simplified In-Brackets",
        "Extraction Index": 3,
        "Dataset": "gsm8k",
        "Total Accuracy": [11.0]
    }))

    frames.append(pd.DataFrame({
        "Model": "gpt-3.5-turbo",
        "Model Index": model_to_index_map["gpt-3.5-turbo"],
        "Extraction": "Initial",
        "Extraction Index": 1,
        "Dataset": "gsm8k",
        "Total Accuracy": [76.3]
    }))

    frames.append(pd.DataFrame({
        "Model": "gpt-3.5-turbo",
        "Model Index": model_to_index_map["gpt-3.5-turbo"],
        "Extraction": "Simplified",
        "Extraction Index": 2,
        "Dataset": "gsm8k",
        "Total Accuracy": [81.16]
    }))

    frames.append(pd.DataFrame({
        "Model": "gpt-3.5-turbo",
        "Model Index": model_to_index_map["gpt-3.5-turbo"],
        "Extraction": "Simplified In-Brackets",
        "Extraction Index": 3,
        "Dataset": "gsm8k",
        "Total Accuracy": [78.5]
    }))

    orig = pd.concat(frames)
    data = pd.melt(orig, id_vars=["Model", "Model Index", "Extraction", "Extraction Index", "Dataset"],
                   value_vars=["Total Accuracy"], value_name="Accuracy").sort_values("Dataset")
    graph_generic(title="Comparison of Answer Extraction Techniques By Model, GSM8k",
                  data=data, groupby="Model Index", output_path="gsm8k-extraction-comparison",
                  chart_labels=["text-davinci-002", "GPT 3.5", "GPT-4"],
                  x_labels=["Initial", "Simplified", "In-Brackets"], figsize=(5, 3),
                  x="Extraction Index", plot_size=(1, 2), y="Accuracy")

    frames = list()

    frames.append(pd.DataFrame({
        "Model": "text-davinci-002",
        "Model Index": model_to_index_map["text-davinci-002"],
        "Modality": "zero_shot",
        "Extraction": "Initial",
        "Extraction Index": 1,
        "Dataset": "multiarith",
        "Total Accuracy": [26.0]
    }))

    frames.append(pd.DataFrame({
        "Model": "text-davinci-002",
        "Model Index": model_to_index_map["text-davinci-002"],
        "Modality": "zero_shot",
        "Extraction": "Simplified",
        "Extraction Index": 2,
        "Dataset": "multiarith",
        "Total Accuracy": [21.2]
    }))

    frames.append(pd.DataFrame({
        "Model": "text-davinci-002",
        "Model Index": model_to_index_map["text-davinci-002"],
        "Modality": "zero_shot",
        "Extraction": "Simplified In-Brackets",
        "Extraction Index": 3,
        "Dataset": "multiarith",
        "Total Accuracy": [15.5]
    }))

    frames.append(pd.DataFrame({
        "Model": "gpt-3.5-turbo",
        "Model Index": model_to_index_map["gpt-3.5-turbo"],
        "Modality": "zero_shot",
        "Extraction": "Initial",
        "Extraction Index": 1,
        "Dataset": "multiarith",
        "Total Accuracy": [90.16]
    }))

    frames.append(pd.DataFrame({
        "Model": "gpt-3.5-turbo",
        "Model Index": model_to_index_map["gpt-3.5-turbo"],
        "Modality": "zero_shot",
        "Extraction": "Simplified",
        "Extraction Index": 2,
        "Dataset": "multiarith",
        "Total Accuracy": [95.3]
    }))

    frames.append(pd.DataFrame({
        "Model": "gpt-3.5-turbo",
        "Model Index": model_to_index_map["gpt-3.5-turbo"],
        "Extraction Index": 3,
        "Modality": "zero_shot",
        "Extraction": "Simplified In-Brackets",
        "Dataset": "multiarith",
        "Total Accuracy": [91.5]
    }))

    orig = pd.concat(frames)
    data = pd.melt(orig, id_vars=["Model", "Model Index", "Extraction", "Extraction Index", "Dataset"],
                   value_vars=["Total Accuracy"], value_name="Accuracy").sort_values("Dataset")
    graph_generic(title="Comparison of Answer Extraction Techniques By Model, MultiArith",
                  data=data, groupby="Model Index", output_path="multiarith-extraction-comparison",
                  chart_labels=["text-davinci-002", "GPT 3.5", "GPT-4"],
                  x_labels=["Initial", "Simplified", "In-Brackets"], figsize=(5, 3),
                  x="Extraction Index", plot_size=(1, 2), y="Accuracy")
