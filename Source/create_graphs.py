import pandas as pd

from config import model_index_map
from data_utils import search_metadata
from graph_utils import graph_generic, graph_stepwise_comparison, graph_dataset_comparison, \
    modality_to_label_map

dataset_to_label_map = {"multiarith": "MultiArith", "gsm8k": "GSM8k", "aqua": "Aqua-RAT", "coin_flip": "Coin Flip",
                        "mmlu": "MMLU"}
model_to_label_map = {"gpt-3.5-turbo": "GPT-3.5 Turbo", "gpt-4": "GPT-4", "text-davinci-002": "GPT-3",
                      "gpt-4-32k": "GPT-4-32k"}


def create_graphs():
    single_dataset_results(dataset="multiarith")
    single_dataset_results(dataset="aqua")
    single_dataset_results(dataset="mmlu",
                           models=["gpt-3.5-turbo", "gpt-4"])
    single_dataset_results(dataset="gsm8k")
    single_dataset_results(dataset="coin_flip",
                           models=["gpt-3.5-turbo", "gpt-4"],
                           modalities=["zero_shot", "zero_shot_cot", "suppressed_cot"],
                           save_discriminator="Supp-Two-Model")

    graph_non_stepwise(model="gpt-4")
    graph_non_stepwise(model="gpt-3.5-turbo")
    graph_non_stepwise(model="text-davinci-002", num_plots=4)

    graph_stepwise(model="gpt-4", max_steps=23)
    graph_stepwise(model="gpt-3.5-turbo", max_steps=23)
    graph_stepwise(model="text-davinci-002", max_steps=23)


def single_dataset_results(dataset: str, models: list[str] = None, modalities: list[str] = None,
                           save_discriminator: str = "Full"):
    if modalities is None:
        modalities = ["zero_shot", "zero_shot_cot", "suppressed_cot", "explanation_first", "answer_first"]
    if models is None:
        models = ["text-davinci-002", "gpt-3.5-turbo", "gpt-4"]

    length = (len(models))

    data = search_metadata(models=models,
                           modalities=modalities,
                           datasets=[dataset])

    compare = pd.melt(data,
                      id_vars=["Model", "Modality", "Model Index", "Modality Index", "Dataset"],
                      value_vars=["Total Accuracy"],
                      value_name="Accuracy").sort_values("Modality Index")

    graph_generic(title=f"{dataset_to_label_map[dataset]} Accuracy by Model",
                  data=compare,
                  group_by="Model Index",
                  sort_by="Modality Index",
                  chart_labels=[model_to_label_map[model] for model in models],
                  modalities=modalities,
                  figsize=(len(modalities) * length, 4), plot_size=(1, length),
                  output_path=f"DatasetAnalysis/{dataset}-{save_discriminator}-Results")

    # CoT Quant results
    compare = pd.melt(data,
                      id_vars=["Model", "Modality", "Model Index", "Modality Index", "Dataset"],
                      value_vars=["Percent of Answers Containing CoT"],
                      value_name="Percentage").sort_values("Modality Index")

    graph_generic(title=F"{dataset_to_label_map[dataset]} CoT Occurrence by Model",
                  data=compare,
                  y="Percentage",
                  y_label="Percent of Answers Containing CoT",
                  group_by="Model Index",
                  sort_by="Modality Index",
                  chart_labels=[model_to_label_map[model] for model in models],
                  modalities=modalities,
                  figsize=(len(modalities) * length, 4), plot_size=(1, length),
                  output_path=f"DatasetAnalysis/{dataset}-{save_discriminator}-CoT-Quant")


def graph_non_stepwise(model: str, num_plots: int = 5):
    modalities = ["zero_shot", "zero_shot_cot", "suppressed_cot", "explanation_first", "answer_first"]
    label = model_to_label_map[model]
    length = (len(modalities))
    # Graph GPT-4 All dataset results
    results = search_metadata(models=[model], modalities=modalities)

    data = pd.melt(results,
                   id_vars=["Model", "Modality", "Dataset Index", "Modality Index", "Dataset"],
                   value_vars=["Total Accuracy"],
                   value_name="Accuracy")

    graph_dataset_comparison(title=f"{label} Performance, All Modalities, All Datasets",
                             data=data,
                             modalities=modalities,
                             figsize=(2 * num_plots, 4),
                             plot_size=(1, num_plots),
                             sort_by="Modality Index",
                             add_legend=True,
                             output_path=f"{model}/{model}-Full-Results")

    # CoT Quant results
    data = pd.melt(results,
                   id_vars=["Model", "Modality", "Dataset Index", "Modality Index", "Dataset"],
                   value_vars=["Percent of Answers Containing CoT"],
                   var_name="Metric",
                   value_name="Percentage").sort_values("Modality Index")
    #
    # graph_cot_data(title=f"{label} CoT Quantification, All Modalities, All Datasets",
    #                data=data,
    #                figsize=(2 * num_plots, 4),
    #                plot_size=(1, num_plots),
    #                hue="Metric",
    #                x="Modality",
    #                xtick_labels=[modality_to_label_map[modality] for modality in modalities],
    #                chart_labels=["MultiArith", "GSM8k", "Aqua-RAT", "Coin Flip", "MMLU"],
    #                group_by="Dataset Index",
    #                output_path=f"{model}/{model}-CoT-Quant")

    graph_dataset_comparison(title=f"{label} CoT Quantification, All Modalities, All Datasets",
                             data=data,
                             y="Percentage",
                             ylabel="Percent of Answers Containing CoT",
                             modalities=modalities,
                             figsize=(2 * num_plots, 4),
                             plot_size=(1, num_plots),
                             sort_by="Modality Index",
                             add_legend=True,
                             output_path=f"{model}/{model}-Full-CoT-Quant")


def graph_stepwise(model: str, max_steps: int = 9):
    label = model_to_label_map[model]
    modalities = ["zero_shot", "zero_shot_cot", "suppressed_cot", "explanation_first", "answer_first"]

    results = search_metadata(models=[model],
                              modalities=modalities,
                              datasets=["stepwise"])
    results = results.loc[results['Steps'] <= max_steps]

    data = pd.melt(frame=results,
                   id_vars=["Model", "Modality", "Modality Index", "Dataset", "Steps", "ci_upper", "ci_lower"],
                   value_vars=["Total Accuracy"],
                   value_name="Accuracy").sort_values(by=["Steps", "Modality Index"])

    graph_stepwise_comparison(title=f"{label} Performance on Stepwise Dataset",
                              data=data,
                              modalities=modalities,
                              output_path=f"{model}/{model}-{max_steps}-Step-Results")

    # CoT Quantification
    data = pd.melt(frame=results,
                   id_vars=["Steps", "Modality", "Modality Index", "Model", "Model Index", "ci_upper", "ci_lower"],
                   value_vars=["Percent of Answers Containing CoT"],
                   var_name="Metric",
                   value_name="Percentage").sort_values(by=["Steps", "Modality Index"])

    graph_stepwise_comparison(title=f"{label} CoT Quantification, Stepwise Dataset",
                              data=data,
                              modalities=modalities,
                              y="Percentage",
                              ci=False,
                              output_path=f"{model}/{model}-{max_steps}-Step-CoT-Quant")


def prompt_comparison():
    frames = list()

    frames.append(pd.DataFrame({
        "Model": "text-davinci-002",
        "Model Index": model_index_map["text-davinci-002"],
        "Extraction": "Initial",
        "Extraction Index": 1,
        "Dataset": "gsm8k",
        "Total Accuracy": [15.0]
    }))

    frames.append(pd.DataFrame({
        "Model": "text-davinci-002",
        "Model Index": model_index_map["text-davinci-002"],
        "Extraction": "Simplified",
        "Extraction Index": 2,
        "Dataset": "gsm8k",
        "Total Accuracy": [12.17]
    }))

    frames.append(pd.DataFrame({
        "Model": "text-davinci-002",
        "Model Index": model_index_map["text-davinci-002"],
        "Extraction": "Simplified In-Brackets",
        "Extraction Index": 3,
        "Dataset": "gsm8k",
        "Total Accuracy": [11.0]
    }))

    frames.append(pd.DataFrame({
        "Model": "gpt-3.5-turbo",
        "Model Index": model_index_map["gpt-3.5-turbo"],
        "Extraction": "Initial",
        "Extraction Index": 1,
        "Dataset": "gsm8k",
        "Total Accuracy": [76.3]
    }))

    frames.append(pd.DataFrame({
        "Model": "gpt-3.5-turbo",
        "Model Index": model_index_map["gpt-3.5-turbo"],
        "Extraction": "Simplified",
        "Extraction Index": 2,
        "Dataset": "gsm8k",
        "Total Accuracy": [81.16]
    }))

    frames.append(pd.DataFrame({
        "Model": "gpt-3.5-turbo",
        "Model Index": model_index_map["gpt-3.5-turbo"],
        "Extraction": "Simplified In-Brackets",
        "Extraction Index": 3,
        "Dataset": "gsm8k",
        "Total Accuracy": [78.5]
    }))

    orig = pd.concat(frames)
    data = pd.melt(orig,
                   id_vars=["Model", "Model Index", "Extraction", "Extraction Index", "Dataset"],
                   value_vars=["Total Accuracy"],
                   value_name="Accuracy").sort_values("Dataset")

    graph_generic(title="Comparison of Answer Extraction Techniques By Model, GSM8k",
                  x="Extraction Index",
                  y="Accuracy",
                  data=data,
                  group_by="Model Index",
                  output_path=r"gsm8k-extraction-comparison",
                  chart_labels=["text-davinci-002", "GPT 3.5", "GPT-4"],
                  x_labels=["Initial", "Simplified", "In-Brackets"],
                  figsize=(5, 3),
                  plot_size=(1, 2))

    frames = list()

    frames.append(pd.DataFrame({
        "Model": "text-davinci-002",
        "Model Index": model_index_map["text-davinci-002"],
        "Modality": "zero_shot",
        "Extraction": "Initial",
        "Extraction Index": 1,
        "Dataset": "multiarith",
        "Total Accuracy": [26.0]
    }))

    frames.append(pd.DataFrame({
        "Model": "text-davinci-002",
        "Model Index": model_index_map["text-davinci-002"],
        "Modality": "zero_shot",
        "Extraction": "Simplified",
        "Extraction Index": 2,
        "Dataset": "multiarith",
        "Total Accuracy": [21.2]
    }))

    frames.append(pd.DataFrame({
        "Model": "text-davinci-002",
        "Model Index": model_index_map["text-davinci-002"],
        "Modality": "zero_shot",
        "Extraction": "Simplified In-Brackets",
        "Extraction Index": 3,
        "Dataset": "multiarith",
        "Total Accuracy": [15.5]
    }))

    frames.append(pd.DataFrame({
        "Model": "gpt-3.5-turbo",
        "Model Index": model_index_map["gpt-3.5-turbo"],
        "Modality": "zero_shot",
        "Extraction": "Initial",
        "Extraction Index": 1,
        "Dataset": "multiarith",
        "Total Accuracy": [90.16]
    }))

    frames.append(pd.DataFrame({
        "Model": "gpt-3.5-turbo",
        "Model Index": model_index_map["gpt-3.5-turbo"],
        "Modality": "zero_shot",
        "Extraction": "Simplified",
        "Extraction Index": 2,
        "Dataset": "multiarith",
        "Total Accuracy": [95.3]
    }))

    frames.append(pd.DataFrame({
        "Model": "gpt-3.5-turbo",
        "Model Index": model_index_map["gpt-3.5-turbo"],
        "Extraction Index": 3,
        "Modality": "zero_shot",
        "Extraction": "Simplified In-Brackets",
        "Dataset": "multiarith",
        "Total Accuracy": [91.5]
    }))

    orig = pd.concat(frames)
    data = pd.melt(orig,
                   id_vars=["Model", "Model Index", "Extraction", "Extraction Index", "Dataset"],
                   value_vars=["Total Accuracy"],
                   value_name="Accuracy").sort_values("Dataset")

    graph_generic(title="Comparison of Answer Extraction Techniques By Model, MultiArith",
                  data=data,
                  group_by="Model Index",
                  output_path="multiarith-extraction-comparison",
                  chart_labels=["text-davinci-002", "GPT 3.5", "GPT-4"],
                  x_labels=["Initial", "Simplified", "In-Brackets"],
                  figsize=(5, 3),
                  x="Extraction Index",
                  plot_size=(1, 2), y="Accuracy")
