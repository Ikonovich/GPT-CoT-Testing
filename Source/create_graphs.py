import os

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from config import model_index_map, GRAPHS_FOLDER
from data_utils import search_metadata
from graph_utils import graph_generic, graph_stepwise_comparison, graph_dataset_comparison, generate_singular_plot, \
    modality_to_color_map

dataset_to_label_map = {"multiarith": "MultiArith", "gsm8k": "GSM8k", "aqua": "Aqua-RAT", "coin_flip": "Coin Flip",
                        "mmlu-combined": "MMLU"}
model_to_label_map = {"gpt-3.5-turbo": "GPT-3.5 Turbo", "gpt-4": "GPT-4", "text-davinci-002": "GPT-3",
                      "gpt-4-32k": "GPT-4-32k"}


def create_graphs():
    extraction_comparison()

    single_dataset_results(dataset="multiarith")
    single_dataset_results(dataset="aqua")
    single_dataset_results(dataset="mmlu-combined",
                           models=["gpt-3.5-turbo", "gpt-4"],
                           save_discriminator="Two-Model")
    single_dataset_results(dataset="gsm8k")
    single_dataset_results(dataset="coin_flip",
                           models=["gpt-3.5-turbo", "gpt-4"],
                           modalities=["zero_shot", "zero_shot_cot", "suppressed_cot"],
                           save_discriminator="Supp-Two-Model")

    single_dataset_results(dataset="coin_flip")
    single_dataset_results(dataset="mmlu-combined")

    graph_non_stepwise(model="gpt-4")
    graph_non_stepwise(model="gpt-3.5-turbo")
    graph_non_stepwise(model="text-davinci-002")

    graph_stepwise(model="gpt-4", max_steps=23)
    graph_stepwise(model="gpt-3.5-turbo", max_steps=23)
    graph_stepwise(model="text-davinci-002", max_steps=9)
    graph_stepwise(model="gpt-4", max_steps=9)
    graph_stepwise(model="gpt-3.5-turbo", max_steps=9)


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
                             figsize=(2 * num_plots, 3),
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


def extraction_comparison():
    frames = list()

    # -- Aqua zero shot frames
    frames.append(pd.DataFrame({
        "Model": "gpt-4",
        "Model Index": model_index_map["gpt-4"],
        "Extraction": "In-Brackets",
        "Extraction Index": 2,
        "Dataset": "aqua",
        "Dataset Index": 2,
        "Total Accuracy": [60.167],
        "Modality": "zero_shot"
    }))

    frames.append(pd.DataFrame({
        "Model": "gpt-4",
        "Model Index": model_index_map["gpt-4"],
        "Extraction": "Two-Stage",
        "Extraction Index": 1,
        "Dataset": "aqua",
        "Dataset Index": 2,
        "Total Accuracy": [62.0],
        "Modality": "zero_shot"
    }))

    frames.append(pd.DataFrame({
        "Model": "gpt-3.5-turbo",
        "Model Index": model_index_map["gpt-3.5-turbo"],
        "Extraction": "In-Brackets",
        "Extraction Index": 2,
        "Dataset": "aqua",
        "Dataset Index": 2,
        "Total Accuracy": [53.5461],
        "Modality": "zero_shot"
    }))

    frames.append(pd.DataFrame({
        "Model": "gpt-3.5-turbo",
        "Model Index": model_index_map["gpt-3.5-turbo"],
        "Extraction": "two-stage",
        "Extraction Index": 1,
        "Dataset": "aqua",
        "Dataset Index": 2,
        "Total Accuracy": [52.5],
        "Modality": "zero_shot"
    }))


    # -- GSM8k zero shot frames
    frames.append(pd.DataFrame({
        "Model": "gpt-4",
        "Model Index": model_index_map["gpt-4"],
        "Extraction": "two-stage",
        "Extraction Index": 1,
        "Dataset": "gsm8k",
        "Dataset Index": 0,
        "Total Accuracy": [92.0],
        "Modality": "zero_shot"
    }))

    frames.append(pd.DataFrame({
        "Model": "gpt-4",
        "Model Index": model_index_map["gpt-4"],
        "Extraction": "in-brackets",
        "Extraction Index": 2,
        "Dataset": "gsm8k",
        "Dataset Index": 0,
        "Total Accuracy": [93.1667],
        "Modality": "zero_shot"
    }))

    frames.append(pd.DataFrame({
        "Model": "gpt-3.5-turbo",
        "Model Index": model_index_map["gpt-3.5-turbo"],
        "Extraction": "in-brackets",
        "Extraction Index": 2,
        "Dataset": "gsm8k",
        "Dataset Index": 0,
        "Total Accuracy": [78.8333],
        "Modality": "zero_shot"
    }))

    frames.append(pd.DataFrame({
        "Model": "gpt-3.5-turbo",
        "Model Index": model_index_map["gpt-3.5-turbo"],
        "Extraction": "two-stage",
        "Extraction Index": 1,
        "Dataset": "gsm8k",
        "Dataset Index": 0,
        "Total Accuracy": [80.16667],
        "Modality": "zero_shot"
    }))

    # -- mmlu zero shot frames
    frames.append(pd.DataFrame({
        "Model": "gpt-4",
        "Model Index": model_index_map["gpt-4"],
        "Extraction": "two-stage",
        "Extraction Index": 1,
        "Dataset": "mmlu-combined",
        "Dataset Index": 3,
        "Total Accuracy": [57.866],
        "Modality": "zero_shot"
    }))

    frames.append(pd.DataFrame({
        "Model": "gpt-4",
        "Model Index": model_index_map["gpt-4"],
        "Extraction": "in-brackets",
        "Extraction Index": 2,
        "Dataset": "mmlu-combined",
        "Dataset Index": 3,
        "Total Accuracy": [57.6],
        "Modality": "zero_shot"
    }))

    frames.append(pd.DataFrame({
        "Model": "gpt-3.5-turbo",
        "Model Index": model_index_map["gpt-3.5-turbo"],
        "Extraction": "in-brackets",
        "Extraction Index": 2,
        "Dataset": "mmlu-combined",
        "Dataset Index": 3,
        "Total Accuracy": [8.9783],
        "Modality": "zero_shot"
    }))

    frames.append(pd.DataFrame({
        "Model": "gpt-3.5-turbo",
        "Model Index": model_index_map["gpt-3.5-turbo"],
        "Extraction": "two-stage",
        "Extraction Index": 1,
        "Dataset": "mmlu-combined",
        "Dataset Index": 3,
        "Total Accuracy": [52.0],
        "Modality": "zero_shot"
    }))

    # -- coin flip zero shot frames
    frames.append(pd.DataFrame({
        "Model": "gpt-4",
        "Model Index": model_index_map["gpt-4"],
        "Extraction": "two-stage",
        "Extraction Index": 1,
        "Dataset": "coin_flip",
        "Dataset Index": 1,
        "Total Accuracy": [55.4],
        "Modality": "zero_shot"
    }))

    frames.append(pd.DataFrame({
        "Model": "gpt-4",
        "Model Index": model_index_map["gpt-4"],
        "Extraction": "in-brackets",
        "Extraction Index": 2,
        "Dataset": "coin_flip",
        "Dataset Index": 1,
        "Total Accuracy": [55.6],
        "Modality": "zero_shot"
    }))

    frames.append(pd.DataFrame({
        "Model": "gpt-3.5-turbo",
        "Model Index": model_index_map["gpt-3.5-turbo"],
        "Extraction": "in-brackets",
        "Extraction Index": 2,
        "Dataset": "coin_flip",
        "Dataset Index": 1,
        "Total Accuracy": [39.6],
        "Modality": "zero_shot"
    }))

    frames.append(pd.DataFrame({
        "Model": "gpt-3.5-turbo",
        "Model Index": model_index_map["gpt-3.5-turbo"],
        "Extraction": "two-stage",
        "Extraction Index": 1,
        "Dataset": "coin_flip",
        "Dataset Index": 1,
        "Total Accuracy": [46.6],
        "Modality": "zero_shot"
    }))

    orig = pd.concat(frames)

    modalities = ["zero_shot"]

    # gpt4_data = data[(data["Model"] == "gpt-4")]

    datasets = ["gsm8k", "coin_flip", "aqua", "mmlu-combined"]
    data = search_metadata(models=["gpt-4"],
                           modalities=["zero_shot"],
                           datasets=datasets,
                           extraction_types=["two-stage", "in-brackets", "two-stage-style-two"],
                           include_secondary=True)

    # data = pd.melt(data,
    #                id_vars=["Dataset Index", "Extraction Type", "Dataset"],
    #                value_vars=["Total Accuracy"],
    #                value_name="Accuracy").sort_values("Model")
    data = pd.melt(data,
                   id_vars=["Model", "Modality", "Dataset Index", "Extraction Type", "Dataset"],
                   value_vars=["Total Accuracy"],
                   value_name="Accuracy")

    palette = [modality_to_color_map[i] for i in modalities]
    fig, ax = plt.subplots(1, 4, figsize=(4, 4), layout="constrained")
    fig.suptitle("Extraction Comparison, GPT-4")


    # sns.set_theme()
    sns.set_style("whitegrid")

    chart_labels = ["GSM8k", "Coin Flip", "AQuA-RAT", "MMLU"]
    i = 0
    frame = data.groupby("Dataset Index")
    for item in frame:
        generate_singular_plot(ax=ax, x="Extraction Type", y="Accuracy", coordinate=i,
                               data=item[1], xtick_labels=None, title=chart_labels[i],
                               palette=palette)

        i += 1
    plt.savefig(os.path.join(GRAPHS_FOLDER, "gpt-4-Extraction Comparison.svg"), format='svg', dpi=1200)
    plt.close("all")

