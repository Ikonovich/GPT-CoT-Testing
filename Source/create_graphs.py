import os

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from config import model_index_map, GRAPHS_FOLDER
from utils.data_utils import search_metadata
from utils.graph_utils import graph_generic, graph_stepwise_comparison, graph_dataset_comparison, \
    generate_singular_plot, \
    modality_to_color_map, graph_modified_cot

dataset_to_label_map = {"multiarith": "MultiArith", "gsm8k": "GSM8k", "aqua": "Aqua-RAT", "coin_flip": "Coin Flip",
                        "mmlu-combined": "MMLU", "stepwise": "Stepwise", "unmodified": "Unmodified CoT",
                     "First-Step-Single-Mod-Off-By-One-Keep-Last": "Mod-CoT: First Step",
                        "Middle-Step-Single-Mod-Off-By-One-Keep-Last": "Mod-CoT: Middle Step",
                     "Last-Step-Single-Mod-Off-By-One-Keep-Last": "Mod-CoT: Last Step"}
model_to_label_map = {"gpt-3.5-turbo": "GPT-3.5 Turbo", "gpt-4": "GPT-4", "text-davinci-002": "GPT-3",
                      "gpt-4-32k": "GPT-4-32k", "goat": "Goat"}


def create_graphs():
    # extraction_comparison()

    modified_cot(model="gpt-4")
    modified_cot(model="gpt-3.5-turbo")
    modified_cot(model="text-davinci-002")
    modified_cot(model="goat")
    single_dataset_results(dataset="multiarith")
    single_dataset_results(dataset="aqua")
    single_dataset_results(dataset="mmlu-combined",
                           models=["gpt-3.5-turbo", "gpt-4"],
                           save_discriminator="Partial")
    # single_dataset_results(dataset="gsm8k")
    single_dataset_results(dataset="coin_flip",
                           models=["gpt-3.5-turbo", "gpt-4", "goat"],
                           modalities=["zero_shot", "zero_shot_cot", "suppressed_cot"],
                           save_discriminator="Supp-Partial")

    single_dataset_results(dataset="coin_flip")
    single_dataset_results(dataset="mmlu-combined")

    graph_non_stepwise(model="gpt-4")
    graph_non_stepwise(model="gpt-3.5-turbo")
    graph_non_stepwise(model="text-davinci-002")
    #graph_non_stepwise(model="goat")

    # graph_stepwise(model="gpt-4", max_steps=23)
    # graph_stepwise(model="gpt-3.5-turbo", max_steps=23)
    # graph_stepwise(model="text-davinci-002", max_steps=9)
    # graph_stepwise(model="text-davinci-002", max_steps=23)
    # graph_stepwise(model="gpt-4", max_steps=9)
    # graph_stepwise(model="gpt-3.5-turbo", max_steps=9)
    # graph_stepwise(model="goat", max_steps=9)


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


def graph_non_stepwise(model: str, modalities: list[str] = None, datasets: list[str] = None):

    if modalities is None:
        modalities = ["zero_shot", "zero_shot_cot", "suppressed_cot", "explanation_first", "answer_first"]
    if datasets is None:
        datasets = ["gsm8k", "multiarith", "mmlu-combined", "coin_flip", "aqua"]
    num_plots = len(datasets)
    label = model_to_label_map[model]
    # Graph GPT-4 All dataset results
    results = search_metadata(models=[model], modalities=modalities, datasets=datasets)

    data = pd.melt(results,
                   id_vars=["Model", "Modality", "Dataset Index", "Modality Index", "Dataset"],
                   value_vars=["Total Accuracy"],
                   value_name="Accuracy")

    graph_dataset_comparison(title=f"{label} Performance, All Modalities, All Datasets",
                             data=data,
                             modalities=modalities,
                             figsize=(2 * num_plots, 5),
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


def modified_cot(model: str, max_steps: int = 9):
    label = model_to_label_map[model]
    modalities = ["zero_shot"]
    datasets = ["unmodified",
                "First-Step-Single-Mod-Off-By-One-Keep-Last",
                "Middle-Step-Single-Mod-Off-By-One-Keep-Last",
                "Last-Step-Single-Mod-Off-By-One-Keep-Last"]

    results = search_metadata(models=[model],
                              modalities=modalities,
                              datasets=datasets)
    results = results.loc[results['Steps'] <= max_steps]

    data = pd.melt(frame=results,
                   id_vars=["Model", "Modality", "Dataset", "Steps", "ci_upper", "ci_lower"],
                   value_vars=["Total Accuracy"],
                   value_name="Accuracy").sort_values(by=["Steps", "Dataset"])

    graph_modified_cot(title=f"{label} Performance with Modified CoT",
                       data=data,
                       datasets=datasets,
                       output_path=f"{model}/{model}-{max_steps}-Modified-CoT-Results")


def extraction_comparison():
    dataframes = []
    datasets = ["gsm8k", "coin_flip", "aqua", "mmlu-college", "mmlu-high-school"]
    for dataset in datasets:
        # Determine extraction types based on dataset
        extraction_types = ["two-stage", "in-brackets",
                            "two-stage-multi-choice"]  # if dataset == "mmlu-combined" else ["two-stage", "in-brackets"]

        # Search metadata for gpt-3.5-turbo
        data_gpt3 = search_metadata(models=["gpt-3.5-turbo"],
                                    modalities=["zero_shot"],
                                    datasets=[dataset],
                                    extraction_types=extraction_types,
                                    include_secondary=True)
        # Melt data for gpt-3.5-turbo
        data_gpt3 = pd.melt(data_gpt3,
                            id_vars=["Model", "Modality", "Dataset Index", "Extraction Type", "Dataset"],
                            value_vars=["Total Accuracy"],
                            value_name="Accuracy")

        # Search metadata for gpt-4
        data_gpt4 = search_metadata(models=["gpt-4"],
                                    modalities=["zero_shot"],
                                    datasets=[dataset],
                                    extraction_types=extraction_types,
                                    include_secondary=True)

        # Melt data for gpt-4
        data_gpt4 = pd.melt(data_gpt4,
                            id_vars=["Model", "Modality", "Dataset Index", "Extraction Type", "Dataset"],
                            value_vars=["Total Accuracy"],
                            value_name="Accuracy")

        # Combine gpt-3.5-turbo and gpt-4 data
        data = pd.concat([data_gpt3, data_gpt4])

        dataframes.append(data)

    # Combine all datasets
    data = pd.concat(dataframes)

    # Set up the plot
    filename = "gpt-extraction-comparison.pdf"

    # Set up the plot
    sns.set_style("whitegrid")
    # Rename the labels in the "Extraction Type" column
    data['Extraction Type'] = data['Extraction Type'].replace({"in-brackets": "Brackets",
                                                               "two-stage": "2-Stage",
                                                               "two-stage-multi-choice": "2-Stage Style 2"})

    # Create faceted bar plot
    g = sns.catplot(x="Extraction Type", y="Accuracy", hue="Model", col="Dataset", data=data, kind="bar", height=4,
                    aspect=.7)

    # Save the plot
    g.savefig(os.path.join(GRAPHS_FOLDER, filename), format='pdf', dpi=1200)

    # Close the plot to free memory
    plt.close()
    print("generated")
