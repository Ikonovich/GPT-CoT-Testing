import copy
import json
import os

import numpy as np

import pandas
import seaborn as sns
import matplotlib.pyplot as plt

import config
from config import RESULTS_FOLDER
from data_utils import collate_results, write_json, get_cross_modality_results, format_name


def create_bar_chart(graph_title: str, x_title: str, y_title: str, data: list[float], column_labels: list,
                     save_path: str = None):

    # Return if no data has been passed. This allows automated graph creation to not fail on uncollected data.
    if len(data) == 0 or len(column_labels) == 0:
        return

    # Creates a bar chart given the specific percentages and x/y labels
    fig, ax = plt.subplots()

    sns.set_style("whitegrid")
    ax = sns.barplot(x=column_labels, y=data, palette=['blue' for i in range(len(column_labels))])

    patches = ax.patches
    for i in range(len(patches)):
        x = patches[i].get_x() + patches[i].get_width() / 2
        y = patches[i].get_height() + .05
        ax.annotate('{:.1f}%'.format(data[i]), (x, y), ha='center')

    ax.set_ylabel(y_title)
    ax.set_xlabel(x_title)
    ax.set_title(graph_title)

    ax.set_ylim(0, 110)
    ax.set_xticks(np.arange(len(column_labels)))
    ax.set_xticklabels(column_labels)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    # plt.show()


def graph_results(graph_title: str, x_title: str, y_title: str, column_labels: list, data: list[float],
                  save_path: str = None):

    if len(data) == 0:
        return

    fig, ax = plt.subplots()

    # Plot the data on the axis
    ax.plot(column_labels, data, label="Stepwise Accuracy")

    # Customize the plot
    ax.set_title(graph_title)
    ax.set_xlabel(x_title)
    ax.set_ylabel(y_title)
    ax.legend()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


def generate_table(row_names: list[str], column_names: list[str], data: np.ndarray):
    # Define your model names and datasets
    datasets = column_names
    model_names = row_names

    # Initialize an empty dictionary to store the results
    results = dict()

    # Replace the following sample data with your actual data
    # Sample data format: results[model_name] = [score_on_dataset_1, score_on_dataset_2, ..., score_on_dataset_n]

    # Create the table visualization
    fig, ax = plt.subplots()

    # Hide axes
    ax.axis('off')
    ax.axis('tight')

    # Create the table
    table = ax.table(cellText=data, rowLabels=model_names, colLabels=datasets, loc='center', cellLoc='center')

    # Set the fontsize for the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)

    # Show the table
    plt.show()


def create_graphs():
    # Creates graphs from all tested data so far

    # Collate metadata
    models = copy.deepcopy(config.chat)
    models.extend(config.completion)

    for model in models:
        initial_prompt, simple_prompt = collate_results(model=model)
        prompt_data = {"Initial": initial_prompt, "Simple": simple_prompt}

        ip_path = os.path.join(RESULTS_FOLDER, model, "Initial-Prompt-Results.json")
        sp_path = os.path.join(RESULTS_FOLDER, model, "Simple-Prompt-Results.json")

        write_json(data=initial_prompt, filepath=ip_path)
        write_json(data=simple_prompt, filepath=sp_path)

        for prompt in prompt_data:
            data = prompt_data[prompt]
            # Create and save multiarith and GSM8k graphs
            for dataset in ["MultiArith", "GSM8k"]:
                # Create and save graphs
                file_name = f"{dataset}-{prompt}-{model}".replace(".", "")
                save_path = os.path.join(config.RESULTS_FOLDER, model, file_name)
                labels, results = get_cross_modality_results(data=data[dataset.lower()])

                create_bar_chart(x_title="Test Modality", y_title="Accuracy",
                                 graph_title=f"{format_name(name=model, is_model=True)}, {dataset}, {prompt} Prompt",
                                 column_labels=labels, data=results, save_path=save_path)

            for modality in data["stepwise"]:
                # Create and save stepwise graphs
                step_data = data["stepwise"][modality]
                if step_data[0] is None:
                    break

                heights = [-10000.0 for i in range(9)]
                for item in step_data:
                    steps = item["Step Count"]
                    accuracy = item["Accuracy"]
                    heights[steps - 1] = accuracy

                file_name = f"Stepwise-{modality}-{prompt}-{model}".replace(".", "")
                save_path = os.path.join(config.RESULTS_FOLDER, model, file_name)
                create_bar_chart(graph_title=f"{format_name(name=model, is_model=True)}, Stepwise, {format_name(modality)},"
                                             f" {prompt} Prompt",
                                 x_title="Number of Steps", y_title="Accuracy", data=heights,
                                 column_labels=[i for i in range(1, 10)], save_path=save_path)

    plt.close('all')
