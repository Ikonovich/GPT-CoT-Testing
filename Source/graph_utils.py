import os

import numpy as np
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame

from config import GRAPHS_FOLDER

colors = sns.color_palette('colorblind')[:5]
# modality_to_color_map = {"zero_shot": "blue", "zero_shot_cot": "red", "suppressed_cot": "black",
#                        "explanation_first": "orange", "answer_first": "green"}
modality_to_color_map = {"zero_shot": colors[0], "zero_shot_cot": colors[1], "suppressed_cot": colors[2],
                         "explanation_first": colors[3], "answer_first": colors[4]}

modality_to_label_map = {"zero_shot": "Zero Shot", "zero_shot_cot": "Zero Shot CoT", "suppressed_cot": "Suppressed CoT",
                         "explanation_first": "Explanation First", "answer_first": "Answer First"}

modality_to_dash_map = {"zero_shot": (0, (1, 0)), "zero_shot_cot": (0, (1, 3)), "suppressed_cot": (0, (5, 5)),
                        "explanation_first": (0, (1, 5, 5, 5)), "answer_first": (0, (5, 10))}


def generate_singular_plot(ax, data: DataFrame, x: str, y: str, coordinate: int | tuple[int, int] | None,
                           title: str | None, xtick_labels: list[str] | None, y_label: str = None,
                           palette: list[str] = None):
    if coordinate is None:
        sns.barplot(x=x, y=y, palette=palette, ax=ax, data=data)

        patches = ax.patches
        for patch in patches:
            percentage = '{:.1f}%'.format(100 * patch.get_height() / 100)
            x = patch.get_x() + patch.get_width() / 2
            y = patch.get_height() + .05
            ax.annotate(percentage, (x, y), ha='center')

        ax.set(title=title, xlabel=None, ylabel=y_label)
        ax.set_ylim(0, 110)
        ax.set_xticks(np.arange(len(xtick_labels)))
        ax.set_xticklabels(xtick_labels, rotation=45)

    else:
        sns.barplot(x=x, y=y, palette=palette, ax=ax[coordinate], data=data)
        patches = ax[coordinate].patches

        for patch in patches:
            percentage = '{:.0f}%'.format(round(patch.get_height()))
            x = patch.get_x() + patch.get_width() / 2
            y = patch.get_height() + .05
            ax[coordinate].annotate(percentage, (x, y), ha='center')

        ax[coordinate].set(title=title, xlabel=None, ylabel=y_label)
        ax[coordinate].set_ylim(0, 110)

        if xtick_labels is None:
            ax[coordinate].set_xticklabels([])
        else:
            ax[coordinate].set_xticks(np.arange(len(xtick_labels)))
            ax[coordinate].set_xticklabels(xtick_labels)


def graph_dataset_comparison(title: str, data: DataFrame,
                             output_path: str,
                             modalities: list[str],
                             figsize: tuple[int, int] = (15, 7),
                             plot_size: tuple[int, int] | None = (2, 2),
                             sort_by: str | None = None,
                             add_legend: bool = False):

    chart_labels = ["MultiArith", "GSM8k", "Aqua-RAT", "Coin Flip", "MMLU"]
    palette = [modality_to_color_map[i] for i in modalities]
    x_labels = [modality_to_label_map[i] for i in modalities]

    fig, ax = plt.subplots(plot_size[0], plot_size[1], figsize=figsize, layout="constrained", sharey=True)
    plt.tick_params(labelright=True)
    plt.ylabel("Accuracy")
    fig.suptitle(title)

    # sns.set_theme()
    sns.set_style("whitegrid")

    i = 0
    frame = data.groupby("Dataset Index")
    for item in frame:
        datum = item[1]
        if sort_by is not None:
            datum = datum.sort_values(sort_by)

        if plot_size == (1, 1):
            coord = None
        elif plot_size[0] == 1:
            coord = i
        elif i < plot_size[1]:
            coord = (0, i)
        else:
            coord = (1, i - plot_size[1])

        generate_singular_plot(ax=ax, x="Modality", y="Accuracy", coordinate=coord,
                               data=datum, xtick_labels=None, title=chart_labels[i],
                               y_label=None, palette=palette)
        i += 1

    while i < plot_size[0] * plot_size[1]:
        if i < plot_size[1]:
            coord = (0, i)
        else:
            coord = (1, i - plot_size[1])
        fig.delaxes(ax[coord])
        i += 1

    # Create and add the legend
    if add_legend:
        handles = list()
        for modality in modalities:
            patch = mpatches.Patch(label=modality_to_label_map[modality], color=modality_to_color_map[modality])
            handles.append(patch)
        plt.legend(handles=handles)

    plt.savefig(os.path.join(GRAPHS_FOLDER, output_path + ".svg"), format='svg', dpi=1200)
    plt.close("all")


def graph_generic(title: str, data: DataFrame, groupby: str, output_path: str,
                  chart_labels: list[str],
                  figsize: tuple[int, int] = (15, 7),
                  plot_size: tuple[int, int] | None = (2, 2),
                  y_label="Accuracy",
                  x_labels: list[str] | None = None,
                  x: str = "Modality",
                  y: str = "Accuracy",
                  sort_by=None,
                  palette=None):
    fig, ax = plt.subplots(plot_size[0], plot_size[1], figsize=figsize, layout="constrained")
    fig.suptitle(title)

    # sns.set_theme()
    sns.set_style("whitegrid")

    i = 0
    frame = data.groupby(groupby)
    for item in frame:
        datum = item[1]
        if sort_by is not None:
            datum = datum.sort_values(sort_by)

        if plot_size == (1, 1):
            coord = None
        elif plot_size[0] == 1:
            coord = i
        elif i < plot_size[1]:
            coord = (0, i)
        else:
            coord = (1, i - plot_size[1])

        generate_singular_plot(ax=ax, x=x, y=y, coordinate=coord,
                               data=datum, xtick_labels=x_labels, title=chart_labels[i],
                               y_label=y_label, palette=palette)
        i += 1

    while i < plot_size[0] * plot_size[1]:
        if i < plot_size[1]:
            coord = (0, i)
        else:
            coord = (1, i - plot_size[1])
        fig.delaxes(ax[coord])
        i += 1

    plt.savefig(os.path.join(GRAPHS_FOLDER, output_path + ".svg"), format='svg', dpi=1200)
    plt.close("all")


def graph_stepwise_comparison(data: DataFrame, title: str, modalities: list[str], output_path: str):
    # Create a lineplot for each modality
    for i in range(len(modalities)):
        modality = modalities[i]
        color = modality_to_color_map[modality]

        df_modality = data[data['Modality'] == modality]
        ax = sns.lineplot(x='Step', y='Accuracy', data=df_modality,
                          color=color,
                          label=modality_to_label_map[modality])

        ax.lines[i].set_linestyle(modality_to_dash_map[modality])
        ax.legend().get_lines()[i].set_linestyle(modality_to_dash_map[modality])

        # Add the confidence intervals
        plt.fill_between(df_modality['Step'], df_modality["ci_lower"], df_modality["ci_upper"], color=color,
                         alpha=.1)

    plt.xlabel("Number of Steps Per Problem")
    plt.savefig(os.path.join(GRAPHS_FOLDER, output_path + ".svg"), format='svg', dpi=1200)
    plt.close("all")


def graph_cot_data(title: str, data: DataFrame, figsize: tuple[int, int], plot_size: tuple[int, int] | None,
                   output_path: str, x: str, hue: str | None = "Metric", groupby: str = "Modality Index",
                   xtick_labels: list[str] | None = None,
                   chart_labels: list[str] | None = None):
    modalities = ["Zero Shot", "Zero Shot CoT", "Suppressed CoT", "Explanation First", "Answer First"]
    datasets = ["MultiArith", "GSM8k", "Aqua-RAT", "Coin Flip", "MMLU"]
    if chart_labels is None:
        if groupby == "Modality Index":
            chart_labels = modalities
        else:
            chart_labels = datasets
    if xtick_labels is None:
        if groupby == "Modality Index":
            xtick_labels = datasets
        else:
            xtick_labels = modalities

    # Store the first coord for setting the legend
    first_coord = None

    # Set the palette
    palette = ['blue', 'orange', 'black', 'gray']

    fig, ax = plt.subplots(plot_size[0], plot_size[1], figsize=figsize, layout="constrained")
    fig.suptitle(title)

    frame = data.groupby(groupby)
    sns.set_style("whitegrid")

    i = 0
    for item in frame:
        datum = item[1]
        if plot_size == (1, 1):
            coord = None
        elif plot_size[0] == 1:
            coord = i
        elif i < plot_size[1]:
            coord = (0, i)
        else:
            coord = (1, i - plot_size[1])

        if first_coord is None:
            first_coord = coord

        sns.barplot(
            data=datum,
            x=x,
            y="Percentage",
            hue=hue,
            palette=palette,
            hue_order=['Answers Containing CoT', 'CoT Accuracy', 'Non-CoT Accuracy', 'Total Accuracy'],
            ax=ax[coord]
        )
        ax[coord].get_legend().remove()
        ax[coord].set(title=chart_labels[i])
        ax[coord].set_xticks(np.arange(len(xtick_labels)))
        ax[coord].set_xticklabels(xtick_labels, rotation=45)
        i += 1

    while i < plot_size[0] * plot_size[1]:
        if i < plot_size[1]:
            coord = (0, i)
        else:
            coord = (1, i - plot_size[1])
        fig.delaxes(ax[coord])
        i += 1

    # Create a single legend for the entire figure
    handles, labels = ax[first_coord].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.90), title="Metrics")

    plt.savefig(os.path.join(GRAPHS_FOLDER, output_path + ".svg"), format='svg', dpi=1200)
    plt.close("all")