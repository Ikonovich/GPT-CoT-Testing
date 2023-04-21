import numpy as np
import matplotlib.pyplot as plt


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
