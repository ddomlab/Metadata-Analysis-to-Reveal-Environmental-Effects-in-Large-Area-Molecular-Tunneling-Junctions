import pandas as pd
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

HERE: Path = Path(__file__).resolve().parent
DATASETS: Path = HERE.parent/ "datasets"
training_data_dir: Path = DATASETS/"training_dataset"
training_data_dir: Path = DATASETS/"training_dataset"
DATA = pd.read_csv(training_data_dir/"substrate_training_data.csv")
saving_dir: Path  = HERE/"general_plots" 




def plot_single_distribution_plot(data: pd.DataFrame, 
                                  feature: str, 
                                  title: str, 
                                  x_label: str, 
                                  fname: str, 
                                  transform_function: callable = None) -> None:
    """
    Plots a single distribution and box plot for a given feature with optional transformation.

    Args:
        data (pd.DataFrame): Data containing the feature to be plotted.
        feature (str): Name of the feature to plot.
        title (str): Title of the plot.
        x_label (str): Label for the x-axis.
        fname (str): Name of the output file.
        saving_dir (Path): Directory to save the plot.
        transform_function (callable): Transformation function to apply to the data (e.g., np.log2).
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Apply transformation function if provided
    transformed_data = transform_function(data[feature]) if transform_function else data[feature]

    # Plot histogram with optional KDE
    sns.histplot(transformed_data, kde=True, ax=ax)
    ax.set_xlabel(x_label)  # Set the label for x-axis
    ax.set_ylabel('Occurrence')  # Set the label for y-axis
    ax.set_title(title)  # Set the title for the plot

    # Create inset box plot
    box_inset = ax.inset_axes([0.01, -0.35, 0.99, 0.2])  # Adjust position for the inset box plot
    sns.boxplot(x=transformed_data, ax=box_inset, color='#259AC1')

    # Customize the inset box plot
    box_inset.set(yticks=[], xlabel=None)  # Remove y-ticks and x-label from the box plot
    box_inset.tick_params(axis='x', labelsize=10)

    # Ensure saving directory exists
    os.makedirs(saving_dir, exist_ok=True)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(saving_dir / f"{fname}.png", dpi=600)

    # Close the plot to free memory
    plt.close()
 



plot_single_distribution_plot(DATA, 'water content', transform_function=np.log10, title="distribution of water content",
                               x_label="log10(water content)", fname= "Distribution plot of log10(water content)")