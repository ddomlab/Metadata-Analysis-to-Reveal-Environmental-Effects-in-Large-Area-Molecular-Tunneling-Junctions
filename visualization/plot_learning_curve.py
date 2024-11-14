import json
from itertools import product
from pathlib import Path
from typing import List, Optional
import os 
# import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

HERE: Path = Path(__file__).resolve().parent
RESULTS: Path = HERE.parent/ "results"

target_list = ["_log(J)at abs(.5) V"]



def clean_array(score_list):
    """
    Extracts the first element from the list if it's nested, 
    otherwise returns the value as is.
    
    Parameters:
    score_list (list or float): The score or list containing the score.
    
    Returns:
    np.array: Cleaned array of scores.
    """
    return np.array([score[0] if isinstance(score, list) else score for score in score_list])



def get_learning_data(
    file_path: Path,
) -> tuple[float, float]:
    """
    Args:
        root_dir: Root directory containing all results
        representation: Representation for which to get scores
        model: Model for which to get scores.
        score: Score to plot
        var: Variance to plot

    Returns:
        Average and variance of score
    """
    if not file_path.exists():
        features, model = None, None
        learning_df = None
    else:
        # for just scaler features
        model:str = file_path.parent.name
        features:str = file_path.name.split(")")[0].replace("(", "")

       
        with open(file_path, "r") as f:
            data = json.load(f)
        train_size = data['aggregated_generalizability_scores']["train_sizes_fraction"]
        train_scores_mean = clean_array(data['aggregated_generalizability_scores']["train_scores_mean"])
        train_scores_std = clean_array(data['aggregated_generalizability_scores']["train_scores_std"])
        test_scores_mean = clean_array(data['aggregated_generalizability_scores']["test_scores_mean"])
        test_scores_std = clean_array(data['aggregated_generalizability_scores']["test_scores_std"])
        learning_df = pd.DataFrame({
                "train_sizes_fraction": train_size,
                "train_scores_mean": train_scores_mean,
                "train_scores_std": train_scores_std,
                "test_scores_mean": test_scores_mean,
                "test_scores_std": test_scores_std
            })
    return features, model, learning_df


def _save_path(features,
                model,
                root_dir,
                ):
    
    saving_folder = root_dir/'learning_curves'/model
    os.makedirs(saving_folder, exist_ok=True)
    
    fname = f"{features}"
    saving_path = saving_folder/ f"{fname}.png"
    return saving_path

def _create_learning_curve(
    root_dir: Path,
    features:str,
    model:str,
    df:pd.DataFrame,
    # x_labels: list[str],
    # y_labels: list[str],
    figsize: tuple[int, int],
    fig_title: str,
    **kwargs,
) -> None:
    """
    Args:
        root_dir: Root directory containing all results
        score: Score to plot
        var: Variance to plot
        x_labels: Labels for x-axis
        y_labels: Labels for y-axis
        figsize: Figure size
        fig_title: Figure title
        x_title: X-axis title
        y_title: Y-axis title
        fname: Filename to save figure
    """

    
    plt.figure(figsize=figsize)

    # Plot training scores
    ax = sns.lineplot(x="train_sizes_fraction", y="train_scores_mean", data=df, label="Training Score", marker='o', color="blue", linewidth=3)
    sns.lineplot(x="train_sizes_fraction", y="test_scores_mean", data=df, label="Test Score", marker='o', color="orange", linewidth=3)


    plt.fill_between(df["train_sizes_fraction"],
                 df["train_scores_mean"] - df["train_scores_std"],
                 df["train_scores_mean"] + df["train_scores_std"],
                 color="blue", alpha=0.2)

    plt.fill_between(df["train_sizes_fraction"],
                 df["test_scores_mean"] - df["test_scores_std"],
                 df["test_scores_mean"] + df["test_scores_std"],
                 color="orange", alpha=0.2)



    plt.title(fig_title, fontsize=16, fontweight='bold')  # Plot title with size and bold
    ax.set_xlabel("Training set size", fontsize=14, fontweight='bold')  # X-axis label
    ax.set_ylabel("$R^2$", fontsize=14, fontweight='bold')  # Y-axis label

    # Customizing tick labels
    ax.tick_params(axis='both', which='major', labelsize=12)  # Set tick label size
    plt.xticks(fontsize=12)  # Set x-tick labels bold
    plt.yticks(fontsize=12)  # Set y-tick labels bold
    plt.legend(fontsize=16)

    # make folder and save the files
    saving_file_path =_save_path(features=features,
                                       model=model, 
                                       root_dir=root_dir,
                                       )

    plt.tight_layout()
    plt.savefig(saving_file_path, dpi=600)

    # plt.show()
    plt.close()



def save_learning_curve(target_dir: Path,
                    # data_type:str,
                    # regressor_model:Optional[str]=None,
) -> tuple[pd.DataFrame,pd.DataFrame]:
    

    pattern: str = "*generalizability_scores.json"
    for model_file in os.listdir(target_dir):
        if "test" not in model_file:

            score_files = [] 
            model_dir = os.path.join(target_dir, model_file)
            score_files: list[Path] = list(Path(model_dir).rglob(pattern))
            
            for file_path in score_files:
                # for structural and mix of structural-scaler
        
                    feats, model, learning_score_data = get_learning_data(file_path=file_path)  
                    _create_learning_curve(target_dir,
                                        feats,
                                        model,
                                        learning_score_data,
                                        figsize=(12, 8),
                                        fig_title =f"Learning Curve of {model} on {feats}"
                                        )





if __name__ == '__main__':
    for target_folder in target_list:
        save_learning_curve(target_dir=RESULTS/target_folder)