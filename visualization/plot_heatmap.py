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
scores_list: list = {"r2", "mae", "rmse"}
var_titles: dict[str, str] = {"stdev": "Standard Deviation", "stderr": "Standard Error"}


def get_results_from_file(
    file_path: Path,
    score: str,
    var: str,
    # impute: bool = False,
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
        avg, std = np.nan, np.nan
    else:
        # for just scaler features

        model:str = file_path.parent.name
        features:str = file_path.name.split(")")[0].replace("(", "")
        print(model)
        print(features)
        
       
        with open(file_path, "r") as f:
            data = json.load(f)

        avg = data[f"{score}_avg"]
        if var == "stdev":
            std = data[f"{score}_stdev"]
        elif var == "stderr":
            std = data[f"{score}_stderr"]
        else:
            raise ValueError(f"Unknown variance type: {var}")

        if score in ["mae", "rmse"]:
            avg, std = abs(avg), abs(std)
        return features, model, avg, std


def get_result_dataframe(target_dir: Path,
                        score: str,
                        var: str,
) -> tuple[pd.DataFrame,pd.DataFrame]:
    
    avg_scores: pd.DataFrame = pd.DataFrame()
    std_scores: pd.DataFrame = pd.DataFrame()
    annotations: pd.DataFrame = pd.DataFrame()
    pattern: str = "*_scores.json"
    for model_file in os.listdir(target_dir):
        if "test" not in model_file:

            model_path:str = os.path.join(target_dir, model_file)
            score_files: list[Path] = list(Path(model_path).rglob(pattern))

            for file_path in score_files:
                features, model, av , std = get_results_from_file(file_path=file_path, score=score, var=var) 
                if features not in avg_scores.columns:

                    avg_scores.loc[features,model] = av
                    std_scores.loc[features,model] = std
                else:
                    avg_scores.at[features,model] = av
                    std_scores.at[features,model] = std
    

    for x, y in product(avg_scores.columns.to_list(), avg_scores.index.to_list()):
        avg: float = avg_scores.loc[y, x]
        std: float = std_scores.loc[y, x]
        avg_txt: str = generate_annotations(avg)
        std_txt: str = generate_annotations(std)
        annotations.loc[y, x] = f"{avg_txt}\n±{std_txt}"

    avg_scores = avg_scores.astype(float)
    annotations = annotations.astype(str)

    return avg_scores, annotations





def generate_annotations(num: float) -> str:
    """
    Args:
        num: Number to annotate

    Returns:
        String to annotate heatmap
    """
    if isinstance(num, float) and not np.isnan(num):
        num_txt: str = f"{round(num, 2)}"
    else:
        num_txt = "NaN"
    return num_txt



def _create_heatmap(
    root_dir: Path,
    score: str,
    var: str,
    avg_scores:pd.DataFrame,
    annotations:pd.DataFrame,
    # x_labels: list[str],
    # y_labels: list[str],
    # parent_dir_labels: list[str],
    figsize: tuple[int, int],
    fig_title: str,
    x_title: str,
    y_title: str,
    fname: str,
    vmin: float = None,
    vmax: float = None,
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

    # Create heatmap

    fig, ax = plt.subplots(figsize=figsize)
    palette: str = "viridis" if score in ["r", "r2"] else "viridis_r"
    custom_cmap = sns.color_palette(palette, as_cmap=True)
    custom_cmap.set_bad(color="lightgray")
    hmap = sns.heatmap(
        avg_scores,
        annot=annotations,
        fmt="",
        cmap=custom_cmap,
        cbar=True,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        mask=avg_scores.isnull(),
        annot_kws={"fontsize": 18},
    )
    
    # Set axis labels and tick labels
    ax.set_xticks(np.arange(len(avg_scores.columns)) + 0.5)
    ax.set_yticks(np.arange(len(avg_scores.index)) + 0.5)
    x_tick_labels: list[str] = [col for col in avg_scores.columns]
    # try:  # handles models as y-axis
    # except:  # handles targets as y-axis
    #     y_tick_labels: list[str] = [target_abbrev_to_full[x] for x in avg_scores.index]
    y_tick_labels: list[str] = avg_scores.index.to_list()

    ax.set_xticklabels(x_tick_labels, rotation=45, ha="right", fontsize=14, fontweight='bold')
    ax.set_yticklabels(y_tick_labels, rotation=0, ha="right", fontsize=14, fontweight='bold')

    # Set plot and axis titles
    plt.title(fig_title,fontsize=18, fontweight='bold')
    ax.set_xlabel(x_title, fontsize=16, fontweight='bold')
    ax.set_ylabel(y_title, fontsize=16, fontweight='bold')
    # Set colorbar title
    score_txt: str = "$R^2$" if score == "r2" else score
    cbar = hmap.collections[0].colorbar
    cbar.set_label(
        f"Average {score_txt.upper()} ± {var_titles[var]}", rotation=270, labelpad=20, 
        fontsize=16, fontweight='bold'
    )
    cbar.ax.tick_params(labelsize=14)
    visualization_folder_path =  root_dir/"heatmap"
    os.makedirs(visualization_folder_path, exist_ok=True)    
    plt.tight_layout()
    plt.savefig(visualization_folder_path / f"{fname}.png", dpi=600)

    # Show the heatmap
    # plt.show()
    plt.close()




def draw_heatmap_results(target_dir:Path,
                        target:str,
                        score:str,
                        var:str,
                        ) -> None:
        ave, anot = get_result_dataframe(target_dir=target_dir,score=score, var=var)
        
        score_txt: str = "$R^2$" if score == "r2" else score.upper()
        _create_heatmap(root_dir=target_dir,
                        score=score,
                        var=var,
                        avg_scores=ave,
                        annotations=anot,
                        figsize=(18, 5),
                        fig_title=f"Average {score_txt} Scores of ML algorithm for Predicting {target}",
                        x_title="Regression Models",
                        y_title="Feature Space",
                        fname=f"Features vs model search heatmap {score} score")
        








if __name__ == '__main__':


    # print(RESULTS)
    for target_folder in target_list:
        for i in scores_list:
            draw_heatmap_results(target_dir=RESULTS/target_folder,
                                target=target_folder,
                                score=i,
                                var='stdev')
            

