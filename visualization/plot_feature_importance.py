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
from plot_heatmap import generate_annotations


HERE: Path = Path(__file__).resolve().parent
RESULTS: Path = HERE.parent/ "results"

target_list = ["_log(J)at abs(.5) V"]

def get_importance_score(file_path: Path,
                         )-> tuple[str,str,float,float]:

    if not file_path.exists():
        features, model = None, None
        avg, std = np.nan, np.nan
    else:
        # for just scaler features

        model:str = file_path.parent.name
        features:str = file_path.name.split(")")[0].replace("(", "")

        data: pd.DataFrame = pd.read_csv(file_path)
        data.drop(columns=['seed'], inplace=True)
        df_avg = data.mean().to_frame().T
        df_std = data.std().to_frame().T
        df_avg.index = [model]
        df_std.index = [model]
    return features, model, df_avg, df_std


def get_importance_space(target_dir, algorithm_list):
    avg_dataframes = []
    std_dataframes = []
    annotations: pd.DataFrame = pd.DataFrame()
    pattern: str = "*_importance.csv"
    for model_file in os.listdir(target_dir):
        if "test" not in model_file:

            model_path:str = os.path.join(target_dir, model_file)
            score_files: list[Path] = list(Path(model_path).rglob(pattern))

            for file_path in score_files:
                if "material-environmental-linear_time_related" in file_path.name:
                    features, model, avg_df , std_df = get_importance_score(file_path=file_path) 
                    # TODO: add the uniform feature below
                    if model in algorithm_list:
                        avg_dataframes.append(avg_df)
                        std_dataframes.append(std_df)
    

    overall_avg_importance = pd.concat(avg_dataframes, axis=0)
    overall_std_importance = pd.concat(std_dataframes, axis=0)
    for x, y in product(overall_avg_importance.columns.to_list(), overall_avg_importance.index.to_list()):
        avg: float = overall_avg_importance.loc[y, x]
        std: float = overall_std_importance.loc[y, x]
        avg_txt: str = generate_annotations(avg)
        std_txt: str = generate_annotations(std)
        annotations.loc[y, x] = f"{avg_txt}\n±{std_txt}"

    overall_avg_importance = overall_avg_importance.astype(float)
    annotations = annotations.astype(str)

    return overall_avg_importance, annotations
 



def creat_heatmap_importance(
    root_dir: Path,
    avg_scores:pd.DataFrame,
    annotations:pd.DataFrame,
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
    palette: str = "viridis" 
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
    cbar = hmap.collections[0].colorbar
    cbar.set_label(
        f"Feature Importance", rotation=270, labelpad=20, 
        fontsize=16, fontweight='bold'
    )
    cbar.ax.tick_params(labelsize=14)
    visualization_folder_path =  root_dir/"heatmap_feature_importance"
    os.makedirs(visualization_folder_path, exist_ok=True)    
    plt.tight_layout()
    plt.savefig(visualization_folder_path / f"{fname}.png", dpi=600)

    # Show the heatmap
    # plt.show()
    plt.close()






def draw_feature_importance(target_dir:Path,
                            target:str,
                            algorithms:list[str]) -> None:
    

    for format, algorithm_list in algorithms.items():
        ave, anott = get_importance_space(target_dir=target_dir, algorithm_list=algorithm_list)
        creat_heatmap_importance(root_dir=target_dir,
                                avg_scores=ave,
                                annotations=anott,
                                figsize=(18, 10),
                                fig_title=f"Average Importance of Features contributing to predicting {target}",
                                x_title="Features",
                                y_title="Regression Models",
                                fname=f"Feature importance vs {format} Models ")
    


models_category:dict = {
    'tree base': ['DT', 'RF'],
    'linear':['MLR', 'Lasso', 'ElasticNet','Ridge'],
}

if __name__ == '__main__':

        for target_folder in target_list:
                draw_feature_importance(target_dir=RESULTS/target_folder,
                                    target=target_folder,algorithms=models_category)

    



