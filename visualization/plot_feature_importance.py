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



def get_importance_score(file_path: Path,
                         )-> tuple[str,str,float,float]:

    if not file_path.exists():
        features, model = None, None
        avg, std = np.nan, np.nan
    else:
        # for just scaler features

        model:str = file_path.parent.name
        features:str = file_path.name.split(")")[0].replace("(", "")
        print(model)
        print(features)

        data: pd.DataFrame = pd.read_csv(file_path)

        df_avg = data.mean().to_frame().T
        df_std = data.std().to_frame().T
        df_avg["regressor model"] = model
        df_std["regressor model"] = model
    return features, model, df_avg, df_std


def get_importance_space(target_dir):
    
    annotations: pd.DataFrame = pd.DataFrame()
    pattern: str = "*_scores.json"
    for model_file in os.listdir(target_dir):
        if "test" not in model_file:

            model_path:str = os.path.join(target_dir, model_file)
            score_files: list[Path] = list(Path(model_path).rglob(pattern))

            for file_path in score_files:
                features, model, av , std = get_importance_score(file_path=file_path, score=score, var=var) 

    return 


def draw_heatmap_importance(target_dir:Path,
                            target:str,) -> None:
    
    return




def creat_importance_heatmap():

    return