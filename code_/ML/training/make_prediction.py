import pandas as pd
from pathlib import Path
from typing import Callable, Optional, Union, Dict, Tuple

from training_utils import train_regressor
from save_results import save_result

from argparse import ArgumentParser


HERE: Path = Path(__file__).resolve().parent

DATASETS: Path = HERE.parent.parent.parent / "datasets"
training_data_dir: Path = DATASETS/"training_dataset"

def main_train(
        dataset:pd.DataFrame,
        regressor_type:str,
        features=list[str],
        target=str,
        transform_type=str,
        hyperparameter_optimization=bool,
        test=bool,
)-> None:

    scores, predictions = train_regressor(
                                        dataset=dataset,
                                        regressor_type=regressor_type,
                                        features=features,
                                        target=target,
                                        transform_type=transform_type,
                                        hyperparameter_optimization=hyperparameter_optimization,
                                        Test=test,
                                        )


    save_result(scores,
                predictions=predictions,
                target_feature=target,
                features=features,
                regressor_type=regressor_type,
                hypop_status=hyperparameter_optimization,
                transform_type=transform_type,
                TEST=test,
                )
    





if __name__ == "main":
    pass