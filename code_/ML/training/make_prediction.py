import pandas as pd
from pathlib import Path
from typing import Callable, Optional, Union, Dict, Tuple

from training_utils import train_regressor
from save_results import save_result

from argparse import ArgumentParser


HERE: Path = Path(__file__).resolve().parent

DATASETS: Path = HERE.parent.parent.parent / "datasets"
training_data_dir: Path = DATASETS/"training_dataset"
DATA: pd.DataFrame = pd.read_csv(training_data_dir/ "substrate_training_data.csv")






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
    





if __name__ == '__main__':
    
    
    feat_list:list[str] = ['location_Encoded','electrode_Encoded',
                            'carbon number','temperature','water content',
                            'hr_in_day_sin', 'hr_in_day_cos', 'day_in_week_sin',
                            'day_in_week_cos','day_in_year_sin', 'day_in_year_cos']
    
    target:str = "mean log(|J|) @ |0.5| V"

    
    main_train(
        dataset=DATA,
        regressor_type="MLR",
        features=feat_list,
        target=target,
        transform_type="Standard",
        hyperparameter_optimization=True,
        test=True
        )