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
        features:list[str],
        target:str,
        transform_type:str,
        hyperparameter_optimization:bool,
        generalizability:bool,
        test:bool,
        feat_importance:bool,
        single_features:Optional[str]=False,

)-> None:
    
    scores, predictions, feature_importance  = train_regressor(
                                            dataset=dataset,
                                            regressor_type=regressor_type,
                                            features=features,
                                            single_features=single_features,
                                            target=target,
                                            transform_type=transform_type,
                                            hyperparameter_optimization=hyperparameter_optimization,
                                            generalizability=generalizability,
                                            Test=test,
                                            feat_importance=feat_importance
                                            )


    save_result(scores,
                predictions=predictions,
                importance_score= feature_importance,
                target_feature=target,
                features=features,
                regressor_type=regressor_type,
                hypop_status=hyperparameter_optimization,
                transform_type=transform_type,
                generalizability=generalizability,
                TEST=test,
                )
    


# all_model = ["MLR", "RF", "DT","Lasso", "Ridge","ElasticNet", "KRR","KNN",]


if __name__ == '__main__':
    
    
    feat_list:list[str] = [
                           "material",
                           "environmental_log2(water content)",
                           "cyclic_time_related"
                           ]
    
    target:str = "mean log(|J|) @ |0.5| V"
    models = ["MLR", "RF", "DT","Lasso", "ElasticNet"]
    for model in models:
    
        main_train(
            dataset=DATA,
            regressor_type=model,
            features=feat_list,
            target=target,
            transform_type="Standard",
            hyperparameter_optimization=True,
            generalizability=True,
            test=False,
            feat_importance=True
            )       