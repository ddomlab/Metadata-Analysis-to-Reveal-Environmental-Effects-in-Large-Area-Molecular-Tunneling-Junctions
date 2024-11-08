from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from all_factories import transformers
import pandas as pd
from typing import Callable, Union
import numpy as np

def sanitize_dataset(training_feats, traget_faet):
    """
    Sanitize the training features and targets in case the target features contain NaN values.

    Args:
        training_features: Training features.
        targets: Targets.
        dropna: Whether to drop NaN values.
        **kwargs: Keyword arguments to pass to filter_dataset.

    Returns:
        Sanitized training features and targets.
    """
    traget_faet: pd.DataFrame = traget_faet.dropna()
    training_feats: pd.DataFrame =training_feats.loc[traget_faet.index]
    return training_feats, traget_faet



def get_data(raw_dataset:pd.DataFrame,
              feats:list,
              target:str):

    training_features:pd.DataFrame = raw_dataset[feats]
    target:pd.DataFrame = raw_dataset[target]
    training_features, target = sanitize_dataset(training_features,target)
    training_test_shape: dict ={
                                "targets_shape": target.shape,
                                "training_features_shape": training_features.shape
                                }

    return training_features, target, training_test_shape


def get_scale(feats:list,
               scaler_type:str)-> Pipeline:
    transformer = [("structural_scaling", transformers[scaler_type], feats)]
    scaling = [("scaling features",
              ColumnTransformer(transformers=[*transformer], remainder="passthrough", verbose_feature_names_out=False)
              )]
    
    return Pipeline(scaling)



def process_results(
    scores: dict[int, dict[str, float]],
    df_shape:dict,
    ) -> dict[Union[int, str], dict[str, float]]:


        avg_r2 = round(np.mean([seed["test_r2"] for seed in scores.values()]), 2)
        stdev_r2 = round(np.std([seed["test_r2"] for seed in scores.values()]), 2)
        print("Average scores:\t",
              f"r2: {avg_r2}±{stdev_r2}")

        first_key = list(scores.keys())[0]
        score_types: list[str] = [
            key for key in scores[first_key].keys() if key.startswith("test_")
        ]
        avgs: list[float] = [
            np.mean([seed[score] for seed in scores.values()]) for score in score_types
        ]
        stdevs: list[float] = [
            np.std([seed[score] for seed in scores.values()]) for score in score_types
        ]

        print(scores)

        score_types: list[str] = [score.replace("test_", "") for score in score_types]
        for score, avg, stdev in zip(score_types, avgs, stdevs ):
            scores[f"{score}_avg"] = abs(avg) if score in ["rmse", "mae"] else avg
            scores[f"{score}_stdev"] = stdev
        print(df_shape)
        score.update(df_shape)
        return scores





