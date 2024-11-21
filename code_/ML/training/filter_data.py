from typing import Callable, Optional, Union, Dict
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline
from all_factories import transformers
import pandas as pd




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



unrolling_feature_factory: dict[str, list[str]] = {
                                                "material":     ['electrode_Encoded', 'carbon number'],
                                                "environmental":  ['location_Encoded','temperature','water content'],
                                                "cyclic_time_related":         ['hr_in_day_sin', 'hr_in_day_cos', 'day_in_week_sin',
                                                                                'day_in_week_cos','day_in_year_sin', 'day_in_year_cos'],
                                                "linear_time_related":     ['time of day', 'day of week', 'day of year'],
                                                "environmental_log2(water content)": ['location_Encoded','temperature','log2(water content)']
                                                 }

def unroll_features(rolled_features:list[str],single_features:bool=False)-> list:
    if single_features:
        return rolled_features
    else:
        unrolled_features =   [feats for features in rolled_features for feats in unrolling_feature_factory[features]]
        return unrolled_features