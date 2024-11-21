
from pathlib import Path
from typing import Callable, Optional, Union, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV

from save_results import remove_unserializable_keys
from all_factories import (regressor_factory,
                           regressor_search_space,
                           transformers,
                           )

from process_scoring import (process_results)

from filter_data import (unroll_features,
                         get_data,
                         get_scale)

from split import (
    cross_validate_regressor,
    get_incremental_split,
    get_feature_importance,
    get_generalizability_score
)



HERE: Path = Path(__file__).resolve().parent



def set_globals(Test: bool=False) -> None:
    global SEEDS, N_FOLDS, BO_ITER
    if not Test:
        SEEDS = [6, 13, 42, 69, 420, 1234567890, 473129]
        N_FOLDS = 5
        BO_ITER = 42
    else:
        SEEDS = [42,13]
        N_FOLDS = 2
        BO_ITER = 1


def train_regressor(
    dataset: pd.DataFrame,
    regressor_type: str,
    features:list,
    target: str,
    transform_type: str,
    generalizability,
    feat_importance,
    hyperparameter_optimization: bool=True,
    single_features: Optional[bool]=False,
    Test:bool=False,
    ) -> None:
        
        set_globals(Test)
        scores, predictions, data_shape,feature_importance = _prepare_data(
                                                    dataset=dataset,
                                                    regressor_type=regressor_type,
                                                    features=features,
                                                    single_features=single_features,
                                                    target=target,
                                                    transform_type=transform_type,
                                                    hyperparameter_optimization=hyperparameter_optimization,
                                                    generalizability=generalizability,
                                                    feat_importance=feat_importance,
                                                    )
    
        scores = process_results(scores, data_shape, generalizability)
  
        return scores, predictions, feature_importance
        



def _prepare_data(
    dataset: pd.DataFrame,
    features:list,
    target: str,
    regressor_type: str,
    transform_type: str,
    hyperparameter_optimization: bool,
    generalizability,
    feat_importance,
    single_features:Optional[bool]=False,
    **kwargs,
    ) -> tuple[dict[int, dict[str, float]], pd.DataFrame]:

    """
    here you should change the names
    """
    unrolled_features:list = unroll_features(features,single_features=single_features)
    X, y, data_shape = get_data(
                                raw_dataset=dataset,
                                feats=unrolled_features,
                                target=target,
                                )

    # Pipline workflow here and preprocessor
    preprocessor: Pipeline = get_scale(feats=unrolled_features,
                                       scaler_type=transform_type)
    
    preprocessor.set_output(transform="pandas")
    
    score,predication,feature_importance = run(X,
                                                y,
                                                preprocessor=preprocessor,
                                                regressor_type=regressor_type,
                                                transform_type=transform_type,
                                                hyperparameter_optimization=hyperparameter_optimization,
                                                generalizability=generalizability,
                                                feat_importance=feat_importance,
                                                )
    return score, predication, data_shape, feature_importance

def run(
    X,
    y,
    preprocessor: Union[ColumnTransformer, Pipeline],
    regressor_type: str,
    transform_type: str,
    generalizability:bool=False,
    hyperparameter_optimization: bool = True,
    feat_importance:bool=False,
    ) -> tuple[dict[int, dict[str, float]], pd.DataFrame]:

    seed_scores: dict[int, dict[str, float]] = {}
    seed_predictions: dict[int, np.ndarray] = {}
    feature_importances:list = []
    for seed in SEEDS:
      cv_outer = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
      y_transform = Pipeline(
                steps=[("y scaler",transformers[transform_type]),
                      ])

      y_transform_regressor = TransformedTargetRegressor(
                    regressor_factory[regressor_type],
                    transformer=y_transform,
                    )

      regressor :Pipeline= Pipeline(steps=[
                    ("preprocessor", preprocessor),
                    ("regressor", y_transform_regressor),
                        ])

      # set_output on dataframe
      regressor.set_output(transform="pandas")
      if hyperparameter_optimization:
            regressor, regressor_params = _optimize_hyperparams(
                X,
                y,
                cv_outer=cv_outer,
                seed=seed,
                regressor_type=regressor_type,
                regressor=regressor,
            )
            scores, predictions = cross_validate_regressor(
                regressor, X, y, cv_outer
            )
            scores["best_params"] = regressor_params
              


      else:
            scores, predictions = cross_validate_regressor(regressor, X, y, cv_outer)
           
                
      if generalizability:
          train_sizes, train_scores, test_scores = get_incremental_split(regressor,
                                                                                X,
                                                                                y,
                                                                                cv_outer,
                                                                                steps=0.2,
                                                                                random_state=seed)
          scores = get_generalizability_score(X,
                                                scores,
                                                train_sizes,
                                                train_scores,
                                                test_scores,
                                                )
          
      if feat_importance:    
            importance = get_feature_importance(X, scores, seed)
            feature_importances.append(importance)

      scores.pop('estimator', None)
      seed_scores[seed] = scores
      seed_predictions[seed] = predictions.flatten()
      # saving generalizability scores

    seed_predictions: pd.DataFrame = pd.DataFrame.from_dict(
                      seed_predictions, orient="columns")
    if feat_importance:
        seed_importance = pd.concat(feature_importances, axis=0, ignore_index=True)
    else:
        seed_importance =None
    return seed_scores, seed_predictions,seed_importance


def _optimize_hyperparams(
    X, y, cv_outer: KFold, seed: int, regressor_type: str, regressor: Pipeline) -> tuple:

    # Splitting for outer cross-validation loop
    estimators: list[BayesSearchCV] = []
    for train_index, _ in cv_outer.split(X, y):

        X_train = split_for_training(X, train_index)
        y_train = split_for_training(y, train_index)
        # print(X_train)
        # Splitting for inner hyperparameter optimization loop
        cv_inner = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        print("\n\n")
        print(
            "OPTIMIZING HYPERPARAMETERS FOR REGRESSOR", regressor_type, "\tSEED:", seed
        )
        # Bayesian hyperparameter optimization
        bayes = BayesSearchCV(
            regressor,
            regressor_search_space[regressor_type],
            n_iter=BO_ITER,
            cv=cv_inner,
            n_jobs=-1,
            random_state=seed,
            refit=True,
            scoring="r2",
            return_train_score=True,
        )
        bayes.fit(X_train, y_train)

        estimators.append(bayes)

    # Extract the best estimator from hyperparameter optimization
    best_idx: int = np.argmax([est.best_score_ for est in estimators])
    best_estimator: Pipeline = estimators[best_idx].best_estimator_
    try:
        regressor_params: dict = best_estimator.named_steps.regressor.get_params()
        regressor_params = remove_unserializable_keys(regressor_params)
    except:
        regressor_params = {"bad params": "couldn't get them"}

    return best_estimator, regressor_params


def split_for_training(
    data: Union[pd.DataFrame, np.ndarray,pd.Series], indices: np.ndarray
) -> Union[pd.DataFrame, np.ndarray, pd.Series]:
    if isinstance(data, pd.DataFrame):
        split_data = data.iloc[indices]
    elif isinstance(data, np.ndarray):
        split_data = data[indices]
    elif isinstance(data, pd.Series):
        split_data = data.iloc[indices]
    else:
        raise ValueError("Data must be either a pandas DataFrame, Series, or a numpy array.")
    return split_data



