from itertools import product
from typing import Callable, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    # mean_squared_error,
    root_mean_squared_error,
    # r2_score,
)
from sklearn.metrics._scorer import r2_scorer
from sklearn.model_selection import cross_val_predict, cross_validate
from sklearn.model_selection import learning_curve


rmse_scorer = make_scorer(root_mean_squared_error, greater_is_better=False)
mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)



def cross_validate_regressor(
    regressor, X, y, cv
    ) -> tuple[dict[str, float], np.ndarray]:

        scores: dict[str, float] = cross_validate(
            regressor,
            X,
            y,
            cv=cv,
            scoring={
                #r pearson is added
                "r2": r2_scorer,
                "rmse": rmse_scorer,
                "mae": mae_scorer,
            },
            return_estimator=True,
            n_jobs=-1,
        )

        predictions: np.ndarray = cross_val_predict(
            regressor,
            X,
            y,
            cv=cv,
            n_jobs=-1,
        )
        return scores, predictions



def get_incremental_split(
        regressor_params, X, y, cv, steps:int,
        random_state:int
    ) -> tuple:
     
    training_sizes, training_scores, testing_scores = learning_curve(
                                                        regressor_params,
                                                        X,
                                                        y,
                                                        cv=cv,
                                                        n_jobs=-1,
                                                        train_sizes=np.linspace(0.1, 1, int(0.9 / steps)),
                                                        scoring="r2",
                                                        shuffle=True,
                                                        random_state=random_state
                                                        )

 
    return training_sizes, training_scores, testing_scores




def get_feature_importance(data:pd.DataFrame,
                            score:dict,
                            seed:int,
                            )->pd.DataFrame:
    # .feature_importances_
    if hasattr(score['estimator'][-1].named_steps['regressor'].regressor_, 'coef_'): 
        importance = pd.DataFrame([
                estimator.named_steps['regressor'].regressor_.coef_ 
                for estimator in score['estimator']
                ],
                columns=data.columns.to_list()
                )
    elif hasattr(score['estimator'][-1].named_steps['regressor'].regressor_, 'feature_importances_'):
        importance = pd.DataFrame([
                estimator.named_steps['regressor'].regressor_.feature_importances_ 
                for estimator in score['estimator']
                ],
                columns=data.columns.to_list()
                )
    
    else:
        raise ValueError(f"Model {score['estimator'][-1].named_steps['regressor'].regressor_} does not support feature importance extraction.")
    importance['seed'] = seed
    return importance


def get_generalizability_score(X:pd.DataFrame,
                               score:dict,
                               train_sizes,
                               train_scores,
                               test_scores,
                            
)->dict:
                score["generalizability_scores"] = {
                "train_sizes": train_sizes,   # 1D array of training sizes used
                "train_sizes_fraction": train_sizes/len(X),
                "train_scores": train_scores,  # 2D array of training scores
                "test_scores": test_scores,  # 2D array of validation (cross-validation) scores
                }
                 

                return score