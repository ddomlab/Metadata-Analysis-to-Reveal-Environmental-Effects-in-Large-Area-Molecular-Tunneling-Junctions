from itertools import product
from typing import Callable, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
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
            # return_estimator=True,
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