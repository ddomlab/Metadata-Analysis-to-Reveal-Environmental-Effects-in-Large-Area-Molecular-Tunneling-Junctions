from sklearn.linear_model import (LinearRegression,
                                  Lasso,
                                  Ridge,
                                  ElasticNet)
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from ngboost import NGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso
from sklearn.kernel_ridge import KernelRidge


from sklearn.preprocessing import (StandardScaler,
                                   QuantileTransformer,
                                   MinMaxScaler,
                                   RobustScaler)

from skopt.space import Integer, Real, Categorical
from typing import Callable, Optional, Union, Dict









transformers: dict[str, Callable] = {
    None:                None,
    "MinMax":            MinMaxScaler(),
    "Standard":          StandardScaler(),
    "Robust Scaler":      RobustScaler(),
    "Uniform Quantile":  QuantileTransformer(),
}


regressor_factory: dict[str, type]={
    "MLR": LinearRegression(),
    "KNN": KNeighborsRegressor(),
    "RF": RandomForestRegressor(),
    "Lasso": Lasso(),
    "Ridge": Ridge(),
    "ElasticNet": ElasticNet(), 
    "KRR": KernelRidge(),
    "DT": DecisionTreeRegressor(),
}


regressor_search_space = {
    "MLR": {
        "regressor__regressor__fit_intercept": [True]
    },

    "Lasso": {
        "regressor__regressor__alpha": Real(1e-4, 1e3, prior="log-uniform"),
        # "regressor__regressor__fit_intercept": [True],
    },

    "Ridge": {
        "regressor__regressor__alpha": Real(1e-4, 1e3, prior="log-uniform"),
        # "regressor__regressor__fit_intercept": [True],
    },
    "ElasticNet": {
        "regressor__regressor__alpha": Real(1e-4, 1e3, prior="log-uniform"),
        'regressor__regressor__l1_ratio': Real(0, 1) 
    },

    "KRR": {
        "regressor__regressor__alpha": Real(1e-3, 1e3, prior="log-uniform"),
        "regressor__regressor__kernel": ["rbf","poly"],
    },

    "KNN": {
        "regressor__regressor__n_neighbors": Integer(1, 50),
        "regressor__regressor__weights": Categorical(["uniform", "distance"]),
        "regressor__regressor__algorithm": Categorical(["ball_tree", "kd_tree", "brute"]),
        "regressor__regressor__leaf_size": Integer(1, 100),
    },

    "RF": {
        "regressor__regressor__n_estimators": Integer(10, 2000, prior="log-uniform"),
        "regressor__regressor__max_depth": [None],
        "regressor__regressor__min_samples_split": Real(0.05, 0.99),
        "regressor__regressor__min_samples_leaf": Real(0.05, 0.99),
        "regressor__regressor__max_features": Categorical(["sqrt", "log2"]),
    },

    "DT": {
        "regressor__regressor__min_samples_split": Real(0.05, 0.99),
        "regressor__regressor__min_samples_leaf": Real(0.05, 0.99),
        "regressor__regressor__max_features": Categorical([None,"sqrt", "log2"]),
        "regressor__regressor__max_depth": [None],
        "regressor__regressor__ccp_alpha": Real(0.05, 0.99),
    },


}