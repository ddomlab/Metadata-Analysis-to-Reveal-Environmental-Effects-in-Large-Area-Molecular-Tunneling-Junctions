import pandas as pd
from typing import Callable, Union
import numpy as np





def process_results(
    scores: dict[int, dict[str, float]],
    df_shape:dict,
    generalizability:bool,
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


        if generalizability:
            process_learning_score(scores)

        score_types: list[str] = [score.replace("test_", "") for score in score_types]
        for score, avg, stdev in zip(score_types, avgs, stdevs ):
            scores[f"{score}_avg"] = abs(avg) if score in ["rmse", "mae"] else avg
            scores[f"{score}_stdev"] = stdev

        scores.update(df_shape)
        return scores

def process_learning_score(score: dict[int, dict[str, np.ndarray]])->None:
     # Initialize arrays for aggregation
    train_scores_mean = None
    train_scores_std = None
    test_scores_mean = None
    test_scores_std = None
    num_seeds = len(score)

    # Loop over seeds and accumulate results
    for _, results in score.items():
        if train_scores_mean is None:
            # Initialize mean and std with the first seed's results
            train_scores_mean = results['generalizability_scores']["train_scores"].mean(axis=1, keepdims=True)
            train_scores_std = results['generalizability_scores']["train_scores"].std(axis=1, keepdims=True)
            test_scores_mean = results['generalizability_scores']["test_scores"].mean(axis=1, keepdims=True)
            test_scores_std = results['generalizability_scores']["test_scores"].std(axis=1, keepdims=True)
        else:
            # Accumulate the means and stds
            train_scores_mean += results['generalizability_scores']["train_scores"].mean(axis=1, keepdims=True)
            train_scores_std += results['generalizability_scores']["train_scores"].std(axis=1, keepdims=True)
            test_scores_mean += results['generalizability_scores']["test_scores"].mean(axis=1, keepdims=True)
            test_scores_std += results['generalizability_scores']["test_scores"].std(axis=1, keepdims=True)

    # Calculate the averages over the number of seeds
    score['aggregated_generalizability_scores']= {
        "train_sizes": results['generalizability_scores']["train_sizes"],
        "train_sizes_fraction": results['generalizability_scores']["train_sizes_fraction"],
        "train_scores_mean": train_scores_mean / num_seeds,
        "train_scores_std": train_scores_std / num_seeds,
        "test_scores_mean": test_scores_mean / num_seeds,
        "test_scores_std": test_scores_std / num_seeds,
    }





