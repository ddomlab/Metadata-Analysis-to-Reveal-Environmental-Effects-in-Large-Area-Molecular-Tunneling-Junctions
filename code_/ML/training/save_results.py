import json
from pathlib import Path
from types import NoneType
from typing import Dict, Optional,Tuple

import numpy as np
import pandas as pd

HERE: Path = Path(__file__).resolve().parent
ROOT: Path = HERE.parent.parent.parent



feature_abbrev: Dict[str, str] = {
    #TODO: edit this
    "Lp (nm)":          "Lp",
    # "Rg1 (nm)":         "Rg",
    # "Voc (V)":            "Voc",
    # "Jsc (mA cm^-2)":            "Jsc",
    # "FF (%)":             "FF",
    # "Concentration (mg/ml)":            "concentration",
    # "Temperature SANS/SLS/DLS/SEC (K)":         "temperature",
    # "Mn (g/mol)":         "Mn", 
    # "Mw (g/mol)":         "Mw", 

}

def remove_unserializable_keys(d: Dict) -> Dict:
    """Remove unserializable keys from a dictionary."""
    # for k, v in d.items():
    #     if not isinstance(v, (str, int, float, bool, NoneType, tuple, list, np.ndarray, np.floating, np.integer)):
    #         d.pop(k)
    #     elif isinstance(v, dict):
    #         d[k] = remove_unserializable_keys(v)
    # return d
    new_d: dict = {k: v for k, v in d.items() if
                   isinstance(v, (str, int, float, bool, NoneType, np.floating, np.integer))}
    return new_d


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, tuple):
            return list(obj)
        else:
            return super(NumpyArrayEncoder, self).default(obj)
        


def _save(scores:dict,
        predictions:pd.DataFrame,
        results_dir:Path,
        features:list[str],
        hypop_status:bool,
        transform_type:str,
        )-> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    feats = "-".join(feature_abbrev.get(key,key) for key in features)
    fname_root = f"({feats})"
    fname_root = f"{transform_type}_{transform_type}"
    fname_root = f"{fname_root}_hypOFF" if hypop_status==False else fname_root
    
    print("Filename saved as:", fname_root)

    if scores:
        scores_file: Path = results_dir / f"{fname_root}_scores.json"
        with open(scores_file, "w") as f:
            json.dump(scores, f, cls=NumpyArrayEncoder, indent=2)
        print(scores_file)

    if predictions is not None and not predictions.empty:
        predictions_file: Path = results_dir / f"{fname_root}_predictions.csv"
        predictions.to_csv(predictions_file, index=False)
        print(predictions_file)

    print('Done Saving scores and prediction!')



def save_result(scores:dict,
                predictions=pd.DataFrame,
                target_feature=str,
                features=list[str],
                regressor_type=str,
                hypop_status=bool,
                transform_type=str,
                TEST=bool,
                ) -> None:
    
    targets_dir = feature_abbrev[target_feature]



    f_root_dir = f"target_{targets_dir}"
    results_dir: Path = ROOT / "results" / f_root_dir
    results_dir: Path = results_dir / "test" if TEST else results_dir
    results_dir: Path = results_dir/ regressor_type

    _save(
        scores,
        predictions,
        results_dir,
        features,
        hypop_status,
        transform_type,
    )