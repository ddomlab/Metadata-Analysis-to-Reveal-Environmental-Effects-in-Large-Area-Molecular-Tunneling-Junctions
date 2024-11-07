import json
from pathlib import Path
from types import NoneType
from typing import Dict, Optional,Tuple

import numpy as np
import pandas as pd



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
        
        

def save_result(score,
                prediction,
                ):
    pass
