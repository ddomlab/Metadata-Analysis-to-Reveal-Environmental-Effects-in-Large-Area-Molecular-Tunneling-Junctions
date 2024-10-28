import os
from pathlib import Path
import re
from typing import Dict,List,Union
import pandas as pd
import numpy as np
import json


HERE: Path = Path(__file__).resolve().parent
DATASET: Path = HERE.parent.parent/'datasets'/'extracted_files'
extracted_full_data_path:Path = DATASET/'full_tunneling_J.pkl'
extracted_grouped_data_path:Path = DATASET/'grouped_tunneling_J.pkl'

full_data : pd.DataFrame =pd.read_pickle(extracted_full_data_path)
grouped_data : pd.DataFrame =pd.read_pickle(extracted_grouped_data_path)

print(full_data.columns)
