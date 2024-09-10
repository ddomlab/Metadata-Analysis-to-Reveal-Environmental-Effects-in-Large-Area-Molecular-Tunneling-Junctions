import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Union, Dict
from pathlib import Path

HERE: Path = Path(__file__).resolve().parent
DATASETS: Path = HERE.parent.parent/ 'datasets'
RAW: Path = DATASETS/ 'raw'

raw_dir = RAW/ 'Thuo alkanethiols.xlsx'
raw = pd.read_excel(raw_dir)

print(raw.shape)