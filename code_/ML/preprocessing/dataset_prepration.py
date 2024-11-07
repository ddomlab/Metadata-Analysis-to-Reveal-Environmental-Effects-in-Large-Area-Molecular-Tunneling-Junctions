import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.preprocessing import LabelEncoder


HERE: Path = Path(__file__).resolve().parent

print(HERE)
DATASETS: Path = HERE.parent.parent.parent / "datasets"
training_data_dir: Path = DATASETS/"training_dataset"

DATA = pd.read_csv(training_data_dir/'substrate_averaged_dropped_shorts_data.csv')

# print(DATA)
DATA['location_Encoded'] = LabelEncoder().fit_transform(DATA['location'])
DATA['electrode_Encoded'] = LabelEncoder().fit_transform(DATA['electrode'])

DATA.to_csv(training_data_dir/"substrate_training_data.csv", index=False)

