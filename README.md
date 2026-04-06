This repository contains the data, preprocessing workflows, and machine learning pipelines developed for the paper:<br>
"**Metadata Analysis to Reveal Environmental Effects in Large-Area 
Molecular Tunneling Junctions**"


## Overview
We built an automated pipeline for data cleaning, variance decomposition, 
and enrichment with historically reconstructed environmental variables. We used delta-learning framework, pairing the Simmons model as a physics-based baseline 
with metadata-trained linear regressions to to explain residual variability beyond molecular structure.

The repository is set-up to make the results easy to reproduce. If you get stuck or like to learn more, please feel free to open an issue.

## Setup
The environment.yml file specifies the conda virtual environment. :<br>
<pre> conda env create -f environment.yml </pre>

The `.ipynb` file is provided as a notebook script and should be executed in a Jupyter Notebook environment.

##  Repository Structure

```bash
code_/
├── utils.py  # Shared helper functions
├── cleaning_curation/
│   ├── Append_weather_features.ipynb  # Add weather metadata
│   ├── Data_cleaning.ipynb  # Clean raw measurements
│   └── EDA_and_Aggregation.ipynb  # Explore and aggregate data
├── compiling_converting_data/
│   └── read_txt.py  # Parse raw text files
└── Modeling/
    ├── Residuals_modeling.ipynb  # Model residual variation
    └── Simmons_fitting.ipynb  # Fit Simmons baseline model

datasets/
├── extracted_files/
│   ├── error_log.json  # Extraction error records
│   ├── full_tunneling_J.csv  # Full extracted current data
│   └── grouped_tunneling_J.csv  # Grouped extracted dataset
├── processed/
│   ├── cleaned_dataset_filtered_by_carbon_number_dropped_tests_with_weather_(no water content).csv  # Weather-joined clean dataset
│   ├── cleaned_dataset_filtered_by_carbon_number_dropped_tests_with_weather_imputed.csv  # Weather-imputed clean dataset
│   ├── cleaned_dataset_filtered_by_carbon_number_dropped_tests.csv  # Carbon-filtered clean dataset
│   ├── cleaned_scan_data.csv  # Cleaned scan-level records
│   ├── spot_averaged_data.csv  # Spot-averaged measurements
│   ├── substrate_averaged_clean_data.csv  # Clean substrate averages
│   ├── substrate_averaged_data.csv  # Substrate average values
│   ├── substrate_hierarchically_averaged_clean_data_residual_included.csv  # Hierarchical averages with residuals
│   └── substrate_hierarchically_averaged_clean_data.csv  # Hierarchical averages
└── raw/


```
## How to cite 

If you liked, please cite the paper:
