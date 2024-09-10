# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 00:14:48 2024

@author: Owner
"""

import os
import re
import pandas as pd
import math
import logging
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.ensemble import IsolationForest
import seaborn as sns
import numpy as np
from scipy.stats import skew, kurtosis
import statistics


def ring_structure(file_path):
    if 'NC4' in file_path:
        return 'Butane'
    elif 'NC5' in file_path:
        return 'Pentane'
    elif 'NC3' in file_path:
        return 'Propane'
    elif 'NC2' in file_path:
        return 'Ethane'
    elif 'DEA' in file_path:
        return 'DEA'
    elif 'DPA' in file_path:
        return 'DPA'



def get_carbon_number_from_path(file_path):
    parts = file_path.split(os.sep)
    for part in parts:
        match = re.search(r'(c|C)(\d+)', part)
        if match:
            return int(match.group(2))
    return None
def classify_time_of_day(time):
    if not time:
        return None

    match = re.search(r'(\d{1,2}):\d{2} (AM|PM)', time)
    if not match:
        return None

    hour = int(match.group(1))
    am_pm = match.group(2)

    if am_pm == 'PM' and hour < 12:
        hour += 12
    elif am_pm == 'AM' and hour == 12:
        hour = 0

    if 0 <= hour < 6:
        return 'Dusk'
    elif 6 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 18:
        return 'Afternoon'
    elif 18 <= hour < 24:
        return 'Evening'

def extract_junction(file_content):
    match = re.search(r'Junction diameter =\s+([\d.]+)', file_content)
    if match:
        return match.group(1)
    return None

def classify_season(month):
    spring = ['March', 'April', 'May']
    summer = ['June', 'July', 'August']
    fall = ['September', 'October', 'November']
    winter = ['December', 'January', 'February']
    
    if month in spring:
        return 'Spring'
    elif month in summer:
        return 'Summer'
    elif month in fall:
        return 'Fall'
    elif month in winter:
        return 'Winter'
    else:
        return 'Unknown'

def extract_time(file_content):
    match = re.search(r'\b\d{1,2}:\d{2} [AP]M\b', file_content)
    if match:
        return match.group()
    return None

def extract_electrode(file_path):
    if 'Ag' in file_path:
        return 'Ag'
    elif 'Au' in file_path:
        return 'Au'
    return 'Unknown'

def extract_metadata(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = [file.readline().strip() for _ in range(15)]
            file_content = "\n".join(lines)
            carbon = get_carbon_number_from_path(file_path)
            if carbon is None:
                print(f"File path: {file_path} - Unknown parameter: Carbon Number")
            
            date_match1 = re.search(r'(\w+), (\w+) (\d{1,2}), (\d{4})', file_content)
            date_match2 = re.search(r'(\w+)-(\d{1,2})-(\d{2,4})', file_content)
            if date_match1:
                date = f"{date_match1.group(2)} {date_match1.group(3)}, {date_match1.group(4)}"
                month = date_match1.group(2)
            elif date_match2:
                date = f"{date_match2.group(1)} {date_match2.group(2)}, 20{date_match2.group(3)}"
                month = date_match2.group(1)
            else:
                date = 'Date not found'
                month = 'Unknown'
                print(f"File path: {file_path} - Unknown parameter: Date")
                print(f"File path: {file_path} - Unknown parameter: Month")
            
            season = classify_season(month)
            if season == 'Unknown':
                print(f"File path: {file_path} - Unknown parameter: Season")
            
            time = extract_time(file_content)
            if time is None:
                print(f"File path: {file_path} - Unknown parameter: Time")
            
            junction_diameter = extract_junction(file_content)
            if junction_diameter is None:
                print(f"File path: {file_path} - Unknown parameter: Junction Diameter")
            
            electrode = extract_electrode(file_path)
            if electrode == 'Unknown':
                print(f"File path: {file_path} - Unknown parameter: Electrode")
            
            time_of_day = classify_time_of_day(time)
            if time_of_day is None:
                print(f"File path: {file_path} - Unknown parameter: Time of Day")
            ring=ring_structure(file_path)
            if ring is None:
                print(f"File path: {file_path} - Unknown parameter: ring structure")
            
            
            metadata = {
                'Date': date,
                'Month': month,
                'File Name': os.path.basename(file_path),
                'Season': season,
                'Time': time,
                'Electrode': electrode,
                'Carbon Number on Chain': carbon,
                'Time of Day': time_of_day,
                'Junction Diameter': junction_diameter,
                'Ring Pattern': ring
                
            }
            with open(f'{file_path}_alldata.txt', 'w') as meta_file:
                for key, value in metadata.items():
                    meta_file.write(f'{key}: {value}\n')
            return metadata
    except Exception as e:
        print(f"Error extracting metadata from file {file_path}: {e}")
        return None

# Function to calculate standard deviation of log(abs current density)
def calculate_std_log_abs_current_density(abs_current_density_values):
    log_abs_current_density_values = [math.log(value) for value in abs_current_density_values if value > 0]
    if not log_abs_current_density_values:
        return None
    return statistics.stdev(log_abs_current_density_values)

# Function to calculate mean of log(abs current density)
def calculate_mean_log_abs_current_density(abs_current_density_values):
    log_abs_current_density_values = [math.log(value) for value in abs_current_density_values if value > 0]
    if not log_abs_current_density_values:
        return None
    return sum(log_abs_current_density_values) / len(log_abs_current_density_values)
def calculate_skewness_log_abs_current_density(abs_current_density_values):
    log_abs_current_density_values = [math.log(value) for value in abs_current_density_values if value > 0]
    if not log_abs_current_density_values:
        return None
    return skew(log_abs_current_density_values)
def calculate_kurtosis_log_abs_current_density(abs_current_density_values):
    log_abs_current_density_values = [math.log(value) for value in abs_current_density_values if value > 0]
    if not log_abs_current_density_values:
        return None
    return kurtosis(log_abs_current_density_values)

# Function to extract voltage from file content
def extract_voltage(file_content):
    voltages = []
    for line in file_content.splitlines():
        match = re.search(r'Voltage: ([\d.]+)', line)
        if match:
            voltages.append(float(match.group(1)))
    return voltages

# Function to extract log(abs current density) from file content
def extract_log_abs_current_density(file_content):
    log_abs_current_densities = []
    for line in file_content.splitlines():
        match = re.search(r'Log Abs Current Density: ([\d.]+)', line)
        if match:
            log_abs_current_densities.append(float(match.group(1)))
    return log_abs_current_densities

# Function to pair files and extract metadata and data
def pair_files(path, folders):
    metadata_files = {}
    data_files = {}
    pairs = 0

    for folder_name in folders:
        folder_path = os.path.join(path, folder_name)
        if not os.path.exists(folder_path):
            print(f"Warning: The path does not exist: {folder_path}")
            continue

        files = os.listdir(folder_path)

        for file_name in files:
            file_path = os.path.join(folder_path, file_name)

            if '(' in file_name or ')' in file_name:
                continue

            if 'data' not in file_name.lower():
                metadata = extract_metadata(file_path)
                if metadata:
                    metadata_files[file_name.replace('.txt', '')] = metadata
                    print(f"Metadata processed for file: {file_name}")
            elif 'data' in file_name.lower():
                base_name = file_name.replace('_data', '').replace('.txt', '')
                try:
                    df = pd.read_csv(file_path, delimiter='\t')
                    if df.empty:
                        print(f"No data found in file: {file_path}")
                        continue

                    abs_current_density_col = [col for col in df.columns if 'absj' in col.lower()]
                    voltage_col = [col for col in df.columns if 'voltage' in col.lower()]

                    if not abs_current_density_col or not voltage_col:
                        print(f"No 'absj' or 'voltage' column found in file: {file_path}")
                        continue

                    abs_current_density_col = abs_current_density_col[0]
                    voltage_col = voltage_col[0]

                    std_log_abs_current_density = calculate_std_log_abs_current_density(df[abs_current_density_col])
                    if std_log_abs_current_density is None:
                        continue
                    mean_log_abs_current_density = calculate_mean_log_abs_current_density(df[abs_current_density_col])
                    if mean_log_abs_current_density is None:
                        continue
                    kurt_log_abs_current_density = calculate_kurtosis_log_abs_current_density(df[abs_current_density_col])
                    if kurt_log_abs_current_density is None:
                        continue
                    skew_log_abs_current_density = calculate_skewness_log_abs_current_density(df[abs_current_density_col])
                    if skew_log_abs_current_density is None:
                        continue
                    log_abs_density = np.log10(df[abs_current_density_col])
                    data = {
    'Abs Current Density': df[abs_current_density_col].tolist(),
    'Mean Log Abs Current Density': mean_log_abs_current_density,
    'Std Log Abs Current Density': std_log_abs_current_density,
    'Kurtosis Log Abs Current Density': kurt_log_abs_current_density,
    'Skewness Log Abs Current Density': skew_log_abs_current_density,
    'Voltage': df[voltage_col].tolist(),
    'Log Abs Current Density': log_abs_density.tolist()
}
                    # Save data to a text file
                    data_file_path = f'{file_path}_data.txt'
                    with open(data_file_path, 'w') as data_file:
                        for key, value in data.items():
                            data_file.write(f'{key}: {value}\n')

                    if base_name in metadata_files:
                        data_files[base_name] = data
                        pairs += 1
                        print(f"Data processed for file: {file_name}")
                        print(f"Pair found for base name: {base_name}")

                        # Create or update the metadata file
                        metadata_file_path = os.path.join(folder_path, f'{base_name}_alldata.txt')
                        with open(metadata_file_path, 'w') as meta_file:
                            for key, value in metadata_files[base_name].items():
                                meta_file.write(f'{key}: {value}\n')

                            for key, value in metadata.items():
                                meta_file.write(f'{key}: {value}\n')
                            meta_file.write(f'Log Abs Std Density: {std_log_abs_current_density}\n')
                            meta_file.write(f'Mean Log Abs Current Density: {mean_log_abs_current_density}\n')
                            meta_file.write(f'Skewness Log Abs Current Density: {skew_log_abs_current_density}\n')
                            meta_file.write(f'Kurtosis Log Abs Current Density: {kurt_log_abs_current_density}\n')
                            meta_file.write(f'Voltage: {data["Voltage"]}\n')
                            meta_file.write(f'Abs Current Density: {data["Abs Current Density"]}\n')
                            meta_file.write(f'Log Abs Current Density: {data["Log Abs Current Density"]}\n')
                            # Iterate over the files in the directory
                            
                except Exception as e:
                    print(f"Error processing data file {file_path}: {e}")
    return metadata_files, data_files, pairs


# Define the paths and folders for AgTS and AuTi datasets
paths = {
    'Dataset 1': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\From Shawn Old\Charge Tunneling Data\Charge Tunneling Data\C10CNC4',
    'Dataset 2': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\From Shawn Old\Charge Tunneling Data\Charge Tunneling Data\C10CNC5',
    'Dataset 3': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\From Shawn Old\Charge Tunneling Data\Charge Tunneling Data\C11CNC4',
    'Dataset 4': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\From Shawn Old\Charge Tunneling Data\Charge Tunneling Data\C11CNC5',
    'Dataset 5': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\From Shawn Old\Charge Tunneling Data\Charge Tunneling Data\C11CNC3',
    'Dataset 6': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\From Shawn Old\Charge Tunneling Data\Charge Tunneling Data\C11CNC2',
    'Dataset 7': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\From Shawn Old\Charge Tunneling Data\Charge Tunneling Data\C10CNC3',
    'Dataset 8': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\From Shawn Old\Charge Tunneling Data\Charge Tunneling Data\C10CNC2',
    'Dataset 9': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\From Shawn Old\Charge Tunneling Data\Charge Tunneling Data\C11DEA',
    'Dataset 10': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\From Shawn Old\Charge Tunneling Data\Charge Tunneling Data\C12DEA',
    'Dataset 11': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\From Shawn Old\Charge Tunneling Data\Charge Tunneling Data\C11DPA',
    'Dataset 12':  r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\From Shawn Old\Charge Tunneling Data\Charge Tunneling Data\C12DPA',
    'Dataset 13': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\From Shawn Old\Charge Tunneling Data\Charge Tunneling Data\C12CNC4', 
    'Dataset 14': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\From Shawn Old\Charge Tunneling Data\Charge Tunneling Data\C12CNC5'
}

folders_Data1 = [r'02132018_1', r'02132018_2', r'12142017']
folders_Data2 = [r'12022017_1', r'12022017_2' ]
folders_Data3=[r'03242018_1', r'C11CNC4\03042018', r'03242018_2', r'04152018_1', r'04152018_2', r'04172018_1', r'04172018_2']
folders_Data4=[r'peak2\02152018', r'peak2\03272018', r'peak2\04082018_1', r'peak2\04082018_2', r'peak2\04182018_1', r'peak2\04202018', r'Peak1\02142018', r'C11CNC5\Peak1\03242018', r'Peak1\04182018_2', r'Peak1\12142017', r'20090825(C11NPPd 3h)', r'20090908(C11NPPd 3h)']
folders_Data5=[r'CD1-81a-12252018', r'CD1-81b-12252018']
folders_Data6=[r'CD1-83a-12262018', r'CD1-83a-12262018\CD1-83b-12262018']
folders_Data7=[r'CD1-80a-12212018', r'CD1-80b-12212018']
folders_Data8=[r'CD1-84c-12272018', r'CD1-85a-01042019']
folders_Data9=[r'20090910 (C11DEA 3h RT)']
folders_Data10=[r'C12DEA']
folders_Data11=[r'20090909 (C11DPA 3h RT)']
folders_Data12=[r'20090813 C12DPA (3h RT)']
folders_Data13=[r'C12prr\1', r'C12prr\2' ]
folders_Data14=[r'C12ppd\1', r'C12ppd\2']

metadata_files_1, data_files_1, num_pairs_1 = pair_files(paths['Dataset 1'], folders_Data1)
metadata_files_2, data_files_2, num_pairs_2 = pair_files(paths['Dataset 2'], folders_Data2)
metadata_files_3, data_files_3, num_pairs_3 = pair_files(paths['Dataset 3'], folders_Data3)
metadata_files_4, data_files_4, num_pairs_4= pair_files(paths['Dataset 4'], folders_Data4)
metadata_files_5, data_files_5, num_pairs_5= pair_files(paths['Dataset 5'], folders_Data5)
metadata_files_6, data_files_6, num_pairs_6= pair_files(paths['Dataset 6'], folders_Data6)
metadata_files_7, data_files_7, num_pairs_7= pair_files(paths['Dataset 7'], folders_Data7)
metadata_files_8, data_files_8, num_pairs_8= pair_files(paths['Dataset 8'], folders_Data8)
metadata_files_9, data_files_9, num_pairs_9= pair_files(paths['Dataset 9'], folders_Data9)
metadata_files_10, data_files_10, num_pairs_10= pair_files(paths['Dataset 10'], folders_Data10)
metadata_files_11, data_files_11, num_pairs_11= pair_files(paths['Dataset 11'], folders_Data11)
metadata_files_12, data_files_12, num_pairs_12= pair_files(paths['Dataset 12'], folders_Data12)
metadata_files_13, data_files_13, num_pairs_13= pair_files(paths['Dataset 13'], folders_Data13)
metadata_files_14, data_files_14, num_pairs_14= pair_files(paths['Dataset 14'], folders_Data14)

print(f"\nNumber of pairs in Data1 dataset: {num_pairs_1}")
print(f"Number of pairs in Data2 dataset: {num_pairs_2}")
print(f"Number of pairs in Data3 dataset: {num_pairs_3}")
print(f"Number of pairs in Data4 dataset: {num_pairs_4}")
print(f"\nNumber of pairs in Data5 dataset: {num_pairs_5}")
print(f"Number of pairs in Data6 dataset: {num_pairs_6}")
print(f"Number of pairs in Data7 dataset: {num_pairs_7}")
print(f"Number of pairs in Data8 dataset: {num_pairs_8}")
print(f"Number of pairs in Data9 dataset: {num_pairs_9}")
print(f"\nNumber of pairs in Data10 dataset: {num_pairs_10}")
print(f"Number of pairs in Data11 dataset: {num_pairs_11}")
print(f"Number of pairs in Data12 dataset: {num_pairs_12}")
print(f"Number of pairs in Data13 dataset: {num_pairs_13}")
print(f"Number of pairs in Data14 dataset: {num_pairs_14}")

# Define the paths and folders for AgTS and AuTi datasets
paths = {
    'Dataset 1': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\From Shawn Old\Charge Tunneling Data\Charge Tunneling Data\C10CNC4',
    'Dataset 2': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\From Shawn Old\Charge Tunneling Data\Charge Tunneling Data\C10CNC5',
    'Dataset 3': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\From Shawn Old\Charge Tunneling Data\Charge Tunneling Data\C11CNC4',
    'Dataset 4': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\From Shawn Old\Charge Tunneling Data\Charge Tunneling Data\C11CNC5',
    'Dataset 5': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\From Shawn Old\Charge Tunneling Data\Charge Tunneling Data\C11CNC3',
    'Dataset 6': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\From Shawn Old\Charge Tunneling Data\Charge Tunneling Data\C11CNC2',
    'Dataset 7': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\From Shawn Old\Charge Tunneling Data\Charge Tunneling Data\C10CNC3',
    'Dataset 8': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\From Shawn Old\Charge Tunneling Data\Charge Tunneling Data\C10CNC2',
    'Dataset 9': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\From Shawn Old\Charge Tunneling Data\Charge Tunneling Data\C11DEA',
    'Dataset 10': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\From Shawn Old\Charge Tunneling Data\Charge Tunneling Data\C12DEA',
    'Dataset 11': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\From Shawn Old\Charge Tunneling Data\Charge Tunneling Data\C11DPA',
    'Dataset 12':  r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\From Shawn Old\Charge Tunneling Data\Charge Tunneling Data\C12DPA',
    'Dataset 13': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\From Shawn Old\Charge Tunneling Data\Charge Tunneling Data\C12CNC4', 
    'Dataset 14': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\From Shawn Old\Charge Tunneling Data\Charge Tunneling Data\C12CNC5'
}

folders_Data1 = [r'02132018_1', r'02132018_2', r'12142017']
folders_Data2 = [r'12022017_1', r'12022017_2' ]
folders_Data3=[r'03242018_1', r'C11CNC4\03042018', r'03242018_2', r'04152018_1', r'04152018_2', r'04172018_1', r'04172018_2']
folders_Data4=[r'peak2\02152018', r'peak2\03272018', r'peak2\04082018_1', r'peak2\04082018_2', r'peak2\04182018_1', r'peak2\04202018', r'Peak1\02142018', r'C11CNC5\Peak1\03242018', r'Peak1\04182018_2', r'Peak1\12142017', r'20090825(C11NPPd 3h)', r'20090908(C11NPPd 3h)']
folders_Data5=[r'CD1-81a-12252018', r'CD1-81b-12252018']
folders_Data6=[r'CD1-83a-12262018', r'CD1-83a-12262018\CD1-83b-12262018']
folders_Data7=[r'CD1-80a-12212018', r'CD1-80b-12212018']
folders_Data8=[r'CD1-84c-12272018', r'CD1-85a-01042019']
folders_Data9=[r'20090910 (C11DEA 3h RT)']
folders_Data10=[r'C12DEA']
folders_Data11=[r'20090909 (C11DPA 3h RT)']
folders_Data12=[r'20090813 C12DPA (3h RT)']
folders_Data13=[r'C12prr\1', r'C12prr\2' ]
folders_Data14=[r'C12ppd\1', r'C12ppd\2']


# Initialize data_dict with combined categories
data_dict = {
    'Pentane': {f'C{carbon}': {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'short_ratio': 0, 'average_log_abs_current_density': 0} for carbon in range(10, 13)},
    'Butane': {f'C{carbon}': {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'short_ratio': 0, 'average_log_abs_current_density': 0} for carbon in range(10, 13)},
    'Propane_DPA': {f'C{carbon}': {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'short_ratio': 0, 'average_log_abs_current_density': 0} for carbon in range(10, 13)},
    'Ethane_DEA': {f'C{carbon}': {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'short_ratio': 0, 'average_log_abs_current_density': 0} for carbon in range(10, 13)}
}

# Function to extract carbon number and ring information from file
def extract_carbon_number_and_ring(file_path):
    ring = None
    carbon = None
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if 'Carbon Number on Chain' in line:
                    carbon = line.split(': ')[1].strip()
                elif 'Ring Pattern' in line:
                    ring = line.split(': ')[1].strip()
    except Exception as e:
        print(f"Error extracting carbon number and ring from file {file_path}: {e}")
    return carbon, ring

# Function to extract log abs current density and voltage from the file
def extract_log_abs_current_density_and_voltage(file_path):
    log_abs_current_density = []
    voltage = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if 'Log Abs Current Density' in line:
                    values = line.split(': ')[1].strip().strip('[]').split(',')
                    log_abs_current_density = [float(value) for value in values]
                elif 'Voltage' in line:
                    values = line.split(': ')[1].strip().strip('[]').split(',')
                    voltage = [float(value) for value in values]
    except Exception as e:
        print(f"Error extracting log abs current density and voltage from file {file_path}: {e}")
    
    return log_abs_current_density, voltage

# Define the paths and folders for AgTS and AuTi datasets
paths = {
    'Dataset 1': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\From Shawn Old\Charge Tunneling Data\Charge Tunneling Data\C10CNC4',
    'Dataset 2': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\From Shawn Old\Charge Tunneling Data\Charge Tunneling Data\C10CNC5',
    'Dataset 3': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\From Shawn Old\Charge Tunneling Data\Charge Tunneling Data\C11CNC4',
    'Dataset 4': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\From Shawn Old\Charge Tunneling Data\Charge Tunneling Data\C11CNC5',
    'Dataset 5': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\From Shawn Old\Charge Tunneling Data\Charge Tunneling Data\C11CNC3',
    'Dataset 6': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\From Shawn Old\Charge Tunneling Data\Charge Tunneling Data\C11CNC2',
    'Dataset 7': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\From Shawn Old\Charge Tunneling Data\Charge Tunneling Data\C10CNC3',
    'Dataset 8': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\From Shawn Old\Charge Tunneling Data\Charge Tunneling Data\C10CNC2',
    'Dataset 9': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\From Shawn Old\Charge Tunneling Data\Charge Tunneling Data\C11DEA',
    'Dataset 10': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\From Shawn Old\Charge Tunneling Data\Charge Tunneling Data\C12DEA',
    'Dataset 11': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\From Shawn Old\Charge Tunneling Data\Charge Tunneling Data\C11DPA',
    'Dataset 12':  r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\From Shawn Old\Charge Tunneling Data\Charge Tunneling Data\C12DPA',
    'Dataset 13': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\From Shawn Old\Charge Tunneling Data\Charge Tunneling Data\C12CNC4', 
    'Dataset 14': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\From Shawn Old\Charge Tunneling Data\Charge Tunneling Data\C12CNC5'
}


# Iterate over the files in the amides datasets
for dataset_name, dataset_path in paths.items():
    for root, dirs, files in os.walk(dataset_path):
        for file_name in files:
            if file_name.endswith("_alldata.txt") and ".txt_metadata" not in file_name:
                file_path = os.path.join(root, file_name)
                carbon, ring = extract_carbon_number_and_ring(file_path)
                
                # Adjust the ring classification
                if ring in ['Propane', 'DPA']:
                    combined_ring = 'Propane_DPA'
                elif ring in ['Ethane', 'DEA']:
                    combined_ring = 'Ethane_DEA'
                else:
                    combined_ring = ring
                
                # Check if both carbon and combined ring are valid before accessing data_dict
                if carbon and combined_ring in data_dict:
                    carbon_key = f'C{carbon}'
                    if carbon_key in data_dict[combined_ring]:
                        log_abs_current_density, voltage = extract_log_abs_current_density_and_voltage(file_path)
                        data_dict[combined_ring][carbon_key]['log_abs_current_density'].extend(log_abs_current_density)
                        data_dict[combined_ring][carbon_key]['voltage'].extend(voltage)
                        data_dict[combined_ring][carbon_key]['file_names'].append(file_name)
                        if any(value > 0 for value in log_abs_current_density):
                            data_dict[combined_ring][carbon_key]['num_shorts'] += 1
                    else:
                        print(f"Skipping file {file_path} with unknown carbon {carbon} or combined ring {combined_ring}")

# Calculate the short ratio and average log abs current density for each amide-carbon combination
for ring, carbon_data in data_dict.items():
    for carbon, values in carbon_data.items():
        unique_file_names = len(set(values['file_names']))
        if unique_file_names > 0:
            values['short_ratio'] = (values['num_shorts'] / unique_file_names) * 100
        else:
            values['short_ratio'] = 0
        if values['log_abs_current_density']:
            values['average_log_abs_current_density'] = sum(values['log_abs_current_density']) / len(values['log_abs_current_density'])
        else:
            values['average_log_abs_current_density'] = 0

# Define output Excel file path
output_path = 'C:\\Users\\Owner\\OneDrive\\Desktop\\sync pending\\log_abs_current_density_voltage_amides_combined.xlsx'

# Create dataframes and write to Excel
with pd.ExcelWriter(output_path) as writer:
    for ring, carbon_data in data_dict.items():
        for carbon, values in carbon_data.items():
            df = pd.DataFrame({
                'Voltage': values['voltage'],
                'Log(Abs Current Density)': values['log_abs_current_density'],
                'Shorts': [values['num_shorts']] * len(values['log_abs_current_density']),
                'Short Ratio': [values['short_ratio']] * len(values['log_abs_current_density']),
                'Average Log(Abs Current Density)': [values['average_log_abs_current_density']] * len(values['log_abs_current_density'])
            })
            df.to_excel(writer, sheet_name=f'{ring}_{carbon}', index=False)

# Function to plot heatmaps
def plot_heatmap(voltage, log_abs_current_density, title, num_shorts, short_ratio, vmin=0, vmax=250, ylim_min=-12, ylim_max=6, **kwargs):
    if not voltage or not log_abs_current_density:
        print(f"No data available for {title}")
        return
    
    log_abs_current_density = np.array(log_abs_current_density)
    print(f"Min and Max log_abs_current_density for {title}: {min(log_abs_current_density)}, {max(log_abs_current_density)}")
    clipped_log_abs_current_density = np.clip(log_abs_current_density, ylim_min, ylim_max)
    yedges = np.linspace(ylim_min, ylim_max, num=51)
    
    heatmap, xedges, yedges = np.histogram2d(voltage, clipped_log_abs_current_density, bins=[50, 50], range=[[min(voltage), max(voltage)], [ylim_min, ylim_max]])
    
    plt.figure(figsize=(12, 8))
    im = plt.imshow(heatmap.T, origin='lower', cmap='viridis', aspect='auto', extent=[xedges[0], xedges[-1], ylim_min, ylim_max], vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im)
    cbar.set_label('Density', fontsize=21, weight='bold')
    cbar.ax.tick_params(labelsize=18)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight('bold')
    
    plt.text(xedges[-1], ylim_min, f"Shorts: {num_shorts}", color='yellow', fontsize=16, ha='right', va='bottom', weight='bold')
    plt.text(xedges[0], ylim_min, f"Short Ratio (%): {short_ratio:.2f}", color='yellow', fontsize=16, ha='left', va='bottom', weight='bold')
    plt.title(title, fontsize=24, weight='bold')
    plt.xlabel('Voltage (V)', fontsize=21, weight='bold')
    plt.ylabel('Log Abs Current Density (A/cm\u00b2)', fontsize=21, weight='bold')
    plt.xticks(fontsize=16, weight='bold')
    plt.yticks(fontsize=16, weight='bold')
    plt.ylim(ylim_min, ylim_max)
    
    if 'image_path' in kwargs:
        plt.savefig(os.path.join(kwargs['image_path'], f'{title}.png'))
        plt.close()
    else:
        plt.show()

# Plotting heatmaps
for ring, carbon_data in data_dict.items():
    for carbon, values in carbon_data.items():
        plot_heatmap(
            values['voltage'], 
            values['log_abs_current_density'], 
            f'{carbon} {ring} Heatmap', 
            num_shorts=values['num_shorts'], 
            short_ratio=values['short_ratio'], 
            image_path='C:\\Users\\Owner\\OneDrive\\Desktop\\Thuo Group Data\\Origin Data\\Heatmap Images\\User Final'
        )
import pandas as pd
import os

# Define the paths for the output Excel files
all_data_excel_path = r'C:\Users\Owner\OneDrive\Desktop\sync pending\log_abs_current_density_amides_seasons.xlsx'
average_per_file_excel_path = r'C:\Users\Owner\OneDrive\Desktop\sync pending\average_log_abs_current_density_per_file_amides_seasons.xlsx'

# Initialize data_dict with required keys and initial values for amides
seasons = ['Winter', 'Spring', 'Summer', 'Fall']
data_dict = {
    'Pentane': {f'C{carbon}': {season: {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'short_ratio': 0, 'average_log_abs_current_density': 0} for season in seasons} for carbon in range(10, 13)},
    'Butane': {f'C{carbon}': {season: {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'short_ratio': 0, 'average_log_abs_current_density': 0} for season in seasons} for carbon in range(10, 13)},
    'Propane': {f'C{carbon}': {season: {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'short_ratio': 0, 'average_log_abs_current_density': 0} for season in seasons} for carbon in range(10, 12)},
    'Ethane': {f'C{carbon}': {season: {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'short_ratio': 0, 'average_log_abs_current_density': 0} for season in seasons} for carbon in range(10, 12)},
    'DEA': {f'C{carbon}': {season: {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'short_ratio': 0, 'average_log_abs_current_density': 0} for season in seasons} for carbon in range(11, 13)},
    'DPA': {f'C{carbon}': {season: {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'short_ratio': 0, 'average_log_abs_current_density': 0} for season in seasons} for carbon in range(11, 13)}
}

# Function to extract carbon number, ring, and season information from file
def extract_carbon_number_ring_and_season(file_path):
    ring = None
    carbon = None
    season = None
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if 'Carbon Number on Chain' in line:
                    carbon = line.split(': ')[1].strip()
                elif 'Ring Pattern' in line:
                    ring = line.split(': ')[1].strip()
                elif 'Season' in line:
                    season = line.split(': ')[1].strip()
    except Exception as e:
        print(f"Error extracting carbon number, ring, and season from file {file_path}: {e}")
    return carbon, ring, season

# Function to extract log abs current density and voltage from the file
def extract_log_abs_current_density_and_voltage(file_path):
    log_abs_current_density = []
    voltage = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if 'Log Abs Current Density' in line:
                    values = line.split(': ')[1].strip().strip('[]').split(',')
                    log_abs_current_density = [float(value) for value in values]
                elif 'Voltage' in line:
                    values = line.split(': ')[1].strip().strip('[]').split(',')
                    voltage = [float(value) for value in values]
    except Exception as e:
        print(f"Error extracting log abs current density and voltage from file {file_path}: {e}")
    
    return log_abs_current_density, voltage

# Initialize lists for storing all data and average data
all_data = []
average_data = []

# Iterate over the files in the amides datasets
for dataset_name, dataset_path in paths.items():
    for root, dirs, files in os.walk(dataset_path):
        for file_name in files:
            if file_name.endswith("_alldata.txt") and ".txt_metadata" not in file_name:
                file_path = os.path.join(root, file_name)
                carbon, ring, season = extract_carbon_number_ring_and_season(file_path)
                
                # Check if both carbon, ring, and season are valid before accessing data_dict
                if carbon and ring in data_dict and season in seasons:
                    carbon_key = f'C{carbon}'
                    if carbon_key in data_dict[ring]:
                        log_abs_current_density, voltage = extract_log_abs_current_density_and_voltage(file_path)
                        data_dict[ring][carbon_key][season]['log_abs_current_density'].extend(log_abs_current_density)
                        data_dict[ring][carbon_key][season]['voltage'].extend(voltage)
                        data_dict[ring][carbon_key][season]['file_names'].append(file_name)
                        if any(value > 0 for value in log_abs_current_density):
                            data_dict[ring][carbon_key][season]['num_shorts'] += 1
                        
                        # Append data for all_data Excel file
                        for ld, v in zip(log_abs_current_density, voltage):
                            all_data.append({
                                'Ring': ring,
                                'Carbon': carbon_key,
                                'Season': season,
                                'File Name': file_name,
                                'Log(Abs Current Density)': ld,
                                'Voltage': v
                            })

# Calculate the short ratio and average log abs current density for each amide-carbon-season combination
for ring, carbon_data in data_dict.items():
    for carbon, season_data in carbon_data.items():
        for season, values in season_data.items():
            unique_file_names = len(set(values['file_names']))
            if unique_file_names > 0:
                values['short_ratio'] = (values['num_shorts'] / unique_file_names) * 100
            else:
                values['short_ratio'] = 0
            if values['log_abs_current_density']:
                values['average_log_abs_current_density'] = sum(values['log_abs_current_density']) / len(values['log_abs_current_density'])
            else:
                values['average_log_abs_current_density'] = 0

            # Append average log density data for average_per_file Excel file
            for file_name in set(values['file_names']):
                avg_log_density = sum([ld for ld, fn in zip(values['log_abs_current_density'], values['file_names']) if fn == file_name]) / values['file_names'].count(file_name)
                average_data.append({
                    'Ring': ring,
                    'Carbon': carbon,
                    'Season': season,
                    'File Name': file_name,
                    'Average Log(Abs Current Density)': avg_log_density
                })

# Create dataframes and write to Excel files
# Write all data to Excel
all_data_df = pd.DataFrame(all_data)
all_data_df.to_excel(all_data_excel_path, index=False, sheet_name='All Data')

# Write average data to Excel
average_data_df = pd.DataFrame(average_data)
average_data_df.to_excel(average_per_file_excel_path, index=False, sheet_name='Average per File')

print("Data has been processed and written to Excel files.")
def plot_heatmap(voltage, log_abs_current_density, title, num_shorts, short_ratio, vmin=0, vmax=250, ylim_min=-12, ylim_max=6, **kwargs):
    if not voltage or not log_abs_current_density:
        print(f"No data available for {title}")
        return
    
    # Convert log_abs_current_density to a NumPy array
    log_abs_current_density = np.array(log_abs_current_density)
    
    # Print min and max values for debugging
    print(f"Min and Max log_abs_current_density for {title}: {min(log_abs_current_density)}, {max(log_abs_current_density)}")
    
    # Clip log_abs_current_density to the ylim range
    clipped_log_abs_current_density = np.clip(log_abs_current_density, ylim_min, ylim_max)
    
    # Recalculate yedges for 50 bins
    yedges = np.linspace(ylim_min, ylim_max, num=51)
    
    # Calculate the heatmap
    heatmap, xedges, yedges = np.histogram2d(voltage, clipped_log_abs_current_density, bins=[50, 50], range=[[min(voltage), max(voltage)], [ylim_min, ylim_max]])
    
    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    im = plt.imshow(heatmap.T, origin='lower', cmap='viridis', aspect='auto', extent=[xedges[0], xedges[-1], ylim_min, ylim_max], vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im)
    
    # Set colorbar label properties
    cbar.set_label('Density', fontsize=21, weight='bold')
    cbar.ax.tick_params(labelsize=18)  # Change the font size of the colorbar scale
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight('bold')  # Make the color scale numbers bold
    
    # Display number of shorts in the bottom right corner
    plt.text(xedges[-1], ylim_min, f"Shorts: {num_shorts}", color='yellow', fontsize=16, ha='right', va='bottom', weight='bold')
    plt.text(xedges[0], ylim_min, f"Short Ratio (%): {short_ratio:.2f}", color='yellow', fontsize=16, ha='left', va='bottom', weight='bold')
    plt.title(title, fontsize=24, weight='bold')  # Increase font size and make it bold
    plt.xlabel('Voltage (V)', fontsize=21, weight='bold')  # Increase font size and make it bold
    plt.ylabel('Log Abs Current Density (A/cm\u00b2)', fontsize=21, weight='bold')  # Increase font size and make it bold
    plt.xticks(fontsize=16, weight='bold')  # Increase font size for ticks
    plt.yticks(fontsize=16, weight='bold')  # Increase font size for ticks
    plt.ylim(ylim_min, ylim_max)  # Set the y-axis limits
    
    # Save the heatmap image if image_path is provided in kwargs
    if 'image_path' in kwargs:
        plt.savefig(os.path.join(kwargs['image_path'], f'{title}.png'))
        plt.close()
    else:
        plt.show()

# Example plotting calls for heatmaps and saving images
for ring, carbon_data in data_dict.items():
    for carbon, season_data in carbon_data.items():
        for season, values in season_data.items():
            plot_heatmap(values['voltage'], values['log_abs_current_density'], f'{carbon} {ring} {season} Heatmap', num_shorts=values['num_shorts'], short_ratio=values['short_ratio'], image_path=r'C:\\Users\\Owner\\OneDrive\\Desktop\\Thuo Group Data\\Origin Data\\Heatmap Images\\Amides Final')

# Ensure all plots are saved to the specified folder path
print("All heatmaps saved successfully.")
# Define time of day categories
time_of_day_categories = ['Morning', 'Evening', 'Afternoon', 'Dusk']


# Define paths for the output Excel files
all_data_excel_path = r'C:\Users\Owner\OneDrive\Desktop\sync pending\log_abs_current_density_amides_time.xlsx'
average_per_file_excel_path = r'C:\Users\Owner\OneDrive\Desktop\sync pending\average_log_abs_current_density_per_file_amides_time.xlsx'

# Define time of day categories
time_of_day_categories = ['Morning', 'Evening', 'Afternoon', 'Dusk']

# Initialize data_dict with required keys and initial values for amides
data_dict = {
    'Pentane': {f'C{carbon}': {time_of_day: {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'short_ratio': 0, 'average_log_abs_current_density': 0} for time_of_day in time_of_day_categories} for carbon in range(10, 13)},
    'Butane': {f'C{carbon}': {time_of_day: {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'short_ratio': 0, 'average_log_abs_current_density': 0} for time_of_day in time_of_day_categories} for carbon in range(10, 13)},
    'Propane': {f'C{carbon}': {time_of_day: {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'short_ratio': 0, 'average_log_abs_current_density': 0} for time_of_day in time_of_day_categories} for carbon in range(10, 12)},
    'Ethane': {f'C{carbon}': {time_of_day: {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'short_ratio': 0, 'average_log_abs_current_density': 0} for time_of_day in time_of_day_categories} for carbon in range(10, 12)},
    'DEA': {f'C{carbon}': {time_of_day: {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'short_ratio': 0, 'average_log_abs_current_density': 0} for time_of_day in time_of_day_categories} for carbon in range(11, 13)},
    'DPA': {f'C{carbon}': {time_of_day: {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'short_ratio': 0, 'average_log_abs_current_density': 0} for time_of_day in time_of_day_categories} for carbon in range(11, 13)}
}

# Function to extract carbon number, ring, and time of day information from file
def extract_carbon_number_ring_and_time_of_day(file_path):
    ring = None
    carbon = None
    time_of_day = None
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if 'Carbon Number on Chain' in line:
                    carbon = line.split(': ')[1].strip()
                elif 'Ring Pattern' in line:
                    ring = line.split(': ')[1].strip()
                elif 'Time of Day' in line:
                    time_of_day = line.split(': ')[1].strip()
    except Exception as e:
        print(f"Error extracting carbon number, ring, and time of day from file {file_path}: {e}")
    return carbon, ring, time_of_day

# Function to extract log abs current density and voltage from the file
def extract_log_abs_current_density_and_voltage(file_path):
    log_abs_current_density = []
    voltage = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if 'Log Abs Current Density' in line:
                    values = line.split(': ')[1].strip().strip('[]').split(',')
                    log_abs_current_density = [float(value) for value in values]
                elif 'Voltage' in line:
                    values = line.split(': ')[1].strip().strip('[]').split(',')
                    voltage = [float(value) for value in values]
    except Exception as e:
        print(f"Error extracting log abs current density and voltage from file {file_path}: {e}")
    
    return log_abs_current_density, voltage

# Initialize lists for storing all data and average data
all_data = []
average_data = []


# Iterate over the files in the amides datasets
for dataset_name, dataset_path in paths.items():
    for root, dirs, files in os.walk(dataset_path):
        for file_name in files:
            if file_name.endswith("_alldata.txt") and ".txt_metadata" not in file_name:
                file_path = os.path.join(root, file_name)
                carbon, ring, time_of_day = extract_carbon_number_ring_and_time_of_day(file_path)
                
                # Check if both carbon, ring, and time of day are valid before accessing data_dict
                if carbon and ring in data_dict and time_of_day in time_of_day_categories:
                    carbon_key = f'C{carbon}'
                    if carbon_key in data_dict[ring]:
                        log_abs_current_density, voltage = extract_log_abs_current_density_and_voltage(file_path)
                        data_dict[ring][carbon_key][time_of_day]['log_abs_current_density'].extend(log_abs_current_density)
                        data_dict[ring][carbon_key][time_of_day]['voltage'].extend(voltage)
                        data_dict[ring][carbon_key][time_of_day]['file_names'].append(file_name)
                        if any(value > 0 for value in log_abs_current_density):
                            data_dict[ring][carbon_key][time_of_day]['num_shorts'] += 1
                        
                        # Append data for all_data Excel file
                        for ld, v in zip(log_abs_current_density, voltage):
                            all_data.append({
                                'Ring': ring,
                                'Carbon': carbon_key,
                                'Time of Day': time_of_day,
                                'File Name': file_name,
                                'Log(Abs Current Density)': ld,
                                'Voltage': v
                            })
                else:
                    print(f"Skipping file {file_path} due to missing or invalid metadata")

# Calculate the short ratio and average log abs current density for each amide-carbon-time of day combination
for ring, carbon_data in data_dict.items():
    for carbon, time_of_day_data in carbon_data.items():
        for time_of_day, values in time_of_day_data.items():
            unique_file_names = len(set(values['file_names']))
            if unique_file_names > 0:
                values['short_ratio'] = (values['num_shorts'] / unique_file_names) * 100
            else:
                values['short_ratio'] = 0
            if values['log_abs_current_density']:
                values['average_log_abs_current_density'] = sum(values['log_abs_current_density']) / len(values['log_abs_current_density'])
            else:
                values['average_log_abs_current_density'] = 0

            # Append average log density data for average_per_file Excel file
            for file_name in set(values['file_names']):
                avg_log_density = sum([ld for ld, fn in zip(values['log_abs_current_density'], values['file_names']) if fn == file_name]) / values['file_names'].count(file_name)
                average_data.append({
                    'Ring': ring,
                    'Carbon': carbon,
                    'Time of Day': time_of_day,
                    'File Name': file_name,
                    'Average Log(Abs Current Density)': avg_log_density
                })

# Create dataframes and write to Excel files
# Write all data to Excel
all_data_df = pd.DataFrame(all_data)
all_data_df.to_excel(all_data_excel_path, index=False, sheet_name='All Data')

# Write average data to Excel
average_data_df = pd.DataFrame(average_data)
average_data_df.to_excel(average_per_file_excel_path, index=False, sheet_name='Average per File')

print("Data has been processed and written to Excel files.")
def plot_heatmap(voltage, log_abs_current_density, title, num_shorts, short_ratio, vmin=0, vmax=250, ylim_min=-12, ylim_max=6, **kwargs):
    if not voltage or not log_abs_current_density:
        print(f"No data available for {title}")
        return
    
    # Convert log_abs_current_density to a NumPy array
    log_abs_current_density = np.array(log_abs_current_density)
    
    # Print min and max values for debugging
    print(f"Min and Max log_abs_current_density for {title}: {min(log_abs_current_density)}, {max(log_abs_current_density)}")
    
    # Clip log_abs_current_density to the ylim range
    clipped_log_abs_current_density = np.clip(log_abs_current_density, ylim_min, ylim_max)
    
    # Recalculate yedges for 50 bins
    yedges = np.linspace(ylim_min, ylim_max, num=51)
    
    # Calculate the heatmap
    heatmap, xedges, yedges = np.histogram2d(voltage, clipped_log_abs_current_density, bins=[50, 50], range=[[min(voltage), max(voltage)], [ylim_min, ylim_max]])
    
    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    im = plt.imshow(heatmap.T, origin='lower', cmap='viridis', aspect='auto', extent=[xedges[0], xedges[-1], ylim_min, ylim_max], vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im)
    
    # Set colorbar label properties
    cbar.set_label('Density', fontsize=21, weight='bold')
    cbar.ax.tick_params(labelsize=18)  # Change the font size of the colorbar scale
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight('bold')  # Make the color scale numbers bold
    
    # Display number of shorts in the bottom right corner
    plt.text(xedges[-1], ylim_min, f"Shorts: {num_shorts}", color='yellow', fontsize=16, ha='right', va='bottom', weight='bold')
    plt.text(xedges[0], ylim_min, f"Short Ratio (%): {short_ratio:.2f}", color='yellow', fontsize=16, ha='left', va='bottom', weight='bold')
    plt.title(title, fontsize=24, weight='bold')  # Increase font size and make it bold
    plt.xlabel('Voltage (V)', fontsize=21, weight='bold')  # Increase font size and make it bold
    plt.ylabel('Log Abs Current Density (A/cm\u00b2)', fontsize=21, weight='bold')  # Increase font size and make it bold
    plt.xticks(fontsize=16, weight='bold')  # Increase font size for ticks
    plt.yticks(fontsize=16, weight='bold')  # Increase font size for ticks
    plt.ylim(ylim_min, ylim_max)  # Set the y-axis limits
    
    # Save the heatmap image if image_path is provided in kwargs
    if 'image_path' in kwargs:
        plt.savefig(os.path.join(kwargs['image_path'], f'{title}.png'))
        plt.close()
    else:
        plt.show()


for ring, carbon_data in data_dict.items():
    for carbon, time_of_day_data in carbon_data.items():
        for time_of_day, values in time_of_day_data.items():
            
            plot_heatmap(values['voltage'], values['log_abs_current_density'], f'{carbon} {ring} {time_of_day} Heatmap', num_shorts=values['num_shorts'], short_ratio=values['short_ratio'], image_path=r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\Heatmap Images\Amides Final')

# Ensure all plots are saved to the specified folder

# Create dictionaries to store log abs current density, voltage values, and file names
data_dict = {
    'Pentane': {f'C{carbon}': {'mean': [], 'skewness': [], 'kurt': [], 'file_names': []} for carbon in range(10, 12)},
    'Butane': {f'C{carbon}': {'mean': [], 'skewness': [], 'kurt': [], 'file_names': []} for carbon in range(10, 12)},
    'Propane': {f'C{carbon}': {'mean': [], 'skewness': [], 'kurt': [], 'file_names': []} for carbon in range(10, 12)},
    'Ethane': {f'C{carbon}': {'mean': [], 'skewness': [], 'kurt': [], 'file_names': []} for carbon in range(10, 12)}
}

# Function to extract carbon number and ring information from file
def extract_carbon_number_and_ring(file_path):
    ring = None
    carbon = None
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if 'Carbon Number on Chain' in line:
                    carbon = line.split(': ')[1].strip()
                elif 'Ring Pattern' in line:
                    ring = line.split(': ')[1].strip()
    except Exception as e:
        print(f"Error extracting carbon number and ring from file {file_path}: {e}")
    return carbon, ring

# Function to extract log mean abs current density from metadata file
def extract_log_mean_abs_current_density(file_path):
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if 'Mean Log Abs Current Density' in line:
                    return float(line.split(': ')[1])
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return None

# Function to extract log skewness abs current density from metadata file
def extract_log_skewness_abs_current_density(file_path):
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if 'Skewness Log Abs Current Density' in line:
                    return float(line.split(': ')[1])
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return None

# Function to extract log kurtosis abs current density from metadata file
def extract_log_kurtosis_abs_current_density(file_path):
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if 'Kurtosis Log Abs Current Density' in line:
                    return float(line.split(': ')[1])
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return None

# Iterate over the files in the dataset
for dataset_path in paths.values():
    for root, dirs, files in os.walk(dataset_path):
        for file_name in files:
            if file_name.endswith("_alldata.txt") and ".txt_metadata" not in file_name:
                file_path = os.path.join(root, file_name)
                carbon, ring = extract_carbon_number_and_ring(file_path)
                mean = extract_log_mean_abs_current_density(file_path)
                skewness = extract_log_skewness_abs_current_density(file_path)
                kurt = extract_log_kurtosis_abs_current_density(file_path)
                if carbon and ring:
                    try:
                        carbon = int(carbon)
                    except ValueError:
                        print(f"Skipping file {file_path} with invalid carbon number {carbon}")
                        continue
                    if 10 <= carbon <= 11:
                        if ring in ['Pentane', 'Butane', 'Propane', 'Ethane']:
                            carbon_key = f'C{carbon}'
                            if carbon_key not in data_dict[ring]:
                                data_dict[ring][carbon_key] = {'mean': [], 'skewness': [], 'kurt': [], 'file_names': []}
                            data_dict[ring][carbon_key]['mean'].append(mean)
                            data_dict[ring][carbon_key]['skewness'].append(skewness)
                            data_dict[ring][carbon_key]['kurt'].append(kurt)
                            data_dict[ring][carbon_key]['file_names'].append(file_name)
                        else:
                            print(f"Skipping file {file_path} with unknown ring {ring}")

# Calculate overall mean, skewness, and kurtosis for each carbon number and ring
results = {'Ring': [], 'Carbon': [], 'Mean': [], 'Skewness': [], 'Kurtosis': []}
for ring, carbon_data in data_dict.items():
    for carbon, values in carbon_data.items():
        valid_means = [v for v in values['mean'] if v is not None]
        valid_skewnesses = [v for v in values['skewness'] if v is not None]
        valid_kurts = [v for v in values['kurt'] if v is not None]
        
        if valid_means:
            overall_mean = sum(valid_means) / len(valid_means)
            overall_skewness = skew(valid_means)
            overall_kurtosis = kurtosis(valid_means)
            results['Ring'].append(ring)
            results['Carbon'].append(carbon)
            results['Mean'].append(overall_mean)
            results['Skewness'].append(overall_skewness)
            results['Kurtosis'].append(overall_kurtosis)

# Convert results to dataframe
results_df = pd.DataFrame(results)

# Scatter plot for Mean
plt.figure(figsize=(10, 6))
sns.scatterplot(data=results_df, x='Carbon', y='Mean', hue='Ring', style='Ring', s=100)
plt.title('Mean Log Abs Current Density')
plt.xlabel('Carbon Number')
plt.ylabel('Mean Log Abs Current Density')
plt.legend(title='Ring Type')
plt.grid(True)
plt.show()

# Scatter plot for Skewness
plt.figure(figsize=(10, 6))
sns.scatterplot(data=results_df, x='Carbon', y='Skewness', hue='Ring', style='Ring', s=100)
plt.title('Skewness Log Abs Current Density')
plt.xlabel('Carbon Number')
plt.ylabel('Skewness Log Abs Current Density')
plt.legend(title='Ring Type')
plt.grid(True)
plt.show()

# Scatter plot for Kurtosis
plt.figure(figsize=(10, 6))
sns.scatterplot(data=results_df, x='Carbon', y='Kurtosis', hue='Ring', style='Ring', s=100)
plt.title('Kurtosis Log Abs Current Density')
plt.xlabel('Carbon Number')
plt.ylabel('Kurtosis Log Abs Current Density')
plt.legend(title='Ring Type')
plt.grid(True)
plt.show()

# Create dictionaries to store log abs current density, voltage values, and file names
data_dict = {
    'Pentane': {f'C{carbon}': {'mean': [], 'skewness': [], 'kurt': [], 'file_names': [], 'season': []} for carbon in range(10, 12)},
    'Butane': {f'C{carbon}': {'mean': [], 'skewness': [], 'kurt': [], 'file_names': [], 'season': []} for carbon in range(10, 12)},
    'Propane': {f'C{carbon}': {'mean': [], 'skewness': [], 'kurt': [], 'file_names': [], 'season': []} for carbon in range(10, 12)},
    'Ethane': {f'C{carbon}': {'mean': [], 'skewness': [], 'kurt': [], 'file_names': [], 'season': []} for carbon in range(10, 12)}
}

# Function to extract carbon number, ring, and season information from file
def extract_metadata(file_path):
    ring = None
    carbon = None
    season = None
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if 'Carbon Number on Chain' in line:
                    carbon = line.split(': ')[1].strip()
                elif 'Ring Pattern' in line:
                    ring = line.split(': ')[1].strip()
                elif 'Season' in line:
                    season = line.split(': ')[1].strip()
    except Exception as e:
        print(f"Error extracting metadata from file {file_path}: {e}")
    return carbon, ring, season

# Function to extract log mean abs current density from metadata file
def extract_log_mean_abs_current_density(file_path):
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if 'Mean Log Abs Current Density' in line:
                    return float(line.split(': ')[1])
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return None

# Function to extract log skewness abs current density from metadata file
def extract_log_skewness_abs_current_density(file_path):
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if 'Skewness Log Abs Current Density' in line:
                    return float(line.split(': ')[1])
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return None

# Function to extract log kurtosis abs current density from metadata file
def extract_log_kurtosis_abs_current_density(file_path):
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if 'Kurtosis Log Abs Current Density' in line:
                    return float(line.split(': ')[1])
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return None

# Iterate over the files in the dataset
for dataset_path in paths.values():
    for root, dirs, files in os.walk(dataset_path):
        for file_name in files:
            if file_name.endswith("_alldata.txt") and ".txt_metadata" not in file_name:
                file_path = os.path.join(root, file_name)
                carbon, ring, season = extract_metadata(file_path)
                mean = extract_log_mean_abs_current_density(file_path)
                skewness = extract_log_skewness_abs_current_density(file_path)
                kurt = extract_log_kurtosis_abs_current_density(file_path)
                if carbon and ring and season:
                    try:
                        carbon = int(carbon)
                    except ValueError:
                        print(f"Skipping file {file_path} with invalid carbon number {carbon}")
                        continue
                    if 10 <= carbon <= 11:
                        if ring in ['Pentane', 'Butane', 'Propane', 'Ethane']:
                            carbon_key = f'C{carbon}'
                            if carbon_key not in data_dict[ring]:
                                data_dict[ring][carbon_key] = {'mean': [], 'skewness': [], 'kurt': [], 'file_names': [], 'season': []}
                            data_dict[ring][carbon_key]['mean'].append(mean)
                            data_dict[ring][carbon_key]['skewness'].append(skewness)
                            data_dict[ring][carbon_key]['kurt'].append(kurt)
                            data_dict[ring][carbon_key]['file_names'].append(file_name)
                            data_dict[ring][carbon_key]['season'].append(season)
                        else:
                            print(f"Skipping file {file_path} with unknown ring {ring}")

# Calculate overall mean, skewness, and kurtosis for each carbon number, ring, and season
results = {'Ring': [], 'Carbon': [], 'Season': [], 'Mean': [], 'Skewness': [], 'Kurtosis': []}
for ring, carbon_data in data_dict.items():
    for carbon, values in carbon_data.items():
        for season in set(values['season']):
            season_means = [v for i, v in enumerate(values['mean']) if values['season'][i] == season and v is not None]
            if season_means:
                overall_mean = sum(season_means) / len(season_means)
                overall_skewness = skew(season_means)
                overall_kurtosis = kurtosis(season_means)
                results['Ring'].append(ring)
                results['Carbon'].append(carbon)
                results['Season'].append(season)
                results['Mean'].append(overall_mean)
                results['Skewness'].append(overall_skewness)
                results['Kurtosis'].append(overall_kurtosis)

# Convert results to dataframe
results_df = pd.DataFrame(results)

# Scatter plot for Mean
plt.figure(figsize=(10, 6))
sns.scatterplot(data=results_df, x='Carbon', y='Mean', hue='Season', style='Ring', s=100)
plt.title('Mean Log Abs Current Density')
plt.xlabel('Carbon Number')
plt.ylabel('Mean Log Abs Current Density')
plt.legend(title='Season')
plt.grid(True)
plt.show()

# Scatter plot for Skewness
plt.figure(figsize=(10, 6))
sns.scatterplot(data=results_df, x='Carbon', y='Skewness', hue='Season', style='Ring', s=100)
plt.title('Skewness Log Abs Current Density')
plt.xlabel('Carbon Number')
plt.ylabel('Skewness Log Abs Current Density')
plt.legend(title='Season')
plt.grid(True)
plt.show()

# Scatter plot for Kurtosis
plt.figure(figsize=(10, 6))
sns.scatterplot(data=results_df, x='Carbon', y='Kurtosis', hue='Season', style='Ring', s=100)
plt.title('Kurtosis Log Abs Current Density')
plt.xlabel('Carbon Number')
plt.ylabel('Kurtosis Log Abs Current Density')
plt.legend(title='Season')
plt.grid(True)
plt.show()

# Create dictionaries to store log abs current density, voltage values, and file names
data_dict = {
    'Pentane': {f'C{carbon}': {'mean': [], 'skewness': [], 'kurt': [], 'file_names': [], 'time_of_day': []} for carbon in range(10, 12)},
    'Butane': {f'C{carbon}': {'mean': [], 'skewness': [], 'kurt': [], 'file_names': [], 'time_of_day': []} for carbon in range(10, 12)},
    'Propane': {f'C{carbon}': {'mean': [], 'skewness': [], 'kurt': [], 'file_names': [], 'time_of_day': []} for carbon in range(10, 12)},
    'Ethane': {f'C{carbon}': {'mean': [], 'skewness': [], 'kurt': [], 'file_names': [], 'time_of_day': []} for carbon in range(10, 12)}
}

# Function to extract carbon number, ring, and time of day information from file
def extract_metadata(file_path):
    ring = None
    carbon = None
    time = None
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if 'Carbon Number on Chain' in line:
                    carbon = line.split(': ')[1].strip()
                elif 'Ring Pattern' in line:
                    ring = line.split(': ')[1].strip()
                elif 'Time of Day' in line:
                    time = line.split(': ')[1].strip()
    except Exception as e:
        print(f"Error extracting metadata from file {file_path}: {e}")
    return carbon, ring, time

# Iterate over the files in the dataset
for dataset_path in paths.values():
    for root, dirs, files in os.walk(dataset_path):
        for file_name in files:
            if file_name.endswith("_alldata.txt") and ".txt_metadata" not in file_name:
                file_path = os.path.join(root, file_name)
                carbon, ring, time = extract_metadata(file_path)
                mean = extract_log_mean_abs_current_density(file_path)
                skewness = extract_log_skewness_abs_current_density(file_path)
                kurt = extract_log_kurtosis_abs_current_density(file_path)
                if carbon and ring and time:
                    try:
                        carbon = int(carbon)
                    except ValueError:
                        print(f"Skipping file {file_path} with invalid carbon number {carbon}")
                        continue
                    if 10 <= carbon <= 11:
                        if ring in ['Pentane', 'Butane', 'Propane', 'Ethane']:
                            carbon_key = f'C{carbon}'
                            if carbon_key not in data_dict[ring]:
                                data_dict[ring][carbon_key] = {'mean': [], 'skewness': [], 'kurt': [], 'file_names': [], 'time_of_day': []}
                            data_dict[ring][carbon_key]['mean'].append(mean)
                            data_dict[ring][carbon_key]['skewness'].append(skewness)
                            data_dict[ring][carbon_key]['kurt'].append(kurt)
                            data_dict[ring][carbon_key]['file_names'].append(file_name)
                            data_dict[ring][carbon_key]['time_of_day'].append(time)
                        else:
                            print(f"Skipping file {file_path} with unknown ring {ring}")

# Calculate overall mean, skewness, and kurtosis for each carbon number, ring, and time of day
results = {'Ring': [], 'Carbon': [], 'Time of Day': [], 'Mean': [], 'Skewness': [], 'Kurtosis': []}
for ring, carbon_data in data_dict.items():
    for carbon, values in carbon_data.items():
        for time in set(values['time_of_day']):
            time_means = [v for i, v in enumerate(values['mean']) if values['time_of_day'][i] == time and v is not None]
            if time_means:
                overall_mean = sum(time_means) / len(time_means)
                overall_skewness = skew(time_means)
                overall_kurtosis = kurtosis(time_means)
                results['Ring'].append(ring)
                results['Carbon'].append(carbon)
                results['Time of Day'].append(time)
                results['Mean'].append(overall_mean)
                results['Skewness'].append(overall_skewness)
                results['Kurtosis'].append(overall_kurtosis)

# Convert results to dataframe
results_df = pd.DataFrame(results)

# Scatter plot for Mean
plt.figure(figsize=(10, 6))
sns.scatterplot(data=results_df, x='Carbon', y='Mean', hue='Time of Day', style='Ring', s=100)
plt.title('Mean Log Abs Current Density')
plt.xlabel('Carbon Number')
plt.ylabel('Mean Log Abs Current Density')
plt.legend(title='Time of Day')
plt.grid(True)
plt.show()

# Scatter plot for Skewness
plt.figure(figsize=(10, 6))
sns.scatterplot(data=results_df, x='Carbon', y='Skewness', hue='Time of Day', style='Ring', s=100)
plt.title('Skewness Log Abs Current Density')
plt.xlabel('Carbon Number')
plt.ylabel('Skewness Log Abs Current Density')
plt.legend(title='Time of Day')
plt.grid(True)
plt.show()

# Scatter plot for Kurtosis
plt.figure(figsize=(10, 6))
sns.scatterplot(data=results_df, x='Carbon', y='Kurtosis', hue='Time of Day', style='Ring', s=100)
plt.title('Kurtosis Log Abs Current Density')
plt.xlabel('Carbon Number')
plt.ylabel('Kurtosis Log Abs Current Density')
plt.legend(title='Time of Day')
plt.grid(True)
plt.show()


# Create dictionaries to store log abs current density, voltage values, and file names
data_dict = {
    'Pentane': {f'C{carbon}': {'mean': [], 'skewness': [], 'kurt': [], 'file_names': [], 'time_of_day': []} for carbon in range(10, 12)},
    'Butane': {f'C{carbon}': {'mean': [], 'skewness': [], 'kurt': [], 'file_names': [], 'time_of_day': []} for carbon in range(10, 12)},
    'Propane': {f'C{carbon}': {'mean': [], 'skewness': [], 'kurt': [], 'file_names': [], 'time_of_day': []} for carbon in range(10, 12)},
    'Ethane': {f'C{carbon}': {'mean': [], 'skewness': [], 'kurt': [], 'file_names': [], 'time_of_day': []} for carbon in range(10, 12)}
}

# Function to extract carbon number, ring, and time of day information from file
def extract_metadata(file_path):
    ring = None
    carbon = None
    time = None
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if 'Carbon Number on Chain' in line:
                    carbon = line.split(': ')[1].strip()
                elif 'Ring Pattern' in line:
                    ring = line.split(': ')[1].strip()
                elif 'Time of Day' in line:
                    time = line.split(': ')[1].strip()
    except Exception as e:
        print(f"Error extracting metadata from file {file_path}: {e}")
    return carbon, ring, time

# Iterate over the files in the dataset
for dataset_path in paths.values():
    for root, dirs, files in os.walk(dataset_path):
        for file_name in files:
            if file_name.endswith("_alldata.txt") and ".txt_metadata" not in file_name:
                file_path = os.path.join(root, file_name)
                carbon, ring, time = extract_metadata(file_path)
                mean = extract_log_mean_abs_current_density(file_path)
                skewness = extract_log_skewness_abs_current_density(file_path)
                kurt = extract_log_kurtosis_abs_current_density(file_path)
                if carbon and ring and time:
                    try:
                        carbon = int(carbon)
                    except ValueError:
                        print(f"Skipping file {file_path} with invalid carbon number {carbon}")
                        continue
                    if 10 <= carbon <= 11:
                        if ring in ['Pentane', 'Butane', 'Propane', 'Ethane']:
                            carbon_key = f'C{carbon}'
                            if carbon_key not in data_dict[ring]:
                                data_dict[ring][carbon_key] = {'mean': [], 'skewness': [], 'kurt': [], 'file_names': [], 'time_of_day': []}
                            data_dict[ring][carbon_key]['mean'].append(mean)
                            data_dict[ring][carbon_key]['skewness'].append(skewness)
                            data_dict[ring][carbon_key]['kurt'].append(kurt)
                            data_dict[ring][carbon_key]['file_names'].append(file_name)
                            data_dict[ring][carbon_key]['time_of_day'].append(time)
                        else:
                            print(f"Skipping file {file_path} with unknown ring {ring}")

# Calculate overall mean, skewness, and kurtosis for each carbon number, ring, and time of day
results = {'Ring': [], 'Carbon': [], 'Time of Day': [], 'Mean': [], 'Skewness': [], 'Kurtosis': []}
for ring, carbon_data in data_dict.items():
    for carbon, values in carbon_data.items():
        for time in set(values['time_of_day']):
            time_means = [v for i, v in enumerate(values['mean']) if values['time_of_day'][i] == time and v is not None]
            if time_means:
                overall_mean = sum(time_means) / len(time_means)
                overall_skewness = skew(time_means)
                overall_kurtosis = kurtosis(time_means)
                results['Ring'].append(ring)
                results['Carbon'].append(carbon)
                results['Time of Day'].append(time)
                results['Mean'].append(overall_mean)
                results['Skewness'].append(overall_skewness)
                results['Kurtosis'].append(overall_kurtosis)

# Convert results to dataframe
results_df = pd.DataFrame(results)

# Scatter plot for Mean
plt.figure(figsize=(10, 6))
sns.scatterplot(data=results_df, x='Carbon', y='Mean', hue='Time of Day', style='Ring', s=100)
plt.title('Mean Log Abs Current Density')
plt.xlabel('Carbon Number')
plt.ylabel('Mean Log Abs Current Density')
plt.legend(title='Time of Day')
plt.grid(True)
plt.show()


# Initialize data_dict with required keys and initial values for amides
data_dict = {
    'Pentane': {f'C{carbon}': {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'short_ratio': 0, 'average_log_abs_current_density': 0} for carbon in range(10, 13)},
    'Butane': {f'C{carbon}': {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'short_ratio': 0, 'average_log_abs_current_density': 0} for carbon in range(10, 13)},
    'Propane': {f'C{carbon}': {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'short_ratio': 0, 'average_log_abs_current_density': 0} for carbon in range(10, 12)},
    'Ethane': {f'C{carbon}': {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'short_ratio': 0, 'average_log_abs_current_density': 0} for carbon in range(10, 12)},
    'DEA': {f'C{carbon}': {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'short_ratio': 0, 'average_log_abs_current_density': 0} for carbon in range(11, 13)},
    'DPA': {f'C{carbon}': {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'short_ratio': 0, 'average_log_abs_current_density': 0} for carbon in range(11, 13)}
}

# Function to extract carbon number, ring, and season information from file
def extract_carbon_number_and_ring(file_path):
    ring = None
    carbon = None
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if 'Carbon Number on Chain' in line:
                    carbon = line.split(': ')[1].strip()
                elif 'Ring Pattern' in line:
                    ring = line.split(': ')[1].strip()
    except Exception as e:
        print(f"Error extracting carbon number and ring from file {file_path}: {e}")
    return carbon, ring

# Function to extract log abs current density and voltage from the file
def extract_log_abs_current_density_and_voltage(file_path):
    log_abs_current_density = []
    voltage = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if 'Log Abs Current Density' in line:
                    values = line.split(': ')[1].strip().strip('[]').split(',')
                    log_abs_current_density = [float(value) for value in values]
                elif 'Voltage' in line:
                    values = line.split(': ')[1].strip().strip('[]').split(',')
                    voltage = [float(value) for value in values]
    except Exception as e:
        print(f"Error extracting log abs current density and voltage from file {file_path}: {e}")
    
    return log_abs_current_density, voltage

# Iterate over the files in the amides datasets
for dataset_name, dataset_path in paths.items():
    for root, dirs, files in os.walk(dataset_path):
        for file_name in files:
            if file_name.endswith("_alldata.txt") and ".txt_metadata" not in file_name:
                file_path = os.path.join(root, file_name)
                carbon, ring = extract_carbon_number_and_ring(file_path)
                
                # Check if both carbon and ring are valid before accessing data_dict
                if carbon and ring in data_dict:
                    carbon_key = f'C{carbon}'
                    if carbon_key in data_dict[ring]:
                        log_abs_current_density, voltage = extract_log_abs_current_density_and_voltage(file_path)
                        data_dict[ring][carbon_key]['log_abs_current_density'].extend(log_abs_current_density)
                        data_dict[ring][carbon_key]['voltage'].extend(voltage)
                        data_dict[ring][carbon_key]['file_names'].append(file_name)
                        if any(value > 0 for value in log_abs_current_density):
                            data_dict[ring][carbon_key]['num_shorts'] += 1
                    else:
                        print(f"Skipping file {file_path} with unknown carbon {carbon} or ring {ring}")
                else:
                    print(f"Skipping file {file_path} due to missing or invalid metadata")

# Calculate statistics for each amide-carbon combination
statistics = []

for ring, carbon_data in data_dict.items():
    for carbon, values in carbon_data.items():
        log_abs_current_density = np.array(values['log_abs_current_density'])
        unique_file_names = len(set(values['file_names']))
        if unique_file_names > 0:
            values['short_ratio'] = (values['num_shorts'] / unique_file_names) * 100
        else:
            values['short_ratio'] = 0
        if log_abs_current_density.size > 0:
            log_abs_current_density = np.where(np.isinf(log_abs_current_density), np.nan, log_abs_current_density)
            average_log_abs_current_density = np.nanmean(log_abs_current_density)
            stdev_log_abs_current_density = np.nanstd(log_abs_current_density)
            kurtosis_log_abs_current_density = kurtosis(log_abs_current_density, nan_policy='omit')
            skewness_log_abs_current_density = skew(log_abs_current_density, nan_policy='omit')
        else:
            average_log_abs_current_density = 0
            stdev_log_abs_current_density = 0
            kurtosis_log_abs_current_density = 0
            skewness_log_abs_current_density = 0
        
        statistics.append({
            'Ring': ring,
            'Carbon': carbon,
            'Average Log Abs Current Density': average_log_abs_current_density,
            'Standard Deviation': stdev_log_abs_current_density,
            'Kurtosis': kurtosis_log_abs_current_density,
            'Skewness': skewness_log_abs_current_density,
            'Num Shorts': values['num_shorts'],
            'Short Ratio': values['short_ratio']
        })

# Convert statistics to a DataFrame
df_statistics = pd.DataFrame(statistics)

# Save statistics to an Excel file
output_excel_path = 'C:\\Users\\Owner\\OneDrive\\Desktop\\amide_plots\\amide_statistics.xlsx'
df_statistics.to_excel(output_excel_path, index=False)
# Define marker styles for each ring type
marker_styles = {
    'Pentane': 'o',
    'Butane': 's',
    'Propane': 'D',
    'Ethane': '^',
    'DEA': 'v',
    'DPA': 'p'
}

# Plot the statistics
plot_folder = 'C:\\Users\\Owner\\OneDrive\\Desktop\\amide_plots'
os.makedirs(plot_folder, exist_ok=True)

# Plotting helper function
def plot_statistic(stat_name):
    plt.figure(figsize=(10, 6))
    for ring, group_data in df_statistics.groupby('Ring'):
        plt.scatter(group_data['Carbon'].apply(lambda x: int(x[1:])), group_data[stat_name], label=ring, marker=marker_styles[ring])
    plt.xlabel('Carbon Number')
    plt.ylabel(stat_name)
    plt.title(f'{stat_name} by Carbon Number and Ring Type')
    plt.legend(title='Ring Type')
    plt.grid(True)
    plt.savefig(os.path.join(plot_folder, f'{stat_name}_by_carbon_and_ring.png'))
    plt.close()

# Plot each statistic
plot_statistic('Average Log Abs Current Density')
plot_statistic('Standard Deviation')
plot_statistic('Kurtosis')
plot_statistic('Skewness')
plot_statistic('Num Shorts')
plot_statistic('Short Ratio')

print("All statistics calculated and plots saved successfully.")


# Initialize data_dict with required keys and initial values for amides
data_dict = {
    'Pentane': {f'C{carbon}': {season: {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'short_ratio': 0, 'average_log_abs_current_density': 0} for season in seasons} for carbon in range(10, 13)},
    'Butane': {f'C{carbon}': {season: {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'short_ratio': 0, 'average_log_abs_current_density': 0} for season in seasons} for carbon in range(10, 13)},
    'Propane': {f'C{carbon}': {season: {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'short_ratio': 0, 'average_log_abs_current_density': 0} for season in seasons} for carbon in range(10, 12)},
    'Ethane': {f'C{carbon}': {season: {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'short_ratio': 0, 'average_log_abs_current_density': 0} for season in seasons} for carbon in range(10, 12)},
    'DEA': {f'C{carbon}': {season: {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'short_ratio': 0, 'average_log_abs_current_density': 0} for season in seasons} for carbon in range(11, 13)},
    'DPA': {f'C{carbon}': {season: {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'short_ratio': 0, 'average_log_abs_current_density': 0} for season in seasons} for carbon in range(11, 13)}
}

# Function to extract carbon number, ring, and season information from file
def extract_carbon_number_ring_and_season(file_path):
    ring = None
    carbon = None
    season = None
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if 'Carbon Number on Chain' in line:
                    carbon = line.split(': ')[1].strip()
                elif 'Ring Pattern' in line:
                    ring = line.split(': ')[1].strip()
                elif 'Season' in line:
                    season = line.split(': ')[1].strip()
    except Exception as e:
        print(f"Error extracting carbon number, ring, and season from file {file_path}: {e}")
    return carbon, ring, season

# Function to extract log abs current density and voltage from the file
def extract_log_abs_current_density_and_voltage(file_path):
    log_abs_current_density = []
    voltage = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if 'Log Abs Current Density' in line:
                    values = line.split(': ')[1].strip().strip('[]').split(',')
                    log_abs_current_density = [float(value) for value in values]
                elif 'Voltage' in line:
                    values = line.split(': ')[1].strip().strip('[]').split(',')
                    voltage = [float(value) for value in values]
    except Exception as e:
        print(f"Error extracting log abs current density and voltage from file {file_path}: {e}")
    
    return log_abs_current_density, voltage

# Iterate over the files in the amides datasets
for dataset_name, dataset_path in paths.items():
    for root, dirs, files in os.walk(dataset_path):
        for file_name in files:
            if file_name.endswith("_alldata.txt") and ".txt_metadata" not in file_name:
                file_path = os.path.join(root, file_name)
                carbon, ring, season = extract_carbon_number_ring_and_season(file_path)
                
                # Check if both carbon, ring, and season are valid before accessing data_dict
                if carbon and ring in data_dict and season in data_dict[ring][f'C{carbon}']:
                    carbon_key = f'C{carbon}'
                    if carbon_key in data_dict[ring]:
                        log_abs_current_density, voltage = extract_log_abs_current_density_and_voltage(file_path)
                        data_dict[ring][carbon_key][season]['log_abs_current_density'].extend(log_abs_current_density)
                        data_dict[ring][carbon_key][season]['voltage'].extend(voltage)
                        data_dict[ring][carbon_key][season]['file_names'].append(file_name)
                        if any(value > 0 for value in log_abs_current_density):
                            data_dict[ring][carbon_key][season]['num_shorts'] += 1
                    else:
                        print(f"Skipping file {file_path} with unknown carbon {carbon} or ring {ring}")
                else:
                    print(f"Skipping file {file_path} due to missing or invalid metadata")

# Calculate statistics for each amide-carbon-season combination
statistics = []

for ring, carbon_data in data_dict.items():
    for carbon, season_data in carbon_data.items():
        for season, values in season_data.items():
            log_abs_current_density = np.array(values['log_abs_current_density'])
            unique_file_names = len(set(values['file_names']))
            if unique_file_names > 0:
                values['short_ratio'] = (values['num_shorts'] / unique_file_names) * 100
            else:
                values['short_ratio'] = 0
            if log_abs_current_density.size > 0:
                log_abs_current_density = np.where(np.isinf(log_abs_current_density), np.nan, log_abs_current_density)
                average_log_abs_current_density = np.nanmean(log_abs_current_density)
                stdev_log_abs_current_density = np.nanstd(log_abs_current_density)
                kurtosis_log_abs_current_density = kurtosis(log_abs_current_density, nan_policy='omit')
                skewness_log_abs_current_density = skew(log_abs_current_density, nan_policy='omit')
            else:
                average_log_abs_current_density = 0
                stdev_log_abs_current_density = 0
                kurtosis_log_abs_current_density = 0
                skewness_log_abs_current_density = 0
            
            statistics.append({
                'Ring': ring,
                'Carbon': carbon,
                'Season': season,
                'Average Log Abs Current Density': average_log_abs_current_density,
                'Standard Deviation': stdev_log_abs_current_density,
                'Kurtosis': kurtosis_log_abs_current_density,
                'Skewness': skewness_log_abs_current_density,
                'Num Shorts': values['num_shorts'],
                'Short Ratio': values['short_ratio']
            })

# Convert statistics to a DataFrame
df_statistics = pd.DataFrame(statistics)

# Save statistics to an Excel file
output_excel_path = 'C:\\Users\\Owner\\OneDrive\\Desktop\\amide_plots\\amide_statistics_seasons.xlsx'
df_statistics.to_excel(output_excel_path, index=False)

# Define marker styles for each ring type
marker_styles = {
    'Pentane': 'o',
    'Butane': 's',
    'Propane': 'D',
    'Ethane': '^',
    'DEA': 'v',
    'DPA': 'p'
}

# Plot the statistics
plot_folder = 'C:\\Users\\Owner\\OneDrive\\Desktop\\amide_plots'

# Plotting helper function
def plot_statistic(stat_name):
    plt.figure(figsize=(10, 6))
    for ring, group_data in df_statistics.groupby('Ring'):
        for season, season_data in group_data.groupby('Season'):
            plt.scatter(season_data['Carbon'].apply(lambda x: int(x[1:])), season_data[stat_name],
                        label=f'{ring} - {season}', marker=marker_styles[ring])
    plt.xlabel('Carbon Number')
    plt.ylabel(stat_name)
    plt.title(f'{stat_name} by Carbon Number, Season, and Ring Type')
    plt.legend(title='Ring - Season')
    plt.grid(True)
    plt.savefig(os.path.join(plot_folder, f'{stat_name}_by_carbon_season_and_ring.png'))
    plt.close()

# Plot each statistic
plot_statistic('Average Log Abs Current Density')
plot_statistic('Standard Deviation')
plot_statistic('Kurtosis')
plot_statistic('Skewness')
plot_statistic('Num Shorts')
plot_statistic('Short Ratio')

print("All statistics calculated and plots saved successfully.")

# Initialize data_dict with required keys and initial values for amides
data_dict = {
    'Pentane': {f'C{carbon}': {season: {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'short_ratio': 0, 'average_log_abs_current_density': 0} for season in seasons} for carbon in range(10, 13)},
    'Butane': {f'C{carbon}': {season: {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'short_ratio': 0, 'average_log_abs_current_density': 0} for season in seasons} for carbon in range(10, 13)},
    'Propane': {f'C{carbon}': {season: {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'short_ratio': 0, 'average_log_abs_current_density': 0} for season in seasons} for carbon in range(10, 12)},
    'Ethane': {f'C{carbon}': {season: {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'short_ratio': 0, 'average_log_abs_current_density': 0} for season in seasons} for carbon in range(10, 12)},
    'DEA': {f'C{carbon}': {season: {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'short_ratio': 0, 'average_log_abs_current_density': 0} for season in seasons} for carbon in range(11, 13)},
    'DPA': {f'C{carbon}': {season: {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'short_ratio': 0, 'average_log_abs_current_density': 0} for season in seasons} for carbon in range(11, 13)}
}

# Function to extract carbon number, ring, and season information from file
def extract_carbon_number_ring_and_season(file_path):
    ring = None
    carbon = None
    season = None
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if 'Carbon Number on Chain' in line:
                    carbon = line.split(': ')[1].strip()
                elif 'Ring Pattern' in line:
                    ring = line.split(': ')[1].strip()
                elif 'Season' in line:
                    season = line.split(': ')[1].strip()
    except Exception as e:
        print(f"Error extracting carbon number, ring, and season from file {file_path}: {e}")
    return carbon, ring, season

# Function to extract log abs current density and voltage from the file
def extract_log_abs_current_density_and_voltage(file_path):
    log_abs_current_density = []
    voltage = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if 'Log Abs Current Density' in line:
                    values = line.split(': ')[1].strip().strip('[]').split(',')
                    log_abs_current_density = [float(value) for value in values]
                elif 'Voltage' in line:
                    values = line.split(': ')[1].strip().strip('[]').split(',')
                    voltage = [float(value) for value in values]
    except Exception as e:
        print(f"Error extracting log abs current density and voltage from file {file_path}: {e}")
    
    return log_abs_current_density, voltage

# Iterate over the files in the amides datasets
for dataset_name, dataset_path in paths.items():
    for root, dirs, files in os.walk(dataset_path):
        for file_name in files:
            if file_name.endswith("_alldata.txt") and ".txt_metadata" not in file_name:
                file_path = os.path.join(root, file_name)
                carbon, ring, season = extract_carbon_number_ring_and_season(file_path)
                
                # Check if both carbon, ring, and season are valid before accessing data_dict
                if carbon and ring in data_dict and season in data_dict[ring][f'C{carbon}']:
                    carbon_key = f'C{carbon}'
                    if carbon_key in data_dict[ring]:
                        log_abs_current_density, voltage = extract_log_abs_current_density_and_voltage(file_path)
                        data_dict[ring][carbon_key][season]['log_abs_current_density'].extend(log_abs_current_density)
                        data_dict[ring][carbon_key][season]['voltage'].extend(voltage)
                        data_dict[ring][carbon_key][season]['file_names'].append(file_name)
                        if any(value > 0 for value in log_abs_current_density):
                            data_dict[ring][carbon_key][season]['num_shorts'] += 1
                    else:
                        print(f"Skipping file {file_path} with unknown carbon {carbon} or ring {ring}")
                else:
                    print(f"Skipping file {file_path} due to missing or invalid metadata")

# Calculate statistics for each amide-carbon-season combination
statistics = []

for ring, carbon_data in data_dict.items():
    for carbon, season_data in carbon_data.items():
        for season, values in season_data.items():
            log_abs_current_density = np.array(values['log_abs_current_density'])
            unique_file_names = len(set(values['file_names']))
            if unique_file_names > 0:
                values['short_ratio'] = (values['num_shorts'] / unique_file_names) * 100
            else:
                values['short_ratio'] = 0
            if log_abs_current_density.size > 0:
                log_abs_current_density = np.where(np.isinf(log_abs_current_density), np.nan, log_abs_current_density)
                average_log_abs_current_density = np.nanmean(log_abs_current_density)
                stdev_log_abs_current_density = np.nanstd(log_abs_current_density)
                kurtosis_log_abs_current_density = kurtosis(log_abs_current_density, nan_policy='omit')
                skewness_log_abs_current_density = skew(log_abs_current_density, nan_policy='omit')
            else:
                average_log_abs_current_density = 0
                stdev_log_abs_current_density = 0
                kurtosis_log_abs_current_density = 0
                skewness_log_abs_current_density = 0
            
            statistics.append({
                'Ring': ring,
                'Carbon': carbon,
                'Season': season,
                'Average Log Abs Current Density': average_log_abs_current_density,
                'Standard Deviation': stdev_log_abs_current_density,
                'Kurtosis': kurtosis_log_abs_current_density,
                'Skewness': skewness_log_abs_current_density,
                'Num Shorts': values['num_shorts'],
                'Short Ratio': values['short_ratio']
            })

# Convert statistics to a DataFrame
df_statistics = pd.DataFrame(statistics)

# Save statistics to an Excel file
output_excel_path = 'C:\\Users\\Owner\\OneDrive\\Desktop\\amide_plots\\amide_statistics_seasons.xlsx'
df_statistics.to_excel(output_excel_path, index=False)

# Define marker styles for each ring type
marker_styles = {
    'Pentane': 'o',
    'Butane': 's',
    'Propane': 'D',
    'Ethane': '^',
    'DEA': 'v',
    'DPA': 'p'
}

# Plot the statistics
plot_folder = 'C:\\Users\\Owner\\OneDrive\\Desktop\\amide_plots'

# Plotting helper function
def plot_statistic(stat_name):
    plt.figure(figsize=(10, 6))
    for ring, group_data in df_statistics.groupby('Ring'):
        for season, season_data in group_data.groupby('Season'):
            plt.scatter(season_data['Carbon'].apply(lambda x: int(x[1:])), season_data[stat_name],
                        label=f'{ring} - {season}', marker=marker_styles[ring])
    
    # Create legend with only the points that were plotted
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = list(set(labels))  # Get unique labels
    unique_handles = [handles[labels.index(label)] for label in unique_labels]  # Match handles to unique labels
    plt.legend(unique_handles, unique_labels, loc='upper left', bbox_to_anchor=(1, 0.5), title='Ring - Season')
    
    
    plt.xlabel('Carbon Number')
    plt.ylabel(stat_name)
    plt.title(f'{stat_name} by Carbon Number, Season, and Ring Type')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, f'{stat_name}_by_carbon_season_and_ring.png'))
    plt.close()

# Plot each statistic
plot_statistic('Average Log Abs Current Density')
plot_statistic('Standard Deviation')
plot_statistic('Kurtosis')
plot_statistic('Skewness')
plot_statistic('Num Shorts')
plot_statistic('Short Ratio')
time_of_days=['Morning', 'Afternoon', 'Night', 'Dusk']
# Define your data_dict with the required keys and initial values
data_dict = {
    'Pentane': {f'C{carbon}': {time_of_day: {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'short_ratio': 0, 'average_log_abs_current_density': 0, 'stdev_log_abs_current_density': 0, 'kurtosis_log_abs_current_density': 0, 'skewness_log_abs_current_density': 0} for time_of_day in time_of_days} for carbon in range(10, 13)},
    'Butane': {f'C{carbon}': {time_of_day: {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'short_ratio': 0, 'average_log_abs_current_density': 0, 'stdev_log_abs_current_density': 0, 'kurtosis_log_abs_current_density': 0, 'skewness_log_abs_current_density': 0} for time_of_day in time_of_days} for carbon in range(10, 13)},
    'Propane': {f'C{carbon}': {time_of_day: {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'short_ratio': 0, 'average_log_abs_current_density': 0, 'stdev_log_abs_current_density': 0, 'kurtosis_log_abs_current_density': 0, 'skewness_log_abs_current_density': 0} for time_of_day in time_of_days} for carbon in range(10, 12)},
    'Ethane': {f'C{carbon}': {time_of_day: {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'short_ratio': 0, 'average_log_abs_current_density': 0, 'stdev_log_abs_current_density': 0, 'kurtosis_log_abs_current_density': 0, 'skewness_log_abs_current_density': 0} for time_of_day in time_of_days} for carbon in range(10, 12)},
    'DEA': {f'C{carbon}': {time_of_day: {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'short_ratio': 0, 'average_log_abs_current_density': 0, 'stdev_log_abs_current_density': 0, 'kurtosis_log_abs_current_density': 0, 'skewness_log_abs_current_density': 0} for time_of_day in time_of_days} for carbon in range(11, 13)},
    'DPA': {f'C{carbon}': {time_of_day: {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'short_ratio': 0, 'average_log_abs_current_density': 0, 'stdev_log_abs_current_density': 0, 'kurtosis_log_abs_current_density': 0, 'skewness_log_abs_current_density': 0} for time_of_day in time_of_days} for carbon in range(11, 13)}
}

# Function to extract carbon number, ring, and time of day information from file
def extract_carbon_number_ring_and_time_of_day(file_path):
    ring = None
    carbon = None
    time_of_day = None
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if 'Carbon Number on Chain' in line:
                    carbon = line.split(': ')[1].strip()
                elif 'Ring Pattern' in line:
                    ring = line.split(': ')[1].strip()
                elif 'Time of Day' in line:
                    time_of_day = line.split(': ')[1].strip()
    except Exception as e:
        print(f"Error extracting carbon number, ring, and time of day from file {file_path}: {e}")
    return carbon, ring, time_of_day

# Function to extract log abs current density and voltage from the file
def extract_log_abs_current_density_and_voltage(file_path):
    log_abs_current_density = []
    voltage = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if 'Log Abs Current Density' in line:
                    values = line.split(': ')[1].strip().strip('[]').split(',')
                    log_abs_current_density = [float(value) for value in values]
                elif 'Voltage' in line:
                    values = line.split(': ')[1].strip().strip('[]').split(',')
                    voltage = [float(value) for value in values]
    except Exception as e:
        print(f"Error extracting log abs current density and voltage from file {file_path}: {e}")
    
    return log_abs_current_density, voltage

# Iterate over the files in the amides datasets
for dataset_name, dataset_path in paths.items():
    for root, dirs, files in os.walk(dataset_path):
        for file_name in files:
            if file_name.endswith("_alldata.txt") and ".txt_metadata" not in file_name:
                file_path = os.path.join(root, file_name)
                carbon, ring, time_of_day = extract_carbon_number_ring_and_time_of_day(file_path)
                
                # Check if both carbon, ring, and time of day are valid before accessing data_dict
                if carbon and ring in data_dict and time_of_day in data_dict[ring][f'C{carbon}']:
                    carbon_key = f'C{carbon}'
                    if carbon_key in data_dict[ring]:
                        log_abs_current_density, voltage = extract_log_abs_current_density_and_voltage(file_path)
                        data_dict[ring][carbon_key][time_of_day]['log_abs_current_density'].extend(log_abs_current_density)
                        data_dict[ring][carbon_key][time_of_day]['voltage'].extend(voltage)
                        data_dict[ring][carbon_key][time_of_day]['file_names'].append(file_name)
                        if any(value > 0 for value in log_abs_current_density):
                            data_dict[ring][carbon_key][time_of_day]['num_shorts'] += 1
                    else:
                        print(f"Skipping file {file_path} with unknown carbon {carbon} or ring {ring}")
               

# Calculate statistics for each amide-carbon-time of day combination
statistics = []

for ring, carbon_data in data_dict.items():
    for carbon, time_of_day_data in carbon_data.items():
        for time_of_day, values in time_of_day_data.items():
            log_abs_current_density = np.array(values['log_abs_current_density'])
            unique_file_names = len(set(values['file_names']))
            if unique_file_names > 0:
                values['short_ratio'] = (values['num_shorts'] / unique_file_names) * 100
            else:
                values['short_ratio'] = 0
            if log_abs_current_density.size > 0:
                log_abs_current_density = np.where(np.isinf(log_abs_current_density), np.nan, log_abs_current_density)
                average_log_abs_current_density = np.nanmean(log_abs_current_density)
                stdev_log_abs_current_density = np.nanstd(log_abs_current_density)
                kurtosis_log_abs_current_density = kurtosis(log_abs_current_density, nan_policy='omit')
                skewness_log_abs_current_density = skew(log_abs_current_density, nan_policy='omit')
            else:
                average_log_abs_current_density = 0
                stdev_log_abs_current_density = 0
                kurtosis_log_abs_current_density = 0
                skewness_log_abs_current_density = 0
            
            statistics.append({
                'Ring': ring,
                'Carbon': carbon,
                'Time of Day': time_of_day,
                'Average Log Abs Current Density': average_log_abs_current_density,
                'Standard Deviation': stdev_log_abs_current_density,
                'Kurtosis': kurtosis_log_abs_current_density,
                'Skewness': skewness_log_abs_current_density,
                'Num Shorts': values['num_shorts'],
                'Short Ratio': values['short_ratio']
            })

# Convert statistics to a DataFrame
df_statistics = pd.DataFrame(statistics)

# Save statistics to an Excel file
output_excel_path = 'C:\\Users\\Owner\\OneDrive\\Desktop\\amide_plots\\amide_statistics_time_of_day.xlsx'
df_statistics.to_excel(output_excel_path, index=False)

# Define marker styles for each ring type
marker_styles = {
    'Pentane': 'o',
    'Butane': 's',
    'Propane': 'D',
    'Ethane': '^',
    'DEA': 'v',
    'DPA': 'p'
}

# Plot the statistics
plot_folder = 'C:\\Users\\Owner\\OneDrive\\Desktop\\amide_plots'

# Plotting helper function
def plot_statistic(stat_name):
    plt.figure(figsize=(10, 6))
    for ring, group_data in df_statistics.groupby('Ring'):
        for time_of_day, time_of_day_data in group_data.groupby('Time of Day'):
            plt.scatter(time_of_day_data['Carbon'].apply(lambda x: int(x[1:])), time_of_day_data[stat_name],
                        label=f'{ring} - {time_of_day}', marker=marker_styles[ring])
    
    # Create legend with only the points that were plotted
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = list(set(labels))  # Get unique labels
    unique_handles = [handles[labels.index(label)] for label in unique_labels]  # Match handles to unique labels
    plt.legend(unique_handles, unique_labels, loc='upper left', bbox_to_anchor=(1, 0.5), title='Ring - Time of Day')
    
    # Place legend at carbon=10.5 on the x-axis
    plt.axvline(x=10.5, color='gray', linestyle='--')  # Vertical line at carbon=10.5 for reference
    
    plt.xlabel('Carbon Number')
    plt.ylabel(stat_name)
    plt.title(f'{stat_name} by Carbon Number, Time of Day, and Ring Type')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, f'{stat_name}_by_carbon_time_of_day_and_ring.png'))
    plt.close()

# Plot each statistic
plot_statistic('Average Log Abs Current Density')
plot_statistic('Standard Deviation')
plot_statistic('Kurtosis')
plot_statistic('Skewness')
plot_statistic('Num Shorts')
plot_statistic('Short Ratio')

# Define paths to your data directories
paths = {
    'dataset1': 'C:\\Users\\Owner\\OneDrive\\Desktop\\Thuo Group Data\\Origin Data\\Dataset1',
    'dataset2': 'C:\\Users\\Owner\\OneDrive\\Desktop\\Thuo Group Data\\Origin Data\\Dataset2',
    # Add more datasets as needed
}

# Initialize data_dict with required keys and initial values for amides
# Assuming seasons and time of day are predefined lists
seasons = ['Spring', 'Summer', 'Fall', 'Winter']
time_of_days = ['Morning', 'Noon', 'Afternoon', 'Evening']

data_dict = {
    'Pentane': {f'C{carbon}': {f'{season}_{time_of_day}': {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'short_ratio': 0, 'average_log_abs_current_density': 0} for time_of_day in time_of_days for season in seasons} for carbon in range(10, 13)},
    'Butane': {f'C{carbon}': {f'{season}_{time_of_day}': {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'short_ratio': 0, 'average_log_abs_current_density': 0} for time_of_day in time_of_days for season in seasons} for carbon in range(10, 13)},
    'Propane': {f'C{carbon}': {f'{season}_{time_of_day}': {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'short_ratio': 0, 'average_log_abs_current_density': 0} for time_of_day in time_of_days for season in seasons} for carbon in range(10, 12)},
    'Ethane': {f'C{carbon}': {f'{season}_{time_of_day}': {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'short_ratio': 0, 'average_log_abs_current_density': 0} for time_of_day in time_of_days for season in seasons} for carbon in range(10, 12)},
    'DEA': {f'C{carbon}': {f'{season}_{time_of_day}': {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'short_ratio': 0, 'average_log_abs_current_density': 0} for time_of_day in time_of_days for season in seasons} for carbon in range(11, 13)},
    'DPA': {f'C{carbon}': {f'{season}_{time_of_day}': {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'short_ratio': 0, 'average_log_abs_current_density': 0} for time_of_day in time_of_days for season in seasons} for carbon in range(11, 13)}
}

# Function to extract carbon number, ring, season, and time of day information from file
def extract_metadata(file_path):
    ring = None
    carbon = None
    season = None
    time_of_day = None
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if 'Carbon Number on Chain' in line:
                    carbon = line.split(': ')[1].strip()
                elif 'Ring Pattern' in line:
                    ring = line.split(': ')[1].strip()
                elif 'Season' in line:
                    season = line.split(': ')[1].strip()
                elif 'Time of Day' in line:
                    time_of_day = line.split(': ')[1].strip()
    except Exception as e:
        print(f"Error extracting metadata from file {file_path}: {e}")
    return carbon, ring, season, time_of_day

# Function to extract log abs current density and voltage from the file
def extract_data(file_path):
    log_abs_current_density = []
    voltage = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if 'Log Abs Current Density' in line:
                    values = line.split(': ')[1].strip().strip('[]').split(',')
                    log_abs_current_density = [float(value) for value in values]
                elif 'Voltage' in line:
                    values = line.split(': ')[1].strip().strip('[]').split(',')
                    voltage = [float(value) for value in values]
    except Exception as e:
        print(f"Error extracting data from file {file_path}: {e}")
    
    return log_abs_current_density, voltage

# Iterate over the files in the datasets
for dataset_name, dataset_path in paths.items():
    for root, dirs, files in os.walk(dataset_path):
        for file_name in files:
            if file_name.endswith("_alldata.txt") and ".txt_metadata" not in file_name:
                file_path = os.path.join(root, file_name)
                carbon, ring, season, time_of_day = extract_metadata(file_path)
                
                # Check if both carbon, ring, season, and time of day are valid before accessing data_dict
                if carbon and ring in data_dict and f'C{carbon}' in data_dict[ring]:
                    if f'{season}_{time_of_day}' in data_dict[ring][f'C{carbon}']:
                        log_abs_current_density, voltage = extract_data(file_path)
                        data_dict[ring][f'C{carbon}'][f'{season}_{time_of_day}']['log_abs_current_density'].extend(log_abs_current_density)
                        data_dict[ring][f'C{carbon}'][f'{season}_{time_of_day}']['voltage'].extend(voltage)
                        data_dict[ring][f'C{carbon}'][f'{season}_{time_of_day}']['file_names'].append(file_name)
                        if any(value > 0 for value in log_abs_current_density):
                            data_dict[ring][f'C{carbon}'][f'{season}_{time_of_day}']['num_shorts'] += 1
                    else:
                        print(f"Skipping file {file_path} with unknown season {season} or time of day {time_of_day}")
                else:
                    print(f"Skipping file {file_path} due to missing or invalid metadata")

# Calculate statistics for each amide-carbon-season-time combination
statistics = []

for ring, carbon_data in data_dict.items():
    for carbon, season_time_data in carbon_data.items():
        for season_time, values in season_time_data.items():
            log_abs_current_density = np.array(values['log_abs_current_density'])
            unique_file_names = len(set(values['file_names']))
            if unique_file_names > 0:
                values['short_ratio'] = (values['num_shorts'] / unique_file_names) * 100
            else:
                values['short_ratio'] = 0
            if log_abs_current_density.size > 0:
                log_abs_current_density = np.where(np.isinf(log_abs_current_density), np.nan, log_abs_current_density)
                average_log_abs_current_density = np.nanmean(log_abs_current_density)
                stdev_log_abs_current_density = np.nanstd(log_abs_current_density)
                kurtosis_log_abs_current_density = kurtosis(log_abs_current_density, nan_policy='omit')
                skewness_log_abs_current_density = skew(log_abs_current_density, nan_policy='omit')
            else:
                average_log_abs_current_density = 0
                stdev_log_abs_current_density = 0
                kurtosis_log_abs_current_density = 0
                skewness_log_abs_current_density = 0
            
            statistics.append({
                'Ring': ring,
                'Carbon': carbon,
                'Season_Time': season_time,
                'Average Log Abs Current Density': average_log_abs_current_density,
                'Standard Deviation': stdev_log_abs_current_density,
                'Kurtosis': kurtosis_log_abs_current_density,
                'Skewness': skewness_log_abs_current_density,
                'Num Shorts': values['num_shorts'],
                'Short Ratio': values['short_ratio']
            })

# Convert statistics to a DataFrame
df_statistics = pd.DataFrame(statistics)

# Save statistics to an Excel file
output_excel_path = 'C:\\Users\\Owner\\OneDrive\\Desktop\\amide_plots\\amide_statistics_season_time.xlsx'
df_statistics.to_excel(output_excel_path, index=False)

# Define marker styles for each ring type
marker_styles = {
    'Pentane': 'o',
    'Butane': 's',
    'Propane': 'D',
    'Ethane': '^',
    'DEA': 'v',
    'DPA': 'p'
}

# Filtered statistics based on what's plotted
filtered_statistics = df_statistics.copy()

# Plotting helper function
def plot_statistic(stat_name):
    plt.figure(figsize=(10, 6))
    for ring, group_data in df_statistics.groupby('Ring'):
        for season_time, season_time_data in group_data.groupby('Season_Time'):
            plt.scatter(season_time_data['Carbon'].apply(lambda x: int(x[1:])), season_time_data[stat_name],
                        label=f'{ring} - {season_time}', marker=marker_styles[ring])
    plt.xlabel('Carbon Number')
    plt.ylabel(stat_name)
    plt.title(f'{stat_name} vs Carbon Number by Ring Type and Season_Time')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xticks(range(10, 14))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, f'{stat_name}_season_time.png'))
    plt.show()

# Plot each statistic
plot_statistic('Average Log Abs Current Density')
plot_statistic('Standard Deviation')
plot_statistic('Kurtosis')
plot_statistic('Skewness')
plot_statistic('Num Shorts')
plot_statistic('Short Ratio')

print("Filtered and plotted combinations successfully.")

# List of seasons and times
seasons = ['Summer', 'Winter', 'Fall', 'Spring']
times = ['Morning', 'Afternoon', 'Evening', 'Dusk']

# Initialize data_dict with required keys and initial values for amides
data_dict = {
    'Pentane': {f'C{carbon}': {season: {time: {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'short_ratio': 0, 'average_log_abs_current_density': 0} for time in times} for season in seasons} for carbon in range(10, 13)},
    'Butane': {f'C{carbon}': {season: {time: {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'short_ratio': 0, 'average_log_abs_current_density': 0} for time in times} for season in seasons} for carbon in range(10, 13)},
    'Propane': {f'C{carbon}': {season: {time: {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'short_ratio': 0, 'average_log_abs_current_density': 0} for time in times} for season in seasons} for carbon in range(10, 12)},
    'Ethane': {f'C{carbon}': {season: {time: {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'short_ratio': 0, 'average_log_abs_current_density': 0} for time in times} for season in seasons} for carbon in range(10, 12)},
    'DEA': {f'C{carbon}': {season: {time: {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'short_ratio': 0, 'average_log_abs_current_density': 0} for time in times} for season in seasons} for carbon in range(11, 13)},
    'DPA': {f'C{carbon}': {season: {time: {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'short_ratio': 0, 'average_log_abs_current_density': 0} for time in times} for season in seasons} for carbon in range(11, 13)}
}

# List of seasons and times
seasons = ['Summer', 'Winter']
times = ['Morning', 'Afternoon', 'Evening']

# Function to extract carbon number, ring, season, and time information from file
def extract_carbon_number_ring_season_and_time(file_path):
    ring = None
    carbon = None
    season = None
    time = None
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if 'Carbon Number on Chain' in line:
                    carbon = line.split(': ')[1].strip()
                elif 'Ring Pattern' in line:
                    ring = line.split(': ')[1].strip()
                elif 'Season' in line:
                    season = line.split(': ')[1].strip()
                elif 'Time' in line:
                    time = line.split(': ')[1].strip()
    except Exception as e:
        print(f"Error extracting carbon number, ring, season, and time from file {file_path}: {e}")
    return carbon, ring, season, time

# Function to extract log abs current density and voltage from the file
def extract_log_abs_current_density_and_voltage(file_path):
    log_abs_current_density = []
    voltage = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if 'Log Abs Current Density' in line:
                    values = line.split(': ')[1].strip().strip('[]').split(',')
                    log_abs_current_density = [float(value) for value in values]
                elif 'Voltage' in line:
                    values = line.split(': ')[1].strip().strip('[]').split(',')
                    voltage = [float(value) for value in values]
    except Exception as e:
        print(f"Error extracting log abs current density and voltage from file {file_path}: {e}")
    
    return log_abs_current_density, voltage



# Define the paths and folders for AgTS and AuTi datasets
paths = {
    'Dataset 1': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\From Shawn Old\Charge Tunneling Data\Charge Tunneling Data\C10CNC4',
    'Dataset 2': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\From Shawn Old\Charge Tunneling Data\Charge Tunneling Data\C10CNC5',
    'Dataset 3': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\From Shawn Old\Charge Tunneling Data\Charge Tunneling Data\C11CNC4',
    'Dataset 4': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\From Shawn Old\Charge Tunneling Data\Charge Tunneling Data\C11CNC5',
    'Dataset 5': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\From Shawn Old\Charge Tunneling Data\Charge Tunneling Data\C11CNC3',
    'Dataset 6': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\From Shawn Old\Charge Tunneling Data\Charge Tunneling Data\C11CNC2',
    'Dataset 7': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\From Shawn Old\Charge Tunneling Data\Charge Tunneling Data\C10CNC3',
    'Dataset 8': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\From Shawn Old\Charge Tunneling Data\Charge Tunneling Data\C10CNC2',
    'Dataset 9': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\From Shawn Old\Charge Tunneling Data\Charge Tunneling Data\C11DEA',
    'Dataset 10': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\From Shawn Old\Charge Tunneling Data\Charge Tunneling Data\C12DEA',
    'Dataset 11': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\From Shawn Old\Charge Tunneling Data\Charge Tunneling Data\C11DPA',
    'Dataset 12':  r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\From Shawn Old\Charge Tunneling Data\Charge Tunneling Data\C12DPA',
    'Dataset 13': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\From Shawn Old\Charge Tunneling Data\Charge Tunneling Data\C12CNC4', 
    'Dataset 14': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\From Shawn Old\Charge Tunneling Data\Charge Tunneling Data\C12CNC5'
}

folders_Data1 = [r'02132018_1', r'02132018_2', r'12142017']
folders_Data2 = [r'12022017_1', r'12022017_2' ]
folders_Data3=[r'03242018_1', r'C11CNC4\03042018', r'03242018_2', r'04152018_1', r'04152018_2', r'04172018_1', r'04172018_2']
folders_Data4=[r'peak2\02152018', r'peak2\03272018', r'peak2\04082018_1', r'peak2\04082018_2', r'peak2\04182018_1', r'peak2\04202018', r'Peak1\02142018', r'C11CNC5\Peak1\03242018', r'Peak1\04182018_2', r'Peak1\12142017', r'20090825(C11NPPd 3h)', r'20090908(C11NPPd 3h)']
folders_Data5=[r'CD1-81a-12252018', r'CD1-81b-12252018']
folders_Data6=[r'CD1-83a-12262018', r'CD1-83a-12262018\CD1-83b-12262018']
folders_Data7=[r'CD1-80a-12212018', r'CD1-80b-12212018']
folders_Data8=[r'CD1-84c-12272018', r'CD1-85a-01042019']
folders_Data9=[r'20090910 (C11DEA 3h RT)']
folders_Data10=[r'C12DEA']
folders_Data11=[r'20090909 (C11DPA 3h RT)']
folders_Data12=[r'20090813 C12DPA (3h RT)']
folders_Data13=[r'C12prr\1', r'C12prr\2' ]
folders_Data14=[r'C12ppd\1', r'C12ppd\2']
# Iterate over the files in the amides datasets
for dataset_name, dataset_path in paths.items():
    for root, dirs, files in os.walk(dataset_path):
        for file_name in files:
            if file_name.endswith("_alldata.txt") and ".txt_metadata" not in file_name:
                file_path = os.path.join(root, file_name)
                carbon, ring, season, time = extract_carbon_number_ring_season_and_time(file_path)
                
                # Check if both carbon, ring, season, and time are valid before accessing data_dict
                if carbon and ring in data_dict and season in data_dict[ring][f'C{carbon}'] and time in data_dict[ring][f'C{carbon}'][season]:
                    carbon_key = f'C{carbon}'
                    if carbon_key in data_dict[ring] and season in data_dict[ring][carbon_key] and time in data_dict[ring][carbon_key][season]:
                        log_abs_current_density, voltage = extract_log_abs_current_density_and_voltage(file_path)
                        data_dict[ring][carbon_key][season][time]['log_abs_current_density'].extend(log_abs_current_density)
                        data_dict[ring][carbon_key][season][time]['voltage'].extend(voltage)
                        data_dict[ring][carbon_key][season][time]['file_names'].append(file_name)
                        if any(value > 0 for value in log_abs_current_density):
                            data_dict[ring][carbon_key][season][time]['num_shorts'] += 1
                    else:
                        print(f"Skipping file {file_path} with unknown carbon {carbon}, season {season}, or time {time} for ring {ring}")
                else:
                    print(f"Skipping file {file_path} due to missing or invalid metadata")


# Calculate statistics for each amide-carbon-season-time combination
statistics = []

for ring, carbon_data in data_dict.items():
    for carbon, season_data in carbon_data.items():
        for season, time_data in season_data.items():
            for time, values in time_data.items():
                log_abs_current_density = np.array(values['log_abs_current_density'])
                unique_file_names = len(set(values['file_names']))
                if unique_file_names > 0:
                    values['short_ratio'] = (values['num_shorts'] / unique_file_names) * 100
                else:
                    values['short_ratio'] = 0
                if log_abs_current_density.size > 0:
                    log_abs_current_density = np.where(np.isinf(log_abs_current_density), np.nan, log_abs_current_density)
                    average_log_abs_current_density = np.nanmean(log_abs_current_density)
                    stdev_log_abs_current_density = np.nanstd(log_abs_current_density)
                    kurtosis_log_abs_current_density = kurtosis(log_abs_current_density, nan_policy='omit')
                    skewness_log_abs_current_density = skew(log_abs_current_density, nan_policy='omit')
                else:
                    average_log_abs_current_density = 0
                    stdev_log_abs_current_density = 0
                    kurtosis_log_abs_current_density = 0
                    skewness_log_abs_current_density = 0
                
                # Only add to statistics if average log abs current density < 2
                if average_log_abs_current_density < 0:
                    statistics.append({
                        'Ring': ring,
                        'Carbon': carbon,
                        'Season': season,
                        'Time': time,
                        'Average Log Abs Current Density': average_log_abs_current_density,
                        'Standard Deviation': stdev_log_abs_current_density,
                        'Kurtosis': kurtosis_log_abs_current_density,
                        'Skewness': skewness_log_abs_current_density,
                        'Num Shorts': values['num_shorts'],
                        'Short Ratio': values['short_ratio']
                    })

# Verify contents of statistics
print("Statistics List:", statistics[:5])  # Print first 5 entries to check

# Convert statistics to a DataFrame
df_statistics = pd.DataFrame(statistics)

# Verify DataFrame contents
print("DataFrame Columns:", df_statistics.columns)
print(df_statistics.head())

# Save statistics to an Excel file
output_excel_path = 'C:\\Users\\Owner\\OneDrive\\Desktop\\amide_plots\\amide_statistics_season_time.xlsx'
df_statistics.to_excel(output_excel_path, index=False)

# Define marker styles for each ring type
marker_styles = {
    'Pentane': 'o',
    'Butane': 's',
    'Propane': 'D',
    'Ethane': '^',
    'DEA': 'v',
    'DPA': 'p'
}

# Plot folder
plot_folder = 'C:\\Users\\Owner\\OneDrive\\Desktop\\amide_plots'
os.makedirs(plot_folder, exist_ok=True)

# Function to plot statistics
def plot_statistic(stat_name):
    plt.figure(figsize=(10, 6))
    plotted_combinations = set()
    
    for ring, group_data in df_statistics.groupby('Ring'):
        for season, season_data in group_data.groupby('Season'):
            for time, time_data in season_data.groupby('Time'):
                x_values = time_data['Carbon'].apply(lambda x: int(x[1:]))
                y_values = time_data[stat_name]
                if not x_values.empty and not y_values.empty:
                    plt.scatter(x_values, y_values, label=f'{ring} - {season} - {time}', marker=marker_styles[ring])
                    plotted_combinations.add((ring, season, time))
    
    # Legend positioning
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.xlabel('Carbon Number')
    plt.ylabel(stat_name)
    plt.title(f'{stat_name} vs Carbon Number by Ring Type, Season, and Time')
    plt.xticks(range(10, 14))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, f'{stat_name}_season_time.png'))
    plt.show()

# Plot each statistic
plot_statistic('Average Log Abs Current Density')
plot_statistic('Standard Deviation')
plot_statistic('Kurtosis')
plot_statistic('Skewness')
plot_statistic('Num Shorts')
plot_statistic('Short Ratio')

print("Script execution completed successfully.")

import pandas as pd

# Load the existing data from the Excel file
input_excel_path = 'C:\\Users\\Owner\\OneDrive\\Desktop\\amide_plots\\amide_statistics_season_time.xlsx'
df = pd.read_excel(input_excel_path)

# Group by unique ring and season combinations
grouped_df = df.groupby(['Ring', 'Season']).agg({
    'Average Log Abs Current Density': 'mean',
    'Standard Deviation': 'mean',
    'Kurtosis': 'mean',
    'Skewness': 'mean',
    'Num Shorts': 'sum',
    'Short Ratio': 'mean'
}).reset_index()

# Verify the grouped DataFrame contents
print(grouped_df.head())

# Save the aggregated statistics to a new Excel file
output_excel_path = 'C:\\Users\\Owner\\OneDrive\\Desktop\\amide_plots\\amide_statistics_by_ring_season.xlsx'
grouped_df.to_excel(output_excel_path, index=False)
