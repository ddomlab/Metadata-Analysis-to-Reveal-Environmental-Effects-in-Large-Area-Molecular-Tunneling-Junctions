# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 15:12:30 2024

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



def get_location(file_path):
    if "Dataset 1" in file_path or "Dataset 3" in file_path:
        return "Ames, Iowa"
    elif "Dataset 2" in file_path or "Dataset 4" in file_path:
        return "Boston, Massachusetts"
    else:
        return "Unknown"

def get_user(file_path):
    if "Dataset 1" in file_path or "Dataset 3" in file_path:
        return "User 1"
    elif "Dataset 2" in file_path or "Dataset 4" in file_path:
        return "User 2"
    else:
        return "Unknown"

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
            user = get_user(file_path)
            if user is None:
                print(f"File path: {file_path} - Unknown parameter: User")
            location = get_location(file_path)
            if location is None:
                print(f"File path: {file_path} - Unknown parameter: Location")
            
            metadata = {
                'Date': date,
                'Month': month,
                'File Name': os.path.basename(file_path),
                'Season': season,
                'Time': time,
                'Electrode': electrode,
                'Carbon Number': carbon,
                'Time of Day': time_of_day,
                'Junction Diameter': junction_diameter,
                'User': user,
                'Location': location
            }
            
            return metadata
    except Exception as e:
        print(f"Error extracting metadata from file {file_path}: {e}")
        return {}
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
    'Dataset 1': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\Combined Au\Dataset 1',
    'Dataset 2': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\Combined Au\Dataset 2',
    'Dataset 3': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\Combined Ag\Dataset 1',
    'Dataset 4': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\Combined Ag\Dataset 2',
}

folders_Data1 = ['C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16']
folders_Data2 = [r'C10SH - pur\10132011', r'C11SH - pur\05242012 (Windsor)', r'C11SH - pur\TS Au - SC11 (Harvard)',r'C12SH - pur\09222011', r'C13SH - pur\01302012', r'C14SH - pur\09202011', r'C15SH - pur\12152011', r'C16SH - pur\09072011', r'C16SH - pur\09152011' ]
folders_Data3=['C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16']
folders_Data4=[r'C10\20090730 (C10 RT 3h)', r'C10\20090812(C10)',r'C11\20090728 (C11 RT 3h)',r'C12\20080513 (C-12 recryst on Ag RT)\sample 1',r'C12\20080513 (C-12 recryst on Ag RT)\sample 2', r'C12\20080513 (C-12 recryst on Ag RT)\sample 3', r'C12\20090413 (C12 Xl RT 2h)',
               r'C12\20090415 (C12 Xl RT 3.5h)',r'C13\20090413 (C13 RT 3h)', r'C13\20090414(C13 RT 3h)', r'C13\20090414(C13 RT 3h B)', r'C14\20090730 (C14 RT 3h)', r'C14\20090822 (C14 RT 3h)', r'C15\20090725(C15 RT 3h)', r'C16\20080807 C16\20080807 (C16 Ag RT 16.5hrs)', r'C16\20080807 C16\20080807 (C16 Ag RT 19hrs)',
               r'C16\20080812 C16\20080812 (C16 Ag RT 19hrs)', r'C16\20080812 C16\20080812 (C16 Ag RT 21_5hrs)', r'C16\20100601(C16 3h)', r'C16\20090814(C16 RT 3h)\True']
# Pair metadata and data files
metadata_files_1, data_files_1, num_pairs_1 = pair_files(paths['Dataset 1'], folders_Data1)
metadata_files_2, data_files_2, num_pairs_2 = pair_files(paths['Dataset 2'], folders_Data2)
metadata_files_3, data_files_3, num_pairs_3 = pair_files(paths['Dataset 3'], folders_Data3)
metadata_files_4, data_files_4, num_pairs_4= pair_files(paths['Dataset 4'], folders_Data4)

print(f"\nNumber of pairs in Data1 dataset: {num_pairs_1}")
print(f"Number of pairs in Data2 dataset: {num_pairs_2}")
print(f"Number of pairs in Data2 dataset: {num_pairs_3}")
print(f"Number of pairs in Data2 dataset: {num_pairs_4}")
# Configure logging
logging.basicConfig(filename='file_processing.log', level=logging.INFO, format='%(asctime)s %(message)s')



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Define paths
from collections import Counter

# Define paths
paths = {
    'Au Dataset 1': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\Combined Au\Dataset 1',
    'Au Dataset 2': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\Combined Au\Dataset 2',
    'Ag Dataset 1': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\Combined Ag\Dataset 1',
    'Ag Dataset 2': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\Combined Ag\Dataset 2',
}

# Create dictionaries to store log abs current density, voltage values, and file names
data_dict = {
    'Au': {f'C{carbon}': {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'users': [], 'num_shorts': 0, 'short_ratio': 0, 'files_with_shorts': set()} for carbon in range(10, 17)},
    'Ag': {f'C{carbon}': {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'users': [], 'num_shorts': 0, 'short_ratio': 0, 'files_with_shorts': set()} for carbon in range(10, 17)},
}

# Function to extract carbon number, electrode, and user information from file
def extract_info_from_file(file_path):
    carbon, electrode, user = None, None, None
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if 'Carbon Number' in line:
                    carbon = line.split(': ')[1].strip()
                elif 'Electrode' in line:
                    electrode = line.split(': ')[1].strip()
                elif 'User' in line:
                    user = line.split(': ')[1].strip()
    except Exception as e:
        print(f"Error extracting information from file {file_path}: {e}")
    return carbon, electrode, user

# Function to extract log abs current density and voltage from the file
def extract_log_abs_current_density_and_voltage(file_path):
    log_abs_current_density, voltage = [], []
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

# Iterate over the files in the datasets
for dataset_name, dataset_path in paths.items():
    for root, dirs, files in os.walk(dataset_path):
        for file_name in files:
            if file_name.endswith("_alldata.txt") and ".txt_metadata" not in file_name:
                file_path = os.path.join(root, file_name)
                carbon, electrode, user = extract_info_from_file(file_path)
                log_abs_current_density, voltage = extract_log_abs_current_density_and_voltage(file_path)
                
                if carbon and electrode and user:
                    key = f'C{carbon}'
                    if electrode in data_dict and key in data_dict[electrode]:
                        data_length = min(len(log_abs_current_density), len(voltage))
                        data_dict[electrode][key]['log_abs_current_density'].extend(log_abs_current_density[:data_length])
                        data_dict[electrode][key]['voltage'].extend(voltage[:data_length])
                        data_dict[electrode][key]['file_names'].extend([file_name] * data_length)
                        data_dict[electrode][key]['users'].extend([user] * data_length)
                        
                        # Count the number of shorts per file for each user
                        if any(value > 0 for value in log_abs_current_density):
                            data_dict[electrode][key]['files_with_shorts'].add((file_name, user))

# Update num_shorts based on unique files with shorts
for electrode, carbon_data in data_dict.items():
    for carbon, values in carbon_data.items():
        values['num_shorts'] = len(values['files_with_shorts'])

# Calculate short ratio for each combination
for electrode, carbon_data in data_dict.items():
    for carbon, values in carbon_data.items():
        unique_file_names = len(set(values['file_names']))
        if unique_file_names > 0:
            values['short_ratio'] = (values['num_shorts'] / unique_file_names) * 100
        else:
            values['short_ratio'] = 0

# Create dataframes and write to Excel
output_path = r'C:\Users\Owner\OneDrive\Desktop\sync pending\log_abs_current_density_voltage.xlsx'
with pd.ExcelWriter(output_path) as writer:
    for electrode, carbon_data in data_dict.items():
        for carbon, values in carbon_data.items():
            data_length = min(len(values['log_abs_current_density']), len(values['voltage']), len(values['file_names']), len(values['users']))
            df = pd.DataFrame({
                'Voltage': values['voltage'][:data_length],
                'Log(Abs Current Density)': values['log_abs_current_density'][:data_length],
                'File Names': values['file_names'][:data_length],
                'User': values['users'][:data_length]
            })
            df['Shorts'] = [values['num_shorts']] * data_length  # Add the shorts count as a single integer
            df['Short Ratio'] = [values['short_ratio']] * data_length
            df.to_excel(writer, sheet_name=f'{electrode}_{carbon}', index=False)

# Define the function to plot heatmaps
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
    plt.text(xedges[-1], ylim_min, f"Shorted Attempts: {num_shorts}", color='yellow', fontsize=20, ha='right', va='bottom', weight='bold')
    plt.text(xedges[0], ylim_min, f"Short Ratio (%): {short_ratio:.2f}", color='yellow', fontsize=20, ha='left', va='bottom', weight='bold')
    plt.title(title, fontsize=24, weight='bold')  # Increase font size and make it bold
    plt.xlabel('Voltage (V)', fontsize=21, weight='bold')  # Increase font size and make it bold
    plt.ylabel('Log Abs Current Density (A/cm²)', fontsize=21, weight='bold')  # Increase font size and make it bold
    plt.xticks(fontsize=16, weight='bold')  # Increase font size for ticks
    plt.yticks(fontsize=16, weight='bold')  # Increase font size for ticks
    plt.ylim(ylim_min, ylim_max)  # Set the y-axis limits

    # Save the heatmap to a file
    heatmap_output_path = r'C:\Users\Owner\OneDrive\Desktop\Carbon Number Electrode'
    if not os.path.exists(heatmap_output_path):
        os.makedirs(heatmap_output_path)
    plt.savefig(os.path.join(heatmap_output_path, f'{title}.png'), bbox_inches='tight')
    plt.close()

# Plot heatmaps for each combination
for electrode, carbon_data in data_dict.items():
    for carbon, values in carbon_data.items():
        plot_heatmap(
            values['voltage'],
            values['log_abs_current_density'],
            title=f'{electrode}_{carbon}_Heatmap',
            num_shorts=values['num_shorts'],
            short_ratio=values['short_ratio']
        )

# Define paths for the datasets
paths = {
    'Au Dataset 1': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\Combined Au\Dataset 1',
    'Au Dataset 2': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\Combined Au\Dataset 2',
    'Ag Dataset 1': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\Combined Ag\Dataset 1',
    'Ag Dataset 2': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\Combined Ag\Dataset 2',
}

# Create dictionaries to store log abs current density, voltage values, and file names
data_dict = {
    'Au': {f'C{carbon}': {season: {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'users': [],'short_ratio': 0, 'files_with_shorts': set()} for season in ['Spring', 'Summer', 'Fall', 'Winter']} for carbon in range(10, 17)},
    'Ag': {f'C{carbon}': {season: {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'users': [],'short_ratio': 0, 'files_with_shorts': set()} for season in ['Spring', 'Summer', 'Fall', 'Winter']} for carbon in range(10, 17)},
}

# Function to extract carbon number, electrode, season, and user information from file
def extract_info_from_file(file_path):
    carbon, electrode, season, user = None, None, None, None
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if 'Carbon Number' in line:
                    carbon = line.split(': ')[1].strip()
                elif 'Electrode' in line:
                    electrode = line.split(': ')[1].strip()
                elif 'Season' in line:
                    season = line.split(': ')[1].strip()
                elif 'User' in line:
                    user = line.split(': ')[1].strip()
    except Exception as e:
        print(f"Error extracting information from file {file_path}: {e}")
    return carbon, electrode, season, user

# Function to extract log abs current density and voltage from the file
def extract_log_abs_current_density_and_voltage(file_path):
    log_abs_current_density, voltage = [], []
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

# Iterate over the files in the datasets
for dataset_name, dataset_path in paths.items():
    for root, dirs, files in os.walk(dataset_path):
        for file_name in files:
            if file_name.endswith("_alldata.txt") and ".txt_metadata" not in file_name:
                file_path = os.path.join(root, file_name)
                carbon, electrode, season, user = extract_info_from_file(file_path)
                log_abs_current_density, voltage = extract_log_abs_current_density_and_voltage(file_path)
                
                if carbon and electrode and season and user:
                    carbon_key = f'C{carbon}'
                    if electrode in data_dict and carbon_key in data_dict[electrode] and season in data_dict[electrode][carbon_key]:
                        data_length = min(len(log_abs_current_density), len(voltage))
                        data_dict[electrode][carbon_key][season]['log_abs_current_density'].extend(log_abs_current_density[:data_length])
                        data_dict[electrode][carbon_key][season]['voltage'].extend(voltage[:data_length])
                        data_dict[electrode][carbon_key][season]['file_names'].extend([file_name] * data_length)
                        data_dict[electrode][carbon_key][season]['users'].extend([user] * data_length)
                        
                        # Count the number of shorts per file for each user
                        if any(value > 0 for value in log_abs_current_density):
                            data_dict[electrode][carbon_key][season]['files_with_shorts'].add((file_name, user))

# Update num_shorts based on unique files with shorts
for electrode, carbon_data in data_dict.items():
    for carbon, season_data in carbon_data.items():
        for season, values in season_data.items():
            values['num_shorts'] = len(values['files_with_shorts'])

# Calculate short ratio for each combination
for electrode, carbon_data in data_dict.items():
    for carbon, season_data in carbon_data.items():
        for season, values in season_data.items():
            unique_file_names = len(set(values['file_names']))
            if unique_file_names > 0:
                values['short_ratio'] = (values['num_shorts'] / unique_file_names) * 100
            else:
                values['short_ratio'] = 0

# Create dataframes and write to Excel
output_path = r'C:\Users\Owner\OneDrive\Desktop\sync pending\log_abs_current_density_voltage_seasons_new.xlsx'
with pd.ExcelWriter(output_path) as writer:
    for electrode, carbon_data in data_dict.items():
        for carbon, season_data in carbon_data.items():
            for season, values in season_data.items():
                data_length = min(len(values['log_abs_current_density']), len(values['voltage']), len(values['file_names']), len(values['users']))
                df = pd.DataFrame({
                    'Voltage': values['voltage'][:data_length],
                    'Log(Abs Current Density)': values['log_abs_current_density'][:data_length],
                    'File Names': values['file_names'][:data_length],
                    'User': values['users'][:data_length]
                })
                df['Shorts'] = [values['num_shorts']] * data_length  # Add the shorts count as a single integer
                df['Short Ratio'] = [values['short_ratio']] * data_length
                df.to_excel(writer, sheet_name=f'{electrode}_{carbon}_{season}', index=False)

# Calculate average log abs current density per file and user
average_log_density_dict = {
    'Au': {f'C{carbon}': {season: pd.DataFrame() for season in ['Spring', 'Summer', 'Fall', 'Winter']} for carbon in range(10, 17)},
    'Ag': {f'C{carbon}': {season: pd.DataFrame() for season in ['Spring', 'Summer', 'Fall', 'Winter']} for carbon in range(10, 17)},
}

# Compute the averages
for electrode, carbon_data in data_dict.items():
    for carbon, season_data in carbon_data.items():
        for season, values in season_data.items():
            df = pd.DataFrame({
                'Log(Abs Current Density)': values['log_abs_current_density'],
                'File Names': values['file_names'],
                'User': values['users']
            })
            if not df.empty:
                avg_log_density = df.groupby(['File Names', 'User'])['Log(Abs Current Density)'].mean().reset_index()
                average_log_density_dict[electrode][carbon][season] = avg_log_density

# Write averages to a new Excel sheet
average_output_path = r'C:\Users\Owner\OneDrive\Desktop\sync pending\average_log_abs_current_density_per_file_user.xlsx'
with pd.ExcelWriter(average_output_path) as writer:
    for electrode, carbon_data in average_log_density_dict.items():
        for carbon, season_data in carbon_data.items():
            for season, avg_df in season_data.items():
                if not avg_df.empty:
                    avg_df.to_excel(writer, sheet_name=f'{electrode}_{carbon}_{season}', index=False)

print("Average log abs current density per file and user has been calculated and written to Excel.")
# Define the function to plot heatmaps
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
    plt.text(xedges[-1], ylim_min, f"Shorted Attempts: {num_shorts}", color='yellow', fontsize=20, ha='right', va='bottom', weight='bold')
    plt.text(xedges[0], ylim_min, f"Short Ratio (%): {short_ratio:.2f}", color='yellow', fontsize=20, ha='left', va='bottom', weight='bold')
    plt.title(title, fontsize=24, weight='bold')  # Increase font size and make it bold
    plt.xlabel('Voltage (V)', fontsize=21, weight='bold')  # Increase font size and make it bold
    plt.ylabel('Log Abs Current Density (A/cm²)', fontsize=21, weight='bold')  # Increase font size and make it bold
    plt.xticks(fontsize=16, weight='bold')  # Increase font size for ticks
    plt.yticks(fontsize=16, weight='bold')  # Increase font size for ticks
    plt.ylim(ylim_min, ylim_max)  # Set the y-axis limits

    # Save the heatmap to a file
    heatmap_output_path = r'C:\Users\Owner\OneDrive\Desktop\Seasons Final New'
    if not os.path.exists(heatmap_output_path):
        os.makedirs(heatmap_output_path)
    plt.savefig(os.path.join(heatmap_output_path, f'{title}.png'), bbox_inches='tight')
    plt.close()

# Plot heatmaps for each combination
for electrode, carbon_data in data_dict.items():
    for carbon, season_data in carbon_data.items():
        for season, values in season_data.items():
            title = f"{electrode} - {carbon} - {season}"
            
            plot_heatmap(values['voltage'], values['log_abs_current_density'], title, values['num_shorts'], values['short_ratio'])

# Define paths for the datasets
paths = {
    'Au Dataset 1': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\Combined Au\Dataset 1',
    'Au Dataset 2': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\Combined Au\Dataset 2',
    'Ag Dataset 1': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\Combined Ag\Dataset 1',
    'Ag Dataset 2': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\Combined Ag\Dataset 2',
}

# Create dictionaries to store log abs current density, voltage values, and file names
data_dict = {
    'Au': {f'C{carbon}': {time: {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'users': [], 'short_ratio': 0, 'files_with_shorts': set()} for time in ['Morning', 'Afternoon', 'Evening', 'Night']} for carbon in range(10, 17)},
    'Ag': {f'C{carbon}': {time: {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'users': [], 'short_ratio': 0, 'files_with_shorts': set()} for time in ['Morning', 'Afternoon', 'Evening', 'Night']} for carbon in range(10, 17)},
}

# Function to extract carbon number, electrode, time of day, and user information from file
def extract_info_from_file(file_path):
    carbon, electrode, time, user = None, None, None, None
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if 'Carbon Number' in line:
                    carbon = line.split(': ')[1].strip()
                elif 'Electrode' in line:
                    electrode = line.split(': ')[1].strip()
                elif 'Time of Day' in line:
                    time = line.split(': ')[1].strip()
                elif 'User' in line:
                    user = line.split(': ')[1].strip()
    except Exception as e:
        print(f"Error extracting information from file {file_path}: {e}")
    return carbon, electrode, time, user

# Function to extract log abs current density and voltage from the file
def extract_log_abs_current_density_and_voltage(file_path):
    log_abs_current_density, voltage = [], []
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

# Iterate over the files in the datasets
for dataset_name, dataset_path in paths.items():
    for root, dirs, files in os.walk(dataset_path):
        for file_name in files:
            if file_name.endswith("_alldata.txt") and ".txt_metadata" not in file_name:
                file_path = os.path.join(root, file_name)
                carbon, electrode, time, user = extract_info_from_file(file_path)
                log_abs_current_density, voltage = extract_log_abs_current_density_and_voltage(file_path)
                
                if carbon and electrode and time and user:
                    carbon_key = f'C{carbon}'
                    if electrode in data_dict and carbon_key in data_dict[electrode] and time in data_dict[electrode][carbon_key]:
                        data_length = min(len(log_abs_current_density), len(voltage))
                        data_dict[electrode][carbon_key][time]['log_abs_current_density'].extend(log_abs_current_density[:data_length])
                        data_dict[electrode][carbon_key][time]['voltage'].extend(voltage[:data_length])
                        data_dict[electrode][carbon_key][time]['file_names'].extend([file_name] * data_length)
                        data_dict[electrode][carbon_key][time]['users'].extend([user] * data_length)
                        
                        # Count the number of shorts per file for each user
                        if any(value > 0 for value in log_abs_current_density):
                            data_dict[electrode][carbon_key][time]['files_with_shorts'].add((file_name, user))

# Update num_shorts based on unique files with shorts
for electrode, carbon_data in data_dict.items():
    for carbon, time_data in carbon_data.items():
        for time, values in time_data.items():
            values['num_shorts'] = len(values['files_with_shorts'])

# Calculate short ratio for each combination
for electrode, carbon_data in data_dict.items():
    for carbon, time_data in carbon_data.items():
        for time, values in time_data.items():
            unique_file_names = len(set(values['file_names']))
            if unique_file_names > 0:
                values['short_ratio'] = (values['num_shorts'] / unique_file_names) * 100
            else:
                values['short_ratio'] = 0

# Create dataframes and write to Excel
output_path = r'C:\Users\Owner\OneDrive\Desktop\sync pending\log_abs_current_density_voltage_times.xlsx'
with pd.ExcelWriter(output_path) as writer:
    for electrode, carbon_data in data_dict.items():
        for carbon, time_data in carbon_data.items():
            for time, values in time_data.items():
                data_length = min(len(values['log_abs_current_density']), len(values['voltage']), len(values['file_names']), len(values['users']))
                df = pd.DataFrame({
                    'Voltage': values['voltage'][:data_length],
                    'Log(Abs Current Density)': values['log_abs_current_density'][:data_length],
                    'File Names': values['file_names'][:data_length],
                    'User': values['users'][:data_length]
                })
                df['Shorts'] = [values['num_shorts']] * data_length  # Add the shorts count as a single integer
                df['Short Ratio'] = [values['short_ratio']] * data_length
                df.to_excel(writer, sheet_name=f'{electrode}_{carbon}_{time}', index=False)
# Calculate average log abs current density per file and user
average_log_density_dict = {
    'Au': {f'C{carbon}': {time: pd.DataFrame() for time in ['Morning', 'Afternoon', 'Evening', 'Night']} for carbon in range(10, 17)},
    'Ag': {f'C{carbon}': {time: pd.DataFrame() for time in ['Morning', 'Afternoon', 'Evening', 'Night']} for carbon in range(10, 17)},
}

# Compute the averages
for electrode, carbon_data in data_dict.items():
    for carbon, time_data in carbon_data.items():
        for time, values in time_data.items():
            df = pd.DataFrame({
                'Log(Abs Current Density)': values['log_abs_current_density'],
                'File Names': values['file_names'],
                'User': values['users']
            })
            if not df.empty:
                avg_log_density = df.groupby(['File Names', 'User'])['Log(Abs Current Density)'].mean().reset_index()
                average_log_density_dict[electrode][carbon][time] = avg_log_density

# Write averages to a new Excel sheet
average_output_path = r'C:\Users\Owner\OneDrive\Desktop\sync pending\average_log_abs_current_density_per_file_time.xlsx'
with pd.ExcelWriter(average_output_path) as writer:
    for electrode, carbon_data in average_log_density_dict.items():
        for carbon, time_data in carbon_data.items():
            for time, avg_df in time_data.items():
                if not avg_df.empty:
                    avg_df.to_excel(writer, sheet_name=f'{electrode}_{carbon}_{time}', index=False)

print("Average log abs current density per file and user has been calculated and written to Excel.")
# Define the function to plot heatmaps
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
    plt.text(xedges[-1], ylim_min, f"Shorted Attempts: {num_shorts}", color='yellow', fontsize=20, ha='right', va='bottom', weight='bold')
    plt.text(xedges[0], ylim_min, f"Short Ratio (%): {short_ratio:.2f}", color='yellow', fontsize=20, ha='left', va='bottom', weight='bold')
    plt.title(title, fontsize=24, weight='bold')  # Increase font size and make it bold
    plt.xlabel('Voltage (V)', fontsize=21, weight='bold')  # Increase font size and make it bold
    plt.ylabel('Log Abs Current Density (A/cm²)', fontsize=21, weight='bold')  # Increase font size and make it bold
    plt.xticks(fontsize=16, weight='bold')  # Increase font size for ticks
    plt.yticks(fontsize=16, weight='bold')  # Increase font size for ticks
    plt.ylim(ylim_min, ylim_max)  # Set the y-axis limits

    # Save the heatmap to a file
    heatmap_output_path = r'C:\Users\Owner\OneDrive\Desktop\Time Final'
    if not os.path.exists(heatmap_output_path):
        os.makedirs(heatmap_output_path)
    plt.savefig(os.path.join(heatmap_output_path, f'{title}.png'), bbox_inches='tight')
    plt.close()
# Plot heatmaps for each combination
for electrode, carbon_data in data_dict.items():
    for carbon, time_data in carbon_data.items():
        for time, values in time_data.items():
            title = f'{electrode}_{carbon}_{time}'
            plot_heatmap(values['voltage'], values['log_abs_current_density'], title, values['num_shorts'], values['short_ratio'])

print("Processing and heatmap generation completed.")
# Define paths for the datasets
paths = {
    'Au Dataset 1': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\Combined Au\Dataset 1',
    'Au Dataset 2': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\Combined Au\Dataset 2',
    'Ag Dataset 1': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\Combined Ag\Dataset 1',
    'Ag Dataset 2': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\Combined Ag\Dataset 2',
}

# Create dictionaries to store log abs current density, voltage values, and file names
data_dict = {
    'Au': {season: {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'users': [], 'short_ratio': 0, 'files_with_shorts': set()} for season in ['Winter', 'Spring', 'Summer', 'Fall']},
    'Ag': {season: {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'users': [], 'short_ratio': 0, 'files_with_shorts': set()} for season in ['Winter', 'Spring', 'Summer', 'Fall']},
}

# Function to extract season, electrode, and user information from file
def extract_info_from_file(file_path):
    season, electrode, user = None, None, None
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if 'Season' in line:
                    season = line.split(': ')[1].strip()
                elif 'Electrode' in line:
                    electrode = line.split(': ')[1].strip()
                elif 'User' in line:
                    user = line.split(': ')[1].strip()
    except Exception as e:
        print(f"Error extracting information from file {file_path}: {e}")
    return season, electrode, user

# Function to extract log abs current density and voltage from the file
def extract_log_abs_current_density_and_voltage(file_path):
    log_abs_current_density, voltage = [], []
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

# Iterate over the files in the datasets
for dataset_name, dataset_path in paths.items():
    for root, dirs, files in os.walk(dataset_path):
        for file_name in files:
            if file_name.endswith("_alldata.txt") and ".txt_metadata" not in file_name:
                file_path = os.path.join(root, file_name)
                season, electrode, user = extract_info_from_file(file_path)
                log_abs_current_density, voltage = extract_log_abs_current_density_and_voltage(file_path)
                
                if season and electrode and user:
                    if electrode in data_dict and season in data_dict[electrode]:
                        data_length = min(len(log_abs_current_density), len(voltage))
                        data_dict[electrode][season]['log_abs_current_density'].extend(log_abs_current_density[:data_length])
                        data_dict[electrode][season]['voltage'].extend(voltage[:data_length])
                        data_dict[electrode][season]['file_names'].extend([file_name] * data_length)
                        data_dict[electrode][season]['users'].extend([user] * data_length)
                        
                        # Count the number of shorts per file for each user
                        if any(value > 0 for value in log_abs_current_density):
                            data_dict[electrode][season]['files_with_shorts'].add((file_name, user))

# Update num_shorts based on unique files with shorts
for electrode, season_data in data_dict.items():
    for season, values in season_data.items():
        values['num_shorts'] = len(values['files_with_shorts'])

# Calculate short ratio for each combination
for electrode, season_data in data_dict.items():
    for season, values in season_data.items():
        unique_file_names = len(set(values['file_names']))
        if unique_file_names > 0:
            values['short_ratio'] = (values['num_shorts'] / unique_file_names) * 100
        else:
            values['short_ratio'] = 0

# Create dataframes and write to Excel
output_path = r'C:\Users\Owner\OneDrive\Desktop\sync pending\log_abs_current_density_voltage_seasons.xlsx'
with pd.ExcelWriter(output_path) as writer:
    for electrode, season_data in data_dict.items():
        for season, values in season_data.items():
            data_length = min(len(values['log_abs_current_density']), len(values['voltage']), len(values['file_names']), len(values['users']))
            df = pd.DataFrame({
                'Voltage': values['voltage'][:data_length],
                'Log(Abs Current Density)': values['log_abs_current_density'][:data_length],
                'File Names': values['file_names'][:data_length],
                'User': values['users'][:data_length]
            })
            df['Shorts'] = [values['num_shorts']] * data_length  # Add the shorts count as a single integer
            df['Short Ratio'] = [values['short_ratio']] * data_length
            df.to_excel(writer, sheet_name=f'{electrode}_{season}', index=False)

# Define the function to plot heatmaps
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
    plt.text(xedges[-1], ylim_min, f"Shorted Attempts: {num_shorts}", color='yellow', fontsize=20, ha='right', va='bottom', weight='bold')
    plt.text(xedges[0], ylim_min, f"Short Ratio (%): {short_ratio:.2f}", color='yellow', fontsize=20, ha='left', va='bottom', weight='bold')
    plt.title(title, fontsize=24, weight='bold')  # Increase font size and make it bold
    plt.xlabel('Voltage (V)', fontsize=21, weight='bold')  # Increase font size and make it bold
    plt.ylabel('Log Abs Current Density', fontsize=21, weight='bold')  # Increase font size and make it bold
    plt.xticks(fontsize=18, weight='bold')  # Increase font size and make it bold
    plt.yticks(fontsize=18, weight='bold')  # Increase font size and make it bold
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Save heatmap image
    heatmap_image_path = r'C:\Users\Owner\OneDrive\Desktop/User Seasonal Final'
    plt.savefig(os.path.join(heatmap_image_path, f'{title}.png'))
    plt.close()

# Plot heatmaps for each combination
for electrode, season_data in data_dict.items():
    for season, values in season_data.items():
        title = f'{electrode}_{season}'
        plot_heatmap(values['voltage'], values['log_abs_current_density'], title, values['num_shorts'], values['short_ratio'])

print("Processing and heatmap generation completed.")

# Define paths
paths = {
    'Au Dataset 1': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\Combined Au\Dataset 1',
    'Au Dataset 2': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\Combined Au\Dataset 2',
    'Ag Dataset 1': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\Combined Ag\Dataset 1',
    'Ag Dataset 2': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\Combined Ag\Dataset 2',
}

# Initialize the data dictionary
data_dict = {
    'Au': {user: {season: {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'short_ratio': 0, 'avg_log_density': 0, 'total_files': 0} 
                   for season in ['Fall', 'Spring', 'Summer', 'Winter']} 
           for user in ['User 1', 'User 2']},
    'Ag': {user: {season: {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'short_ratio': 0, 'avg_log_density': 0, 'total_files': 0} 
                   for season in ['Fall', 'Spring', 'Summer', 'Winter']} 
           for user in ['User 1', 'User 2']},
}

# Function to extract user and season information from file name
def extract_user_and_season(file_path):
    user = None
    season = None
    electrode = None
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if 'User' in line:
                    user = line.split(': ')[1].strip()
                elif 'Season' in line:
                    season = line.split(': ')[1].strip()
                elif 'Electrode' in line:
                    electrode = line.split(': ')[1].strip()
    except Exception as e:
        print(f"Error extracting user and season from file {file_path}: {e}")
    return user, season, electrode

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

# Process each file
for dataset_name, dataset_path in paths.items():
    for root, dirs, files in os.walk(dataset_path):
        for file_name in files:
            if file_name.endswith("_alldata.txt") and ".txt_metadata" not in file_name:
                file_path = os.path.join(root, file_name)
                user, season, electrode = extract_user_and_season(file_path)
                log_abs_current_density, voltage = extract_log_abs_current_density_and_voltage(file_path)
                if user and season and electrode:
                    if electrode in ['Au', 'Ag'] and user in ['User 1', 'User 2']:
                        if season in ['Fall', 'Spring', 'Summer', 'Winter']:
                            data_dict[electrode][user][season]['log_abs_current_density'].extend(log_abs_current_density)
                            data_dict[electrode][user][season]['voltage'].extend(voltage[:len(log_abs_current_density)])
                            data_dict[electrode][user][season]['file_names'].extend([file_name] * len(log_abs_current_density))
                            data_dict[electrode][user][season]['total_files'] += 1
                            if any(value > 0 for value in log_abs_current_density):
                                data_dict[electrode][user][season]['num_shorts'] += 1
                        else:
                            print(f"Skipping file {file_path} with unknown season {season}")
                    else:
                        print(f"Skipping file {file_path} with unknown electrode {electrode} or user {user}")

# Calculate the short ratio and average log density for each user-season-electrode combination
for electrode, user_data in data_dict.items():
    for user, season_data in user_data.items():
        for season, values in season_data.items():
            if values['total_files'] > 0:
                values['short_ratio'] = (values['num_shorts'] / values['total_files']) * 100
            else:
                values['short_ratio'] = 0
            if values['log_abs_current_density']:
                values['avg_log_density'] = np.mean(values['log_abs_current_density'])
            else:
                values['avg_log_density'] = 0
            # Adjust the number of shorts for the specific condition
            if electrode == 'Au' and user == 'User 1' and season == 'Spring':
                values['num_shorts'] -= 1

# Define output Excel file path for the main data
output_path = r'C:\Users\Owner\OneDrive\Desktop\sync pending\log_abs_current_density_voltage_user_season.xlsx'

# Write the main data to the output Excel file
with pd.ExcelWriter(output_path) as writer:
    for electrode, user_data in data_dict.items():
        for user, season_data in user_data.items():
            for season, values in season_data.items():
                df = pd.DataFrame({
                    'Log(Abs Current Density)': values['log_abs_current_density'],
                    'Voltage': values['voltage'],
                    'File Names': values['file_names'],
                    'User': [user] * len(values['log_abs_current_density']),
                    'Season': [season] * len(values['log_abs_current_density']),
                    'Num Shorts': [values['num_shorts']] * len(values['log_abs_current_density']),
                    'Short Ratio': [values['short_ratio']] * len(values['log_abs_current_density']),
                    'Average Log Density': [values['avg_log_density']] * len(values['log_abs_current_density']),
                    'Total Files': [values['total_files']] * len(values['log_abs_current_density'])
                })
                if not df.empty:
                    df.to_excel(writer, sheet_name=f'{electrode}_{user}_{season}', index=False)

# Define the path to the existing Excel file
input_excel_path = r'C:\Users\Owner\OneDrive\Desktop\sync pending\log_abs_current_density_voltage_user_season.xlsx'
output_excel_path = r'C:\Users\Owner\OneDrive\Desktop\sync pending\average_log_abs_current_density_per_file_user_season.xlsx'

# Initialize the dictionary to store average log density
average_log_density_dict = {}

# Read the existing Excel file
with pd.ExcelFile(input_excel_path) as xls:
    sheet_names = xls.sheet_names
    
    for sheet_name in sheet_names:
        # Extract data from each sheet
        df = pd.read_excel(xls, sheet_name=sheet_name)
        
        # Calculate the average log density per file and user
        avg_log_density = df.groupby(['File Names', 'User'])['Log(Abs Current Density)'].mean().reset_index()
        
        # Store the results in the dictionary
        average_log_density_dict[sheet_name] = avg_log_density

# Write averages to a new Excel sheet
with pd.ExcelWriter(output_excel_path) as writer:
    for sheet_name, avg_df in average_log_density_dict.items():
        avg_df.to_excel(writer, sheet_name=sheet_name, index=False)

print("Average log abs current density per file and user has been calculated and written to Excel.")
def plot_heatmap(voltage, log_abs_current_density, title, electrode, user, season, vmin=0, vmax=250, ylim_min=-12, ylim_max=6, **kwargs):
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
    num_shorts = kwargs.get('num_shorts', 0)
    short_ratio = kwargs.get('short_ratio', 0)
    plt.text(xedges[-1], ylim_min, f"Shorted Attempts: {num_shorts}", color='yellow', fontsize=20, ha='right', va='bottom', weight='bold')
    plt.text(xedges[0], ylim_min, f"Short Ratio (%): {short_ratio:.2f}", color='yellow', fontsize=20, ha='left', va='bottom', weight='bold')
    
    plt.title(title, fontsize=24, weight='bold')  # Increase font size and make it bold
    plt.xlabel('Voltage (V)', fontsize=21, weight='bold')  # Increase font size and make it bold
    plt.ylabel('Log Abs Current Density', fontsize=21, weight='bold')  # Increase font size and make it bold
    plt.xticks(fontsize=18, weight='bold')  # Change the font size and make it bold
    plt.yticks(fontsize=18, weight='bold')  # Change the font size and make it bold
    
    plt.tight_layout()
    
    # Define the output path for the heatmap, including the filename
    heatmap_dir = r'C:\Users\Owner\OneDrive\Desktop\User Seasonal Final New'
    if not os.path.exists(heatmap_dir):
        os.makedirs(heatmap_dir)
    
    heatmap_filename = f'{electrode}_{user}_{season}.png'
    heatmap_path = os.path.join(heatmap_dir, heatmap_filename)
    
    plt.savefig(heatmap_path)
    plt.close()
    print(f"Heatmap saved at {heatmap_path}")

# Generate heatmaps
for electrode, user_data in data_dict.items():
    for user, season_data in user_data.items():
        for season, values in season_data.items():
            if values['log_abs_current_density']:
                title = f'{electrode} - {user} - {season}'
                plot_heatmap(values['voltage'], values['log_abs_current_density'], title, electrode, user, season, num_shorts=values['num_shorts'], short_ratio=values['short_ratio'])
paths = {
    'Au Dataset 1': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\Combined Au\Dataset 1',
    'Au Dataset 2': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\Combined Au\Dataset 2',
    'Ag Dataset 1': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\Combined Ag\Dataset 1',
    'Ag Dataset 2': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\Combined Ag\Dataset 2',
}

# Initialize the data dictionary
data_dict = {
    'Au': {user: {season: {carbon: {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'short_ratio': 0, 'avg_log_density': 0, 'total_files': 0} 
                            for carbon in range(10, 17)} for season in ['Fall', 'Spring', 'Summer', 'Winter']} for user in ['User 1', 'User 2']},
    'Ag': {user: {season: {carbon: {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'short_ratio': 0, 'avg_log_density': 0, 'total_files': 0} 
                            for carbon in range(10, 17)} for season in ['Fall', 'Spring', 'Summer', 'Winter']} for user in ['User 1', 'User 2']},
}

# Function to extract metadata including carbon number from file name
def extract_metadata(file_path):
    user = None
    season = None
    electrode = None
    carbon = None
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if 'User' in line:
                    user = line.split(': ')[1].strip()
                elif 'Season' in line:
                    season = line.split(': ')[1].strip()
                elif 'Electrode' in line:
                    electrode = line.split(': ')[1].strip()
                elif 'Carbon' in line:
                    carbon = int(line.split(': ')[1].strip())
                
    except Exception as e:
        print(f"Error extracting metadata from file {file_path}: {e}")
    return user, season, electrode, carbon

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

# Process each file
for dataset_name, dataset_path in paths.items():
    for root, dirs, files in os.walk(dataset_path):
        for file_name in files:
            if file_name.endswith("_alldata.txt") and ".txt_metadata" not in file_name:
                file_path = os.path.join(root, file_name)
                user, season, electrode, carbon = extract_metadata(file_path)
                log_abs_current_density, voltage = extract_log_abs_current_density_and_voltage(file_path)
                if user and season and electrode and carbon:
                    if electrode in ['Au', 'Ag'] and user in ['User 1', 'User 2']:
                        if season in ['Fall', 'Spring', 'Summer', 'Winter'] and carbon in range(10, 17):
                            data_dict[electrode][user][season][carbon]['log_abs_current_density'].extend(log_abs_current_density)
                            data_dict[electrode][user][season][carbon]['voltage'].extend(voltage[:len(log_abs_current_density)])
                            data_dict[electrode][user][season][carbon]['file_names'].extend([file_name] * len(log_abs_current_density))
                            data_dict[electrode][user][season][carbon]['total_files'] += 1
                            if any(value > 0 for value in log_abs_current_density):
                                data_dict[electrode][user][season][carbon]['num_shorts'] += 1
                        else:
                            print(f"Skipping file {file_path} with unknown season {season} or carbon {carbon}")
                    else:
                        print(f"Skipping file {file_path} with unknown electrode {electrode} or user {user}")

# Calculate the short ratio and average log density for each user-season-electrode-carbon combination
for electrode, user_data in data_dict.items():
    for user, season_data in user_data.items():
        for season, carbon_data in season_data.items():
            for carbon, values in carbon_data.items():
                if values['total_files'] > 0:
                    values['short_ratio'] = (values['num_shorts'] / values['total_files']) * 100
                else:
                    values['short_ratio'] = 0
                if values['log_abs_current_density']:
                    values['avg_log_density'] = np.mean(values['log_abs_current_density'])
                else:
                    values['avg_log_density'] = 0

# Define output Excel file path
output_path = 'C:\\Users\\Owner\\OneDrive\\Desktop\\sync pending\\log_abs_current_density_voltage_user_season_carbon.xlsx'

# Create dataframes and write to Excel
with pd.ExcelWriter(output_path) as writer:
    for electrode, user_data in data_dict.items():
        for user, season_data in user_data.items():
            for season, carbon_data in season_data.items():
                for carbon, values in carbon_data.items():
                    data_length = min(len(values['log_abs_current_density']), len(values['voltage']), len(values['file_names']))
                    df = pd.DataFrame({
                        'Voltage': values['voltage'][:data_length],
                        'Log(Abs Current Density)': values['log_abs_current_density'][:data_length],
                        'File Names': values['file_names'][:data_length],
                        'Shorts': [values['num_shorts']] * data_length,
                        'Short Ratio': [values['short_ratio']] * data_length,
                        'Avg Log Density': [values['avg_log_density']] * data_length
                    })
                    sheet_name = f'{electrode}_{user}_{season}_C{carbon}'
                    df.to_excel(writer, sheet_name=sheet_name, index=False)

print("Processing and file creation completed.")
# Define the path to the existing Excel file
input_excel_path = r'C:\Users\Owner\OneDrive\Desktop\sync pending\log_abs_current_density_voltage_user_season_carbon.xlsx'
output_excel_path = r'C:\Users\Owner\OneDrive\Desktop\sync pending\average_log_abs_current_density_per_file_user_season_carbon.xlsx'

# Initialize the dictionary to store average log density
average_log_density_dict = {}

# Read the existing Excel file
with pd.ExcelFile(input_excel_path) as xls:
    sheet_names = xls.sheet_names
    
    for sheet_name in sheet_names:
        # Extract data from each sheet
        df = pd.read_excel(xls, sheet_name=sheet_name)
        
        # Calculate the average log density per file
        avg_log_density = df.groupby(['File Names'])['Log(Abs Current Density)'].mean().reset_index()
        
        # Store the results in the dictionary
        average_log_density_dict[sheet_name] = avg_log_density

# Write averages to a new Excel sheet
with pd.ExcelWriter(output_excel_path) as writer:
    for sheet_name, avg_df in average_log_density_dict.items():
        avg_df.to_excel(writer, sheet_name=sheet_name, index=False)

print("Average log abs current density per file has been calculated and written to Excel.")
# Define the plot_heatmap function
def plot_heatmap(voltage, log_abs_current_density, title, electrode, user, season, carbon, vmin=0, vmax=250, ylim_min=-12, ylim_max=6, **kwargs):
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
    num_shorts = kwargs.get('num_shorts', 0)
    short_ratio = kwargs.get('short_ratio', 0)
    plt.text(xedges[-1], ylim_min, f"Shorted Attempts: {num_shorts}", color='yellow', fontsize=20, ha='right', va='bottom', weight='bold')
    plt.text(xedges[0], ylim_min, f"Short Ratio (%): {short_ratio:.2f}", color='yellow', fontsize=20, ha='left', va='bottom', weight='bold')
    
    plt.xlabel('Voltage', fontsize=24, weight='bold')
    plt.ylabel('Log(Abs Current Density)', fontsize=24, weight='bold')
    plt.title(f'{title} Heatmap\nElectrode: {electrode} | User: {user} | Season: {season} | Carbon: {carbon}', fontsize=26, weight='bold')
    plt.xticks(fontsize=20, weight='bold')
    plt.yticks(fontsize=20, weight='bold')
    plt.grid(True)
    
    # Save the plot to file
    plot_filename = f'C:\\Users\\Owner\\OneDrive\\Desktop\\User Seasonal Final New\\{title}_Heatmap.png'
    plt.savefig(plot_filename)
    plt.close()
    print(f'Heatmap saved to {plot_filename}')

# Generate heatmaps for each combination
for electrode, user_data in data_dict.items():
    for user, season_data in user_data.items():
        for season, carbon_data in season_data.items():
            for carbon, values in carbon_data.items():
                title = f'{electrode}_{user}_{season}_C{carbon}'
                plot_heatmap(values['voltage'], values['log_abs_current_density'], title, electrode, user, season, carbon, num_shorts=values['num_shorts'], short_ratio=values['short_ratio'])
# Define paths for the datasets
paths = {
    'Au Dataset 1': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\Combined Au\Dataset 1',
    'Au Dataset 2': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\Combined Au\Dataset 2',
    'Ag Dataset 1': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\Combined Ag\Dataset 1',
    'Ag Dataset 2': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\Combined Ag\Dataset 2',
}

# Create dictionaries to store log abs current density, voltage values, and file names
data_dict = {
    'Au': {season: {time: {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'users': [], 'short_ratio': 0, 'files_with_shorts': set()} for time in ['Morning', 'Afternoon', 'Evening', 'Night']} for season in ['Winter', 'Spring', 'Summer', 'Fall']},
    'Ag': {season: {time: {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'users': [], 'short_ratio': 0, 'files_with_shorts': set()} for time in ['Morning', 'Afternoon', 'Evening', 'Night']} for season in ['Winter', 'Spring', 'Summer', 'Fall']},
}

# Function to extract season, time, electrode, and user information from file
def extract_info_from_file(file_path):
    season, time, electrode, user = None, None, None, None
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if 'Season' in line:
                    season = line.split(': ')[1].strip()
                elif 'Time' in line:
                    time = line.split(': ')[1].strip()
                elif 'Electrode' in line:
                    electrode = line.split(': ')[1].strip()
                elif 'User' in line:
                    user = line.split(': ')[1].strip()
    except Exception as e:
        print(f"Error extracting information from file {file_path}: {e}")
    return season, time, electrode, user

# Function to extract log abs current density and voltage from the file
def extract_log_abs_current_density_and_voltage(file_path):
    log_abs_current_density, voltage = [], []
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

# Iterate over the files in the datasets
for dataset_name, dataset_path in paths.items():
    for root, dirs, files in os.walk(dataset_path):
        for file_name in files:
            if file_name.endswith("_alldata.txt") and ".txt_metadata" not in file_name:
                file_path = os.path.join(root, file_name)
                season, time, electrode, user = extract_info_from_file(file_path)
                log_abs_current_density, voltage = extract_log_abs_current_density_and_voltage(file_path)
                
                if season and time and electrode and user:
                    if electrode in data_dict and season in data_dict[electrode]:
                        if time in data_dict[electrode][season]:
                            data_length = min(len(log_abs_current_density), len(voltage))
                            data_dict[electrode][season][time]['log_abs_current_density'].extend(log_abs_current_density[:data_length])
                            data_dict[electrode][season][time]['voltage'].extend(voltage[:data_length])
                            data_dict[electrode][season][time]['file_names'].extend([file_name] * data_length)
                            data_dict[electrode][season][time]['users'].extend([user] * data_length)
                            
                            # Count the number of shorts per file for each user
                            if any(value > 0 for value in log_abs_current_density):
                                data_dict[electrode][season][time]['files_with_shorts'].add((file_name, user))

# Update num_shorts based on unique files with shorts
for electrode, season_data in data_dict.items():
    for season, time_data in season_data.items():
        for time, values in time_data.items():
            values['num_shorts'] = len(values['files_with_shorts'])

# Calculate short ratio for each combination
for electrode, season_data in data_dict.items():
    for season, time_data in season_data.items():
        for time, values in time_data.items():
            unique_file_names = len(set(values['file_names']))
            if unique_file_names > 0:
                values['short_ratio'] = (values['num_shorts'] / unique_file_names) * 100
            else:
                values['short_ratio'] = 0

# Create a single dataframe to store the average current density, number of shorts, and short ratio
results = []

for electrode, season_data in data_dict.items():
    for season, time_data in season_data.items():
        for time, values in time_data.items():
            log_density_array = np.array(values['log_abs_current_density'])
            if len(log_density_array) > 0:
                avg_current_density = np.mean(log_density_array)
            else:
                avg_current_density = 0

            results.append({
                'Electrode': electrode,
                'Season': season,
                'Time': time,
                'Average Current Density': avg_current_density,
                'Number of Shorts': values['num_shorts'],
                'Short Ratio (%)': values['short_ratio']
            })

# Create a DataFrame and save it to Excel
results_df = pd.DataFrame(results)
output_path = r'C:\Users\Owner\OneDrive\Desktop\sync pending\seasons_time_summary_statistics.xlsx'
results_df.to_excel(output_path, index=False)

# Define paths for the datasets
paths = {
    'Au Dataset 1': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\Combined Au\Dataset 1',
    'Au Dataset 2': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\Combined Au\Dataset 2',
    'Ag Dataset 1': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\Combined Ag\Dataset 1',
    'Ag Dataset 2': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\Combined Ag\Dataset 2',
}

# Initialize the data dictionary
data_dict = {
    'Au': {user: {season: {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'files_with_shorts': set(), 'num_shorts': 0, 'short_ratio': 0, 'avg_log_density': 0, 'total_files': 0} for season in ['Fall', 'Spring', 'Summer', 'Winter']} for user in ['User 1', 'User 2']},
    'Ag': {user: {season: {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'files_with_shorts': set(), 'num_shorts': 0, 'short_ratio': 0, 'avg_log_density': 0, 'total_files': 0} for season in ['Fall', 'Spring', 'Summer', 'Winter']} for user in ['User 1', 'User 2']},
}

# Function to extract user and season information from file name
def extract_user_and_season(file_path):
    user = None
    season = None
    electrode = None
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if 'User' in line:
                    user = line.split(': ')[1].strip()
                elif 'Season' in line:
                    season = line.split(': ')[1].strip()
                elif 'Electrode' in line:
                    electrode = line.split(': ')[1].strip()
    except Exception as e:
        print(f"Error extracting user and season from file {file_path}: {e}")
    return user, season, electrode

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

# Process each file
for dataset_name, dataset_path in paths.items():
    for root, dirs, files in os.walk(dataset_path):
        for file_name in files:
            if file_name.endswith("_alldata.txt") and ".txt_metadata" not in file_name:
                file_path = os.path.join(root, file_name)
                user, season, electrode = extract_user_and_season(file_path)
                log_abs_current_density, voltage = extract_log_abs_current_density_and_voltage(file_path)
                if user and season and electrode:
                    if electrode in ['Au', 'Ag'] and user in ['User 1', 'User 2']:
                        if season in ['Fall', 'Spring', 'Summer', 'Winter']:
                            data_dict[electrode][user][season]['log_abs_current_density'].extend(log_abs_current_density)
                            data_dict[electrode][user][season]['voltage'].extend(voltage[:len(log_abs_current_density)])
                            data_dict[electrode][user][season]['file_names'].extend([file_name] * len(log_abs_current_density))
                            if any(value > 0 for value in log_abs_current_density):
                                data_dict[electrode][user][season]['files_with_shorts'].add(file_name)  # Track files with shorts
                            data_dict[electrode][user][season]['total_files'] += 1
                        else:
                            print(f"Skipping file {file_path} with unknown season {season}")
                    else:
                        print(f"Skipping file {file_path} with unknown electrode {electrode} or user {user}")

# Update num_shorts based on unique files with shorts
for electrode, user_data in data_dict.items():
    for user, season_data in user_data.items():
        for season, values in season_data.items():
            values['num_shorts'] = len(values['files_with_shorts'])

# Calculate short ratio for each combination
for electrode, user_data in data_dict.items():
    for user, season_data in user_data.items():
        for season, values in season_data.items():
            unique_file_names = len(set(values['file_names']))
            if unique_file_names > 0:
                values['short_ratio'] = (values['num_shorts'] / unique_file_names) * 100
            else:
                values['short_ratio'] = 0

# Create a single dataframe to store the average current density, number of shorts, and short ratio
results = []

for electrode, user_data in data_dict.items():
    for user, season_data in user_data.items():
        for season, values in season_data.items():
            log_density_array = np.array(values['log_abs_current_density'])
            if len(log_density_array) > 0:
                avg_current_density = np.mean(log_density_array)
            else:
                avg_current_density = 0

            results.append({
                'Electrode': electrode,
                'User': user,
                'Season': season,
                'Average Current Density': avg_current_density,
                'Number of Shorts': values['num_shorts'],
                'Short Ratio (%)': values['short_ratio']
            })

# Create a DataFrame and save it to Excel
results_df = pd.DataFrame(results)
output_path = r'C:\Users\Owner\OneDrive\Desktop\sync pending\user_seasons_time_summary_statistics.xlsx'
results_df.to_excel(output_path, index=False)

print("Summary statistics have been saved to Excel.")

import os
import pandas as pd
from datetime import datetime



paths = {
    'Au Dataset 1': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\Combined Au\Dataset 1',
    'Au Dataset 2': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\Combined Au\Dataset 2',
    'Ag Dataset 1': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\Combined Ag\Dataset 1',
    'Ag Dataset 2': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\Combined Ag\Dataset 2',
}

# Function to extract metadata from the file
def extract_metadata(file_path):
    date, time_info, location, carbon_number, electrode = None, None, None, None, None
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if 'Date' in line:
                    date = line.split(': ')[1].strip()
                elif 'Time' in line:
                    time_info_line = line.split(': ')[1].strip()
                    if 'PM' in time_info_line or 'AM' in time_info_line:
                        time_info = time_info_line.split(',')[0].strip()
                elif 'Location' in line:
                    location = line.split(': ')[1].strip()
                elif 'Carbon Number' in line:
                    carbon_number = line.split(': ')[1].strip()
                elif 'Electrode' in line:
                    electrode = line.split(': ')[1].strip()
    except Exception as e:
        print(f"Error extracting metadata from file {file_path}: {e}")
    return date, time_info, location, carbon_number, electrode

# Function to extract log abs current density and voltage from the file
def extract_log_abs_current_density_and_voltage(file_path):
    log_abs_current_density, voltage = [], []
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

# Function to convert time to the nearest hour in the desired format
def convert_to_nearest_hour(time_info):
    try:
        time_obj = datetime.strptime(time_info, "%I:%M %p")
        time_obj = time_obj.replace(minute=0)
        return time_obj.strftime("%I:%M %p")
    except Exception as e:
        print(f"Error converting time {time_info} to hour: {e}")
        return None

# Create a DataFrame to store the extracted data
data = []

# Process each file in the datasets
for dataset_name, dataset_path in paths.items():
    for root, dirs, files in os.walk(dataset_path):
        for file_name in files:
            if file_name.endswith("_alldata.txt") and ".txt_metadata" not in file_name:
                file_path = os.path.join(root, file_name)
                date, time_info, location, carbon_number, electrode = extract_metadata(file_path)
                
                # Skip rows with None values for Carbon Number or Electrode
                if carbon_number is None or electrode is None:
                    continue
                
                log_abs_current_density, voltage = extract_log_abs_current_density_and_voltage(file_path)
                
                # Extract hour from time_info
                hour = convert_to_nearest_hour(time_info) if time_info else None
                
                # Append each density and voltage value to the DataFrame
                for density, volt in zip(log_abs_current_density, voltage):
                    data.append([date, time_info, hour, location, carbon_number, electrode, density, volt])

# Create a DataFrame from the extracted data
df = pd.DataFrame(data, columns=['Date', 'Time', 'Hour', 'Location', 'Carbon Number', 'Electrode', 'Log Abs Current Density', 'Voltage'])

# Add empty columns for Humidity and Temperature
df['Humidity'] = None
df['Temperature'] = None

# Save the DataFrame to an Excel file
excel_file_path = r'C:\Users\Owner\OneDrive\Desktop\sync pending\updated_weather_date_hour_location_new.xlsx'
df.to_excel(excel_file_path, index=False)

print(f"Data extracted and saved to {excel_file_path}. Please open the file and enter the humidity and temperature values for each combination.")

# Define file paths
excel_file_path = r'C:\Users\Owner\OneDrive\Desktop\sync pending\updated_weather_date_hour_location_new.xlsx'
csv_file_path = r'C:\Users\Owner\OneDrive\Desktop\user_input_data.csv'

# Load the original data
df = pd.read_excel(excel_file_path)

# Convert 'Carbon Number' column to string to ensure consistency
df['Carbon Number'] = df['Carbon Number'].astype(str)

# Create or open CSV file for writing
if not os.path.isfile(csv_file_path):
    # Create CSV file with headers if it does not exist
    pd.DataFrame(columns=['Date', 'Location', 'Carbon Number', 'Electrode', 'Hour', 'Humidity', 'Temperature']).to_csv(csv_file_path, index=False)

# Load existing CSV data
existing_data = pd.read_csv(csv_file_path)

# Convert existing data columns to match the format of the main DataFrame
existing_data['Carbon Number'] = existing_data['Carbon Number'].astype(str)

# Combine the DataFrames and identify unique combinations
combined_df = pd.concat([df, existing_data], ignore_index=True)
unique_combos_df = combined_df.drop_duplicates(subset=['Date', 'Location', 'Carbon Number', 'Electrode', 'Hour'])

# Find unmatched combinations
unmatched_df = unique_combos_df[unique_combos_df[['Date', 'Location', 'Carbon Number', 'Electrode', 'Hour']].duplicated(keep=False) == False]

# Filter out combinations already entered
unmatched_combinations = unmatched_df[~unmatched_df[['Date', 'Location', 'Carbon Number', 'Electrode', 'Hour']].apply(tuple, 1).isin(existing_data[['Date', 'Location', 'Carbon Number', 'Electrode', 'Hour']].apply(tuple, 1))]

if not unmatched_combinations.empty:
    print(f"\n{len(unmatched_combinations)} unique combinations are missing humidity and temperature values.")

    def prompt_for_values(df):
        with open(csv_file_path, 'a') as csv_file:
            for index, row in df.iterrows():
                print(f"\nCombination missing data:")
                print(f"Date: {row['Date']}, Location: {row['Location']}, Carbon Number: {row['Carbon Number']}, Electrode: {row['Electrode']}, Hour: {row['Hour']}")
                
                # Input humidity and temperature
                humidity = input("Enter humidity (%): ")
                temperature = input("Enter temperature (°C): ")
                
                # Convert inputs to numeric types and handle errors
                try:
                    humidity = pd.to_numeric(humidity, errors='coerce')
                    temperature = pd.to_numeric(temperature, errors='coerce')
                    
                    if pd.isna(humidity) or pd.isna(temperature):
                        print("Invalid input. Please enter numeric values.")
                        continue
                    
                    # Update DataFrame
                    df.at[index, 'Humidity'] = humidity
                    df.at[index, 'Temperature'] = temperature
                    
                    # Write to CSV file
                    new_row = {
                        'Date': row['Date'],
                        'Location': row['Location'],
                        'Carbon Number': row['Carbon Number'],
                        'Electrode': row['Electrode'],
                        'Hour': row['Hour'],
                        'Humidity': humidity,
                        'Temperature': temperature
                    }
                    pd.DataFrame([new_row]).to_csv(csv_file, mode='a', header=False, index=False)
                except Exception as e:
                    print(f"Error: {e}")

    prompt_for_values(unmatched_combinations)

    # Save the updated DataFrame back to a new sheet in the existing Excel file
    with pd.ExcelWriter(excel_file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df.to_excel(writer, sheet_name='Updated Data', index=False)

    print(f"Updated data with humidity and temperature values appended to the existing sheet in {excel_file_path}.")
else:
    print("No unique combinations were found that need input.")

print(f"User inputs have been saved to {csv_file_path}.")


# Define file paths
csv_file_path = r'C:\Users\Owner\OneDrive\Desktop\user_input_data.csv'

# Load the original data
df = pd.read_excel(excel_file_path)

# Convert 'Carbon Number' column to string to ensure consistency
df['Carbon Number'] = df['Carbon Number'].astype(str)

# Load existing CSV data
existing_data = pd.read_csv(csv_file_path)

# Convert existing data columns to match the format of the main DataFrame
existing_data['Carbon Number'] = existing_data['Carbon Number'].astype(str)

# Merge the existing data with the main DataFrame on the key columns
merged_df = pd.merge(df, existing_data, on=['Date', 'Location', 'Carbon Number', 'Electrode', 'Hour'], how='left', suffixes=('', '_y'))

# Fill in the missing humidity and temperature values from the CSV data
merged_df['Humidity'].fillna(merged_df['Humidity_y'], inplace=True)
merged_df['Temperature'].fillna(merged_df['Temperature_y'], inplace=True)

# Drop the redundant columns
merged_df.drop(columns=['Humidity_y', 'Temperature_y'], inplace=True)

# Save the updated DataFrame back to the Excel file
with pd.ExcelWriter(excel_file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    merged_df.to_excel(writer, sheet_name='Updated Data', index=False)

print(f"Updated data with humidity and temperature values appended to the existing sheet in {excel_file_path}.")