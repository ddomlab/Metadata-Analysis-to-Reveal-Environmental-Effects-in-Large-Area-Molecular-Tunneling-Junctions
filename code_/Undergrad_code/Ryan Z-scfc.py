# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 00:04:53 2024

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
import os
import re
import requests
import time
from datetime import datetime
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
   
    # Check for specific cases
    for part in parts:
        if 'Bad Data (Carbon Ten)' in part:
            return 10
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

def extract_terminal(file_path):
    if 'ferrocene' in file_path:
        return 'Fc-CN-SH'
    
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
            
            terminal = extract_terminal(file_path)
            if terminal == 'Unknown':
                print(f"File path: {file_path} - Unknown parameter: Terminal Group")
            
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
                'Terminal Group': terminal,
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
    'Dataset 1': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\Fc-Cn-SH',
}

folders_Data1 = ['sC6-ferrocene', 'sC7-ferrocene', 'sC8-ferrocene', 'sC9-ferrocene', 'sC11-ferrocene', 
                 'sC12-ferrocene', 'sC13-ferrocene', 'sC14-ferrocene', 'sC15-ferrocene']

# Pair metadata and data files
metadata_files_1, data_files_1, num_pairs_1 = pair_files(paths['Dataset 1'], folders_Data1)

print(f"\nNumber of pairs in Data1 dataset: {num_pairs_1}")


# Initialize data_dict with required keys and initial values for amides
data_dict = {
    'Fc-CN-SH': {f'C{carbon}': {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'short_ratio': 0, 'average_log_abs_current_density': 0} for carbon in range(6, 16)},
    
}

# Function to extract carbon number and ring information from file
def extract_carbon_number_and_terminal(file_path):
    terminal = None
    carbon = None
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if 'Carbon Number' in line:
                    carbon = line.split(': ')[1].strip()
                elif 'Terminal Group' in line:
                    terminal = line.split(': ')[1].strip()
    except Exception as e:
        print(f"Error extracting carbon number and ring from file {file_path}: {e}")
    return carbon, terminal

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
                carbon, terminal = extract_carbon_number_and_terminal(file_path)
                
                # Check if both carbon and ring are valid before accessing data_dict
                if carbon and terminal in data_dict:
                    carbon_key = f'C{carbon}'
                    if carbon_key in data_dict[terminal]:
                        log_abs_current_density, voltage = extract_log_abs_current_density_and_voltage(file_path)
                        data_dict[terminal][carbon_key]['log_abs_current_density'].extend(log_abs_current_density)
                        data_dict[terminal][carbon_key]['voltage'].extend(voltage)
                        data_dict[terminal][carbon_key]['file_names'].append(file_name)
                        if any(value > 0 for value in log_abs_current_density):
                            data_dict[terminal][carbon_key]['num_shorts'] += 1
                    else:
                        print(f"Skipping file {file_path} with unknown carbon {carbon} or ring {terminal}")

# Calculate the short ratio and average log abs current density for each amide-carbon combination
for terminal, carbon_data in data_dict.items():
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
output_path = 'C:\\Users\\Owner\\OneDrive\\Desktop\\log_abs_current_density_voltage_Fc_CN_SH.xlsx'

# Create dataframes and write to Excel
with pd.ExcelWriter(output_path) as writer:
    for terminal, carbon_data in data_dict.items():
        for carbon, values in carbon_data.items():
            df = pd.DataFrame({
                'Voltage': values['voltage'],
                'Log(Abs Current Density)': values['log_abs_current_density'],
                'Shorts': [values['num_shorts']] * len(values['log_abs_current_density']),
                'Short Ratio': [values['short_ratio']] * len(values['log_abs_current_density']),
                'Average Log(Abs Current Density)': [values['average_log_abs_current_density']] * len(values['log_abs_current_density'])
            })
            df.to_excel(writer, sheet_name=f'{terminal}_{carbon}', index=False)

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LinearSegmentedColormap

# Function to create a custom colormap from white to viridis
def white_to_viridis():
    viridis_cmap = plt.cm.get_cmap('viridis')
    newcolors = viridis_cmap(np.linspace(0, 1, 256))
    white = np.array([1, 1, 1, 1])  # RGBA values for white
    newcolors[:10, :] = white  # Set the first 10 colors to white
    newcmp = LinearSegmentedColormap.from_list("WhiteToViridis", newcolors)
    return newcmp

# Function to plot heatmap with white to viridis colormap and purple text for shorts
def plot_heatmap(voltage, log_abs_current_density, title, num_shorts, short_ratio, vmin=0, vmax=1500, ylim_min=-12, ylim_max=6, **kwargs):
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

    # Get the custom colormap
    cmap = white_to_viridis()

    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    im = plt.imshow(heatmap.T, origin='lower', cmap=cmap, aspect='auto', extent=[xedges[0], xedges[-1], ylim_min, ylim_max], vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im)

    # Set colorbar label properties
    cbar.set_label('Density', fontsize=21, weight='bold')
    cbar.ax.tick_params(labelsize=18)  # Change the font size of the colorbar scale
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight('bold')  # Make the color scale numbers bold

    # Display number of shorts in the bottom right corner
    plt.text(xedges[-1], ylim_min, f"Shorts: {num_shorts}", color='purple', fontsize=16, ha='right', va='bottom', weight='bold')
    plt.text(xedges[0], ylim_min, f"Short Ratio (%): {short_ratio:.2f}", color='purple', fontsize=16, ha='left', va='bottom', weight='bold')
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


def plot_derivative(voltage, log_abs_current_density, title):
    if len(voltage) < 2 or len(log_abs_current_density) < 2:
        print(f"Not enough data points for {title}")
        return
    
    d_voltage = [(voltage[i + 1] - voltage[i]) for i in range(len(voltage) - 1)]
    d_current_density = [(log_abs_current_density[i + 1] - log_abs_current_density[i]) for i in range(len(log_abs_current_density) - 1)]
    derivative = [d_cd / d_v if d_v != 0 else 0 for d_v, d_cd in zip(d_voltage, d_current_density)]
    
    if not derivative:
        print(f"Derivative calculation resulted in an empty list for {title}")
        return

    max_slope_index = derivative.index(max(derivative))
    voltage_midpoints = [(voltage[i] + voltage[i + 1]) / 2 for i in range(len(voltage) - 1)]
    data = pd.DataFrame({'Voltage': voltage_midpoints, 'Derivative': derivative})
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Voltage', y='Derivative', data=data, label='Derivative')
    plt.scatter(voltage_midpoints[max_slope_index], derivative[max_slope_index], color='red', label='Steepest Point')
    plt.title(title)
    plt.ylim(-100, 100)
    plt.xlabel('Voltage (V)')
    plt.ylabel('Derivative of Current Density')
    plt.legend()
    plt.show()
    print(f'The steepest point is at Voltage = {voltage[max_slope_index]:.2f} V with Current Density = {log_abs_current_density[max_slope_index]:.2e} A/cm²')


# Example plotting calls
for terminal, carbon_data in data_dict.items():
    for carbon, values in carbon_data.items():
        plot_heatmap(values['voltage'], values['log_abs_current_density'], f'{carbon} {terminal} Heatmap', num_shorts=values['num_shorts'], short_ratio=values['short_ratio'],image_path='C:\\Users\\Owner\\OneDrive\\Desktop\\Thuo Group Data\\Origin Data\\Heatmap Images\\Ferrocene')


# Define the paths and folders for AgTS and AuTi datasets
paths = {
    'Dataset 1': r'C:\Users\Owner\OneDrive\Desktop\Thuo Group Data\Origin Data\Fc-Cn-SH',
}

folders_Data1 = ['sC6-ferrocene', 'sC7-ferrocene', 'sC8-ferrocene', 'sC9-ferrocene', 'sC11-ferrocene', 
                 'sC12-ferrocene', 'sC13-ferrocene', 'sC14-ferrocene', 'sC15-ferrocene']

# Initialize data_dict with required keys and initial values for amides
data_dict = {
    'Fc-CN-SH': {f'C{carbon}': {'Morning': {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'short_ratio': 0, 'average_log_abs_current_density': 0},
                                'Afternoon': {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'short_ratio': 0, 'average_log_abs_current_density': 0},
                                'Evening': {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'short_ratio': 0, 'average_log_abs_current_density': 0},
                                'Dusk': {'log_abs_current_density': [], 'voltage': [], 'file_names': [], 'num_shorts': 0, 'short_ratio': 0, 'average_log_abs_current_density': 0}}
                 for carbon in range(6, 16)}
}

# Function to extract carbon number, terminal, and time of day from file
def extract_carbon_number_and_terminal(file_path):
    terminal = None
    carbon = None
    time_of_day = None
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if 'Carbon Number' in line:
                    carbon = line.split(': ')[1].strip()
                elif 'Terminal Group' in line:
                    terminal = line.split(': ')[1].strip()
                elif 'Time of Day' in line:
                    time_of_day = line.split(': ')[1].strip()
    except Exception as e:
        print(f"Error extracting carbon number, terminal, or time of day from file {file_path}: {e}")
    return carbon, terminal, time_of_day

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
                carbon, terminal, time_of_day = extract_carbon_number_and_terminal(file_path)
                
                # Check if both carbon and terminal are valid before accessing data_dict
                if carbon and terminal in data_dict:
                    carbon_key = f'C{carbon}'
                    if carbon_key in data_dict[terminal]:
                        log_abs_current_density, voltage = extract_log_abs_current_density_and_voltage(file_path)
                        if time_of_day in data_dict[terminal][carbon_key]:
                            data_dict[terminal][carbon_key][time_of_day]['log_abs_current_density'].extend(log_abs_current_density)
                            data_dict[terminal][carbon_key][time_of_day]['voltage'].extend(voltage)
                            data_dict[terminal][carbon_key][time_of_day]['file_names'].append(file_name)
                            if any(value > 0 for value in log_abs_current_density):
                                data_dict[terminal][carbon_key][time_of_day]['num_shorts'] += 1
                        else:
                            print(f"Skipping file {file_path} with unknown time of day {time_of_day}")
                    else:
                        print(f"Skipping file {file_path} with unknown carbon {carbon} or terminal {terminal}")

# Calculate the short ratio and average log abs current density for each amide-carbon-time_of_day combination
for terminal, carbon_data in data_dict.items():
    for carbon, time_data in carbon_data.items():
        for time_of_day, values in time_data.items():
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
output_path = 'C:\\Users\\Owner\\OneDrive\\Desktop\\log_abs_current_density_voltage_Fc_CN_SH.xlsx'

# Create dataframes and write to Excel
with pd.ExcelWriter(output_path) as writer:
    for terminal, carbon_data in data_dict.items():
        for carbon, time_data in carbon_data.items():
            for time_of_day, values in time_data.items():
                df = pd.DataFrame({
                    'Voltage': values['voltage'],
                    'Log(Abs Current Density)': values['log_abs_current_density'],
                    'Shorts': [values['num_shorts']] * len(values['log_abs_current_density']),
                    'Short Ratio': [values['short_ratio']] * len(values['log_abs_current_density']),
                    'Average Log(Abs Current Density)': [values['average_log_abs_current_density']] * len(values['log_abs_current_density']),
                    'Time of Day': [time_of_day] * len(values['log_abs_current_density'])
                })
                df.to_excel(writer, sheet_name=f'{terminal}_{carbon}_{time_of_day}', index=False)

def plot_heatmap(voltage, log_abs_current_density, title, num_shorts, short_ratio, time_of_day, vmin=0, vmax=500, ylim_min=-12, ylim_max=6, **kwargs):
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

# Example plotting calls
for terminal, carbon_data in data_dict.items():
    for carbon, time_data in carbon_data.items():
        for time_of_day, values in time_data.items():
            plot_heatmap(values['voltage'], values['log_abs_current_density'], f'{carbon} {terminal} {time_of_day} Heatmap', num_shorts=values['num_shorts'], short_ratio=values['short_ratio'], time_of_day=time_of_day, image_path='C:\\Users\\Owner\\OneDrive\\Desktop\\Thuo Group Data\\Origin Data\\Heatmap Images\\Ferrocene')
# Dictionary to store aggregated data
aggregate_data = {
    'Terminal': [],
    'Carbon': [],
    'Time of Day': [],
    'Average Log(Abs Current Density)': [],
    'Total Shorts': []
}

# Aggregate data for each combination
for terminal, carbon_data in data_dict.items():
    for carbon, time_data in carbon_data.items():
        for time_of_day, values in time_data.items():
            # Calculate average log abs current density
            if values['log_abs_current_density']:
                avg_log_abs_current_density = sum(values['log_abs_current_density']) / len(values['log_abs_current_density'])
            else:
                avg_log_abs_current_density = 0
            
            # Total number of shorts
            total_shorts = values['num_shorts']
            
            # Append data to aggregate dictionary
            aggregate_data['Terminal'].append(terminal)
            aggregate_data['Carbon'].append(carbon)
            aggregate_data['Time of Day'].append(time_of_day)
            aggregate_data['Average Log(Abs Current Density)'].append(avg_log_abs_current_density)
            aggregate_data['Total Shorts'].append(total_shorts)

# Convert aggregate_data to a DataFrame
aggregate_df = pd.DataFrame(aggregate_data)

# Define output Excel file path
output_path_aggregate = 'C:\\Users\\Owner\\OneDrive\\Desktop\\aggregate_data_log_abs_current_density.xlsx'

# Write aggregate data to Excel
aggregate_df.to_excel(output_path_aggregate, index=False)

print(f"Aggregate data saved to {output_path_aggregate}")


# Define path to the existing Excel sheet
existing_excel_path = 'C:\\Users\\Owner\\OneDrive\\Desktop\\aggregate_data_log_abs_current_density.xlsx'

# Read the existing Excel sheet into a DataFrame
existing_df = pd.read_excel(existing_excel_path)

# Group by 'Time of Day' and calculate averages and totals
aggregate_df = existing_df.groupby('Time of Day').agg({
    'Average Log(Abs Current Density)': 'mean',
    'Total Shorts': 'sum'
}).reset_index()

# Define output Excel file path for aggregated data
output_path_aggregate = 'C:\\Users\\Owner\\OneDrive\\Desktop\\aggregate_data_by_time_of_day.xlsx'

# Write aggregated data to Excel
aggregate_df.to_excel(output_path_aggregate, index=False)

print(f"Aggregate data by Time of Day saved to {output_path_aggregate}")
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
import os
import re
import requests
import time
from datetime import datetime
# Define the labels and data for summer and times of day
season_labels = ['Summer']
time_of_day_labels = ['Dusk', 'Morning', 'Afternoon', 'Evening']

# Manually enter the actual current density and shorts values for each combo
# Example data (replace with your actual values)
data = {
    'Summer Dusk': {
        'Average Log(Abs Current Density)': -3.11,
        'Total Shorts': 2
    },
    'Summer Morning': {
        'Average Log(Abs Current Density)': -0.56,
        'Total Shorts': 0
    },
    'Summer Afternoon': {
        'Average Log(Abs Current Density)': -4.25,
        'Total Shorts': 14
    },
    'Summer Evening': {
        'Average Log(Abs Current Density)': -4.32,
        'Total Shorts': 19
    }
}


# Extracting data for current density and shorts
current_density_values = [
    [data['Summer Dusk']['Average Log(Abs Current Density)'],
     data['Summer Morning']['Average Log(Abs Current Density)'],
     data['Summer Afternoon']['Average Log(Abs Current Density)'],
     data['Summer Evening']['Average Log(Abs Current Density)']
    ]
]

shorts_values = [
    [data['Summer Dusk']['Total Shorts'],
     data['Summer Morning']['Total Shorts'],
     data['Summer Afternoon']['Total Shorts'],
     data['Summer Evening']['Total Shorts']
    ]
]

# Plotting the Average Log(Abs Current Density) Heatmap
plt.figure(figsize=(10, 6))
heatmap = sns.heatmap(current_density_values, annot=True, cmap='viridis', fmt='.2f', linewidths=.5, cbar_kws={'label': 'Average Log(Abs Current Density)'})
plt.title('FC-CN-SH Time of Day Current Density Heatmap', fontsize=16, weight='bold')
plt.xlabel('Time of Day', fontsize=14, weight='bold')
plt.ylabel('Season', fontsize=14, weight='bold')
plt.xticks(np.arange(len(time_of_day_labels)) + 0.5, time_of_day_labels, fontsize=12, weight='bold')  # Adjust x-ticks positions and labels
plt.yticks(np.arange(len(season_labels)) + 0.5, season_labels, fontsize=12, weight='bold')  # Adjust y-ticks positions and labels

# Add exact values to heatmap cells
for i in range(len(season_labels)):
    for j in range(len(time_of_day_labels)):
        plt.text(j + 0.5, i + 0.5, f'{current_density_values[i][j]:.2f}', ha='center', va='center', color='red', fontsize=10, weight='bold')

plt.show()

# Plotting the Total Shorts Heatmap
plt.figure(figsize=(10, 6))
heatmap = sns.heatmap(shorts_values, annot=True, cmap='viridis', fmt='g', linewidths=.5, cbar_kws={'label': 'Total Shorts'})
plt.title('FC-CN-SH Time of Day Shorts Heatmap', fontsize=16, weight='bold')
plt.xlabel('Time of Day', fontsize=14, weight='bold')
plt.ylabel('Season', fontsize=14, weight='bold')
plt.xticks(np.arange(len(time_of_day_labels)) + 0.5, time_of_day_labels, fontsize=12, weight='bold')  # Adjust x-ticks positions and labels
plt.yticks(np.arange(len(season_labels)) + 0.5, season_labels, fontsize=12, weight='bold')  # Adjust y-ticks positions and labels

# Add exact values to heatmap cells
for i in range(len(season_labels)):
    for j in range(len(time_of_day_labels)):
        plt.text(j + 0.5, i + 0.5, f'{shorts_values[i][j]}', ha='center', va='center', color='red', fontsize=10, weight='bold')

plt.show()