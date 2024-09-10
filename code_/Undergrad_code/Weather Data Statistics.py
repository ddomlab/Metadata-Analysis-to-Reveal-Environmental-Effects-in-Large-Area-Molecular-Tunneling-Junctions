

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os



# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 20:57:27 2024

@author: Owner
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import re
import pandas as pd
import re
import pandas as pd
import re

# Load the data from the updated Excel file
file_path = r'C:\Users\Owner\OneDrive\Desktop\sync pending\updated_weather_date_hour_location_new.xlsx'
all_data = pd.read_excel(file_path, sheet_name=None)

# Combine all sheets into a single DataFrame
df_list = [df for df in all_data.values()]
combined_df = pd.concat(df_list)

# Convert hour from "8:00 PM" or "9:00 PM" to 24-hour format
def convert_hour(hour_str):
    match = re.match(r'(\d{1,2}):(\d{2})\s*(AM|PM)', hour_str)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2))
        am_pm = match.group(3)
        if am_pm == 'PM' and hour != 12:
            hour += 12
        elif am_pm == 'AM' and hour == 12:
            hour = 0
        return f'{hour:02d}:{minute:02d}'
    return hour_str

combined_df['Hour'] = combined_df['Hour'].astype(str).apply(convert_hour)

# Function to map date to season
def get_season(date_str):
    try:
        month = pd.to_datetime(date_str).month
    except Exception:
        return 'Unknown'  # Or np.nan to exclude such rows
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Autumn'

combined_df['Season'] = combined_df['Date'].apply(get_season)
combined_df = combined_df.dropna(subset=['Season'])  # Drop rows where season is unknown

# Group by unique combinations of date, hour, location, carbon number, and electrode
grouped = combined_df.groupby(['Date', 'Electrode', 'Hour', 'Carbon Number', 'Location', 'Humidity', 'Temperature'])

# Calculate the average and standard deviation of current density for each group
result = grouped['Log Abs Current Density'].agg(['mean', 'std']).reset_index()

# Rename columns for clarity
result = result.rename(columns={'mean': 'Avg Current Density', 'std': 'Std Dev'})

# Count the number of shorts per combination (shorts are defined as current density > 0)
result['Number of Shorts'] = grouped['Log Abs Current Density'].apply(lambda x: (x > 0).sum()).reset_index(drop=True)

# Determine if there is any current density above zero in the combination
result['Current Density Above Zero'] = grouped['Log Abs Current Density'].apply(lambda x: 'Yes' if (x > 0).any() else 'No').reset_index(drop=True)

# Output to Excel with the calculated columns
output_file_path = r'C:\Users\Owner\OneDrive\Desktop\train_GBR_output_carbon_number_with_season_final.xlsx'
result.to_excel(output_file_path, index=False)

print(f"Data has been successfully saved to {output_file_path}")

# Load the data
excel_file_path = r'C:\Users\Owner\OneDrive\Desktop\sync pending\updated_weather_date_hour_location_new.xlsx'
all_data = pd.read_excel(excel_file_path, sheet_name=None)

# Combine all sheets into a single DataFrame
data = []
for sheet_name, df in all_data.items():
    data.append(df)

# Concatenate all DataFrames into a single DataFrame
all_data = pd.concat(data, ignore_index=True)

# Ensure proper types for merge
all_data['Date'] = pd.to_datetime(all_data['Date'], errors='coerce')
all_data['Humidity'] = all_data['Humidity'].astype(float)
all_data['Temperature'] = all_data['Temperature'].astype(float)
all_data['Log Abs Current Density'] = all_data['Log Abs Current Density'].astype(float)

# Define the bins
humidity_bins = [20, 30, 40, 50, 60, 70, 80, 90, 100]
humidity_labels = ['21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']

temperature_bins = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
temperature_labels = ['11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']

# Bin the data
all_data['Humidity Bin'] = pd.cut(all_data['Humidity'], bins=humidity_bins, labels=humidity_labels, right=False)
all_data['Temperature Bin'] = pd.cut(all_data['Temperature'], bins=temperature_bins, labels=temperature_labels, right=False)

# Function to calculate the average current density and standard deviation for each bin range
def calculate_stats(df, bin_column):
    grouped = df.groupby(['Electrode', bin_column])['Log Abs Current Density']
    stats = grouped.agg(['mean', 'std']).reset_index()
    stats = stats.rename(columns={'mean': 'Avg Current Density', 'std': 'Std Dev'})
    return stats

# Calculate stats for humidity and temperature bins
stats_humidity = calculate_stats(all_data, 'Humidity Bin')
stats_temperature = calculate_stats(all_data, 'Temperature Bin')

# Sort by Electrode and Bin Range
stats_humidity = stats_humidity.sort_values(by=['Electrode', 'Humidity Bin'])
stats_temperature = stats_temperature.sort_values(by=['Electrode', 'Temperature Bin'])

# Add a column to indicate the type of bin (Humidity or Temperature)
stats_humidity['Bin Type'] = 'Humidity'
stats_temperature['Bin Type'] = 'Temperature'

# Rename the bin columns for uniformity
stats_humidity = stats_humidity.rename(columns={'Humidity Bin': 'Bin Range'})
stats_temperature = stats_temperature.rename(columns={'Temperature Bin': 'Bin Range'})

# Combine the results into a single DataFrame
combined_data = pd.concat([stats_humidity, stats_temperature], ignore_index=True)

# Save the combined data to a single Excel sheet
output_path = r'C:\Users\Owner\OneDrive\Desktop\combined_avg_density_heatmap_points.xlsx'
combined_data.to_excel(output_path, index=False)

print(f"Excel sheet saved to {output_path}")

# Define plotting function
def plot_scatter_with_error_bars(df, bin_type):
    electrode_types = df['Electrode'].unique()
    for electrode in electrode_types:
        subset = df[df['Electrode'] == electrode]
        plt.figure(figsize=(12, 8))
        plt.errorbar(subset['Bin Range'], subset['Avg Current Density'], yerr=subset['Std Dev'], fmt='o', capsize=5, label=electrode, markersize=20)
        plt.xlabel(f'{bin_type} Bin Range', fontsize=20, fontweight='bold')
        plt.ylabel('Average Log Abs Current Density', fontsize=20, fontweight='bold')
        plt.title(f'Average Log Abs Current Density vs {bin_type} for {electrode}', fontsize=22, fontweight='bold')
        plt.xticks(fontsize=16, fontweight='bold')
        plt.yticks(fontsize=16, fontweight='bold')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'C:\\Users\\Owner\\OneDrive\\Desktop\\plots\\{electrode}_{bin_type}.png')
        plt.show()

# Create output directory for plots
plot_dir = r'C:\Users\Owner\OneDrive\Desktop\plots'
os.makedirs(plot_dir, exist_ok=True)

# Plot the data
plot_scatter_with_error_bars(stats_humidity, 'Humidity')
plot_scatter_with_error_bars(stats_temperature, 'Temperature')



import matplotlib.pyplot as plt
import os

import pandas as pd
import matplotlib.pyplot as plt
import os# Load the data from the Excel file
excel_file_path = r'C:\Users\Owner\OneDrive\Desktop\sync pending\updated_weather_date_hour_location_new.xlsx'
all_data = pd.read_excel(excel_file_path, sheet_name=None)

# Combine all sheets into a single DataFrame
data = pd.concat(all_data.values(), ignore_index=True)

# Ensure proper types
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data['Humidity'] = data['Humidity'].astype(float)
data['Temperature'] = data['Temperature'].astype(float)
data['Log Abs Current Density'] = data['Log Abs Current Density'].astype(float)

# Define the humidity and temperature bins
humidity_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
temperature_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
humidity_labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']
temperature_labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']

# Bin the humidity and temperature values
data['Humidity Bin'] = pd.cut(data['Humidity'], bins=humidity_bins, labels=humidity_labels, right=False)
data['Temperature Bin'] = pd.cut(data['Temperature'], bins=temperature_bins, labels=temperature_labels, right=False)

# Mark shorts (1 if Current Density > 0, else 0) per file per bin range
data['Shorts'] = (data['Log Abs Current Density'] > 0).astype(int)

# Filter data for Au and Ag electrodes
au_data = data[data['Electrode'].str.lower() == 'au']
ag_data = data[data['Electrode'].str.lower() == 'ag']

# Function to calculate shorts per bin range per file
def calculate_shorts_per_bin(data, bin_column):
    # Group by File Name, Electrode, and Bin Column and take max to ensure 1 short per file per bin range
    shorts_summary = data.groupby(['Time', 'Electrode', bin_column])['Shorts'].max().reset_index()
    return shorts_summary

# Calculate shorts summary for Au and Ag electrodes for Humidity and Temperature bins
au_humidity_shorts = calculate_shorts_per_bin(au_data, 'Humidity Bin')
au_temperature_shorts = calculate_shorts_per_bin(au_data, 'Temperature Bin')
ag_humidity_shorts = calculate_shorts_per_bin(ag_data, 'Humidity Bin')
ag_temperature_shorts = calculate_shorts_per_bin(ag_data, 'Temperature Bin')

# Define the folder to save the results
output_folder = r'C:\Users\Owner\OneDrive\Desktop\shorts_summary_per_electrode'

# Create the folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Save the results to separate Excel sheets for Au and Ag electrodes and for Humidity and Temperature bins
with pd.ExcelWriter(os.path.join(output_folder, 'shorts_summary_per_electrode.xlsx')) as writer:
    au_humidity_shorts.to_excel(writer, sheet_name='AU Humidity Shorts', index=False)
    au_temperature_shorts.to_excel(writer, sheet_name='AU Temperature Shorts', index=False)
    ag_humidity_shorts.to_excel(writer, sheet_name='AG Humidity Shorts', index=False)
    ag_temperature_shorts.to_excel(writer, sheet_name='AG Temperature Shorts', index=False)

# Define the path to the existing Excel sheet
input_folder = r'C:\Users\Owner\OneDrive\Desktop\shorts_summary_per_electrode'
input_file = os.path.join(input_folder, 'shorts_summary_per_electrode.xlsx')

# Load the Excel file
all_sheets = pd.read_excel(input_file, sheet_name=None)

# Initialize an empty DataFrame to store the aggregated results
summary_data = pd.DataFrame()

# Process each sheet in the loaded Excel file
for sheet_name, df in all_sheets.items():
    # Check if the sheet name contains 'AU' or 'AG' and 'Humidity' or 'Temperature' to identify relevant data
    if 'au' in sheet_name.lower() or 'ag' in sheet_name.lower():
        electrode_type = sheet_name[:2].upper()  # Extract 'AU' or 'AG'
        if 'humidity' in sheet_name.lower():
            bin_column = 'Humidity Bin'
        elif 'temperature' in sheet_name.lower():
            bin_column = 'Temperature Bin'
        else:
            continue
        
        # Group by bin column and sum up the shorts
        shorts_summary = df.groupby(bin_column)['Shorts'].sum().reset_index()
        
        # Add electrode type column
        shorts_summary['Electrode'] = electrode_type
        
        # Append to summary_data
        summary_data = pd.concat([summary_data, shorts_summary])

# Define the folder to save the results
output_folder = r'C:\Users\Owner\OneDrive\Desktop\shorts_summary_per_electrode'

# Create the folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Save the aggregated results to a new Excel file without file name column
output_file = os.path.join(output_folder, 'shorts_summary_per_electrode_no_filename.xlsx')
summary_data.to_excel(output_file, index=False)

print(f"Results saved successfully to {output_file}.")

# Define the path to the existing Excel sheet
input_folder = r'C:\Users\Owner\OneDrive\Desktop\shorts_summary_per_electrode'
input_file = os.path.join(input_folder, 'shorts_summary_per_electrode_no_filename.xlsx')

# Load the Excel file
summary_data = pd.read_excel(input_file)

# Ensure the columns are of type string for proper plotting
summary_data['Humidity Bin'] = summary_data['Humidity Bin'].astype(str)
summary_data['Temperature Bin'] = summary_data['Temperature Bin'].astype(str)

# Define the output folder for saving plots
output_folder = r'C:\Users\Owner\OneDrive\Desktop\shorts_summary_plots'
os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist

# Define the function to plot the scatter plots for each electrode type
def plot_shorts_by_electrode(data, bin_column, title_prefix, ylabel, file_name_suffix):
    for electrode in ['AU', 'AG']:
        plt.figure(figsize=(12, 8))
        subset = data[(data['Electrode'] == electrode) & (data[bin_column].str.strip() != 'nan')]
        plt.scatter(subset[bin_column], subset['Shorts'], s=100, label=electrode)  # Set marker size to 100
        plt.xlabel(f'{bin_column} Bin Range', fontsize=20, fontweight='bold')
        plt.ylabel(ylabel, fontsize=20, fontweight='bold')
        plt.title(f'{title_prefix} for Electrode {electrode}', fontsize=22, fontweight='bold')
        plt.xticks(fontsize=16, fontweight='bold')
        plt.yticks(fontsize=16, fontweight='bold')
        plt.grid(True)
        plt.legend(title='Electrode', fontsize=14, title_fontsize='16', loc='best')
        plt.tight_layout()  # Adjust layout to fit all elements
        plt.savefig(os.path.join(output_folder, f'{file_name_suffix}_{electrode}.png'))  # Save the plot as an image
        plt.close()  # Close the figure to free up memory

# Filter out NaN values from data
humidity_data = summary_data[summary_data['Humidity Bin'].str.strip() != 'nan']
temperature_data = summary_data[summary_data['Temperature Bin'].str.strip() != 'nan']

# Plot scatter plots for Humidity Bin for each electrode type
plot_shorts_by_electrode(humidity_data, 'Humidity Bin', 'Number of Shorts vs Humidity Bin Range', 'Number of Shorts', 'humidity_bins_vs_shorts')

# Plot scatter plots for Temperature Bin for each electrode type
plot_shorts_by_electrode(temperature_data, 'Temperature Bin', 'Number of Shorts vs Temperature Bin Range', 'Number of Shorts', 'temperature_bins_vs_shorts')
