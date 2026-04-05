import os
from pathlib import Path
import re
from typing import Dict,List,Union
import pandas as pd
import numpy as np
import json


HERE: Path = Path(__file__).resolve().parent
RAW: Path = HERE.parent.parent/'datasets'/'raw'
DATASET:Path = HERE.parent.parent/'datasets'
print("here:",RAW)


data_dic: Dict = {
                'location':[],
                'carbon number':[],
                'electrode':[],
                'voltage':[], 
                'absJ':[], 
                'log_absJ':[], 
                'J':[],
                'current':[],
                'time':[],#below is metadata
                'date':[],
                'start time':[],
                'V(high)':[],
                'V(low)':[],
                'V(start/end)':[],
                'step':[],
                'NPLC':[],
                'delay':[],
                'autozero time':[],
                'number of V(high) to V(low) spans':[],
                'Junction diameter':[],
                'Magnification':[],
                'scan V direction':[],
                'spot ID':[],
                # 'temperature':[],
                # 'humidity':[],
                'scan ID':[],
                # 'file_path':[],
                # 'substrate_ID':[],
                'path':[]
                } 


data_arranged_by_array: Dict = {
                                    'location':[],
                                    'carbon number':[],
                                    'electrode':[],
                                    'voltage':[], 
                                    'absJ':[], 
                                    'log_absJ':[], 
                                    'J':[],
                                    'current':[],
                                    'time':[],#below is metadata
                                    'date':[],
                                    'start time':[],
                                    'V(high)':[],
                                    'V(low)':[],
                                    'V(start/end)':[],
                                    'step':[],
                                    'NPLC':[],
                                    'delay':[],
                                    'autozero time':[],
                                    'number of V(high) to V(low) spans':[],
                                    'Junction diameter':[],
                                    'Magnification':[],
                                    'scan V direction':[],
                                    'spot ID':[],
                                    # 'temperature':[],
                                    # 'humidity':[],
                                    'scan ID':[],
                                    # 'file_path':[],
                                    # 'substrate_ID':[],
                                    'path':[]
                                    } 


error_log = []



def extract_C_number(s: str) -> int:
    # Define the pattern to find 'C' followed by digits
    match = re.search(r'C(\d+)', s, re.IGNORECASE)
    if match:
        return int(match.group(1))
    else:
        return None


def convert_short_year(date_str: str) -> str:
    """Convert a date string with a short year to a string with a full year."""
    parts = date_str.split('-')
    if len(parts) == 3 and len(parts[2]) == 2:  # Check if the year is short-form
        month, day, year = parts
        year = '20' + year if int(year) < 100 else year  # Convert short year to full year
        return f"{month} {day}, {year}"
    return date_str



def parse_metadata(file_path) -> Dict:
    # Variable to track if we are before the first "Comments:" section
    before_comments = True
    
    # Dictionary to collect metadata
    collected_metadata: Dict = {
        'date': None,
        'start time': None,
        'V(high)': None,
        'V(low)': None,
        'V(start/end)': None,
        'step': None,
        'NPLC': None,
        'delay': None,
        'autozero time': None,
        'number of V(high) to V(low) spans': None,
        'Junction diameter': None,
        'Magnification': None,
    }

    try:
        with open(file_path, "r") as file:
            for line in file:
                line = line.strip()

                # Stop processing after the first "Comments:" line
                if line.startswith("Comments:"):
                    before_comments = False
                    break

                # Match date by checking common patterns (long-form date or short-form date)
                if before_comments:
                    if "," in line:  # Long-form date (e.g., May 06, 2010)
                        splited_parts = line.split(", ")
                        collected_metadata['date'] = ', '.join(splited_parts[1:])
                        continue
                    elif "-" in line:  # Short-form date (e.g., September-22-11)
                        date_parts = line.split('-')
                        if len(date_parts) == 3:
                            # Convert short year to full year
                            full_date = convert_short_year(line)
                            collected_metadata['date'] = full_date
                            continue

                    # Match time by checking for AM or PM
                    if "AM" in line or "PM" in line:
                        time_parts = line.split()
                        if len(time_parts) == 2:  # Time should be in "HH:MM AM/PM" format
                            collected_metadata['start time'] = line
                            continue

                # Skip empty lines
                if not line:
                    continue

                try:
                    # Process key-value pairs using split
                    if "=" in line and before_comments:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip()

                        # Attempt to convert value to float
                        try:
                            value = float(value.split()[0])
                        except ValueError:
                            value = None  # Use None for non-numeric values

                        if key in collected_metadata:
                            collected_metadata[key] = value
                except Exception as e:
                    raise RuntimeError(f"Failed to convert metadata to dict: {e}")

    except Exception as e:
        error_log.append({
            "file": str(file_path),  # Store the file that caused the error
            "error": str(e)           # Store the error message
        })

    return collected_metadata


def clean_column_names(column_name:str) -> str:
    # Remove numbers and extra underscores
    parts = column_name.strip().split("_") 
    return parts[0]


def get_spotID(file_path: Path) -> int:
    file_name = file_path.name
    print(file_name)
    
    try:
        # Initialize spot_ID to None
        spot_ID = None
        
        # Check for 'spot' or 'line' in the file name
        if "spot" in file_name or "line" in file_name:
            spot_ID = file_name.split(" ")[1].strip()
        
        # Check for 'test' in the file name, if found, set spot_ID to None
        elif "test" in file_name:
            spot_ID = None
        
        # If neither 'spot', 'line', nor 'test', try to use the first part of the file name
        else:
            spot_ID = file_name.split(" ")[0].strip()

        # If spot_ID is None, raise an error or return as None
        if spot_ID is None:
            raise ValueError(f"Invalid spot ID for file: {file_name}")
        
    except Exception as e:
        # Store errors in error_log if something goes wrong
        error_log.append({
            "file": str(file_path),
            "error": str(e)
        })
        return None  # Return None if an error occurs
    
    print(spot_ID)
    
    # Convert the extracted spot_ID to an integer
    try:
        return int(spot_ID)
    except ValueError:
        # Handle the case where the spot_ID cannot be converted to an integer
        error_log.append({
            "file": str(file_path),
            "error": "Spot ID is not a valid integer"
        })
        return None

def convert_txt_to_dataframe(txt_file_path:Path) -> pd.DataFrame:
    try:
        # Read the text file
        df = pd.read_csv(txt_file_path, sep='\t')
        df.columns = [clean_column_names(col) for col in df.columns]

# Optionally, drop any unwanted columns by name
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        return df 
    
    except Exception as e:
        raise RuntimeError(f"Failed to convert '{txt_file_path}': {e}")



v_scan_pos = [*(np.arange(0, 55, 5)/100), *(np.arange(50, -55, -5)/100), *(np.arange(-50, 5, 5)/100)]
v_scan_neg = [*(np.arange(0, -55, -5)/100), *(np.arange(-50, 55, 5)/100), *(np.arange(50, -5, -5)/100)]

v_half_scan_post = [*(np.arange(0, 55, 5)/100)]
v_half_scan_neg = [*(np.arange(0, -55, -5)/100)]


def assign_pattern_id(data: pd.DataFrame, column: str, new_column_name: str, scan_direction_column: str) -> tuple[pd.DataFrame,pd.DataFrame]:
    pattern_length = len(v_scan_pos)
    assert len(v_scan_pos) == len(v_scan_neg), "Patterns are not the same length!"
    current_id = 0
    ids = []
    scan_direction = []

    i = 0
    while i <= len(data) - pattern_length:
        # Check if the current slice matches the pattern
        if list(data[column].iloc[i:i + pattern_length]) == v_scan_pos:
            # Assign the current ID to all rows in the matched pattern
            ids.extend([current_id] * pattern_length)
            # If scan is toward positive V first, assign 1
            scan_direction.extend([1] * pattern_length)
            current_id += 1
            i += pattern_length  # Move to the next section after the matched pattern
        elif list(data[column].iloc[i:i + pattern_length]) == v_scan_neg:
            # Assign the current ID to all rows in the matched pattern
            ids.extend([current_id] * pattern_length)
            # If scan is toward negative V first, assign 0
            scan_direction.extend([-1] * pattern_length)
            current_id += 1
            i += pattern_length  # Move to the next section after the matched pattern
        else:
            
            if list(data[column].iloc[i:i + pattern_length]) == v_half_scan_post:
                ids.extend([current_id] * pattern_length)
                # If scan is toward positive V first, assign 1
                scan_direction.extend([.5] * pattern_length)
                current_id += 1
                i += pattern_length  # Move to the next section after the matched pattern


            elif list(data[column].iloc[i:i + pattern_length]) == v_half_scan_neg:
                ids.extend([current_id] * pattern_length)
                # If scan is toward positive V first, assign 1
                scan_direction.extend([-.5] * pattern_length)
                current_id += 1
                i += pattern_length  # Move to the next section after the matched pattern


            else:
                ids.append(0)
                scan_direction.append(0)
                i += 1

    # For any remaining rows at the end that don't have enough to match the pattern, assign NaN
    padding = [np.nan] * (len(data) - len(ids))
    ids.extend(padding)
    scan_direction.extend(padding)

    data[new_column_name] = ids
    data[scan_direction_column] = scan_direction

    aggregated_data = data.groupby([new_column_name,scan_direction_column]).agg({
        'voltage': lambda x: list(x),
        'absJ': lambda x: list(x),
        'J': lambda x: list(x),
        'time': lambda x: list(x),
        'log_absJ': lambda x: list(x),
        'current': lambda x: list(x),
    }).reset_index()

    return data,aggregated_data


def run_extracting_data()->None:
    for location in os.listdir(RAW):
        # print(location)
            # print(location)
            print(f'location is {location}')

            location_path = os.path.join(RAW, location)
            for electrode in os.listdir(location_path):
                print(f'electrode:{electrode}')
                carbon_number_path = os.path.join(location_path,electrode)
                for carbon_number in os.listdir(carbon_number_path):     
                    print(f'carbon_number is {carbon_number}')    
                    sub_path = os.path.join(carbon_number_path,carbon_number)
                    for sub in os.listdir(sub_path):
                        print(f'sub is : {sub}')
                        device_path = Path(os.path.join(sub_path,sub))
                        #     # two section: one is .text and another is wothout it
                        # add a try for the not meta pairs and add to list of errors!
                        metadata_file_paths= []
                        data_file_paths:List[Path] = list(device_path.rglob("*_data.txt"))
                        for data_file in data_file_paths:
                            # making the list of metadata
                            meta_data_files = data_file.with_name(data_file.stem.replace("_data", ""))
                            metadata_file_paths.append(meta_data_files)
                                

                                
                                            # processing both files simultaneously 
                        # print(data_file_paths)
                        for meta_file_path , data_file_path in zip(metadata_file_paths,data_file_paths):
                            # print(meta_file_path,  '\t'  , data_file_path )
                            C_number:int = extract_C_number(carbon_number)
                            spot_id: int = get_spotID(data_file_path)
                            df_extracted: pd.DataFrame = convert_txt_to_dataframe(data_file_path)
                            # Calculate the log of absJ
                            df_extracted['log_absJ'] = np.log10(df_extracted['absJ'])
                            df_extracted, grouped_pattern = assign_pattern_id(df_extracted, "voltage", "scan ID", "scan V direction")
                            extracted_data_dict: Dict = df_extracted.to_dict(orient='list')
                            grouped_pattern_dict: Dict = grouped_pattern.to_dict(orient='list')
                        
                            # try:

                            meta_dict: Dict = parse_metadata(file_path=meta_file_path)
                            # except Exception as e:
                            #     # Capture any error and add it to the error log
                            #     error_log.append({
                            #         "file": str(data_file_path),   # Store the file that caused the error
                            #         "error": str(e)           # Store the error message
                            #     })
                            

                            for key in extracted_data_dict:
                                if key in data_dic:
                                    data_dic[key].extend(extracted_data_dict[key])

                            # for grouped
                            for key in grouped_pattern_dict:
                                if key in data_arranged_by_array:
                                    data_arranged_by_array[key].extend(grouped_pattern_dict[key])

                            data_length = len(df_extracted)
                            grouped_length = len(grouped_pattern)
                            
                            if any(meta_dict.values()):
                                for key, value in meta_dict.items():
                                    if key in data_dic:
                                        data_dic[key].extend([value] * data_length)
                                    else:
                                        print(f"Key '{key}' not found in data_dic.")
                                    
                                    if key in data_arranged_by_array:
                                        data_arranged_by_array[key].extend([value] * grouped_length)
                                    else:
                                        print(f"Key '{key}' not found in data_arranged_by_array.")
                            
                            else:
                                for key, _ in meta_dict.items():
                                    if key in data_arranged_by_array:
                                        data_arranged_by_array[key].extend([np.nan] * grouped_length)
                                    else:    
                                        print(f"Key '{key}' not found in data_arranged_by_array.")

                                    if key in data_dic:
                                        data_dic[key].extend([np.nan] * data_length)
                                    else:
                                            print(f"Key '{np.nan}' not found in data_dict.")

  

                                


                            relative_path = data_file_path.parts[data_file_path.parts.index('raw')+1:]
                            relative_data_path = Path(*relative_path)
                            
                            # for not groupped
                            data_dic['spot ID'].extend([spot_id] * data_length)
                            data_dic['carbon number'].extend([C_number] * data_length)
                            data_dic['electrode'].extend([electrode] * data_length)
                            data_dic['location'].extend([location] * data_length)
                            data_dic['path'].extend([relative_data_path] * data_length)

                            # for grouped
                            data_arranged_by_array['spot ID'].extend([spot_id] * grouped_length)
                            data_arranged_by_array['carbon number'].extend([C_number] * grouped_length)
                            data_arranged_by_array['electrode'].extend([electrode] * grouped_length)
                            data_arranged_by_array['location'].extend([location] * grouped_length)
                            data_arranged_by_array['path'].extend([relative_data_path] * grouped_length)



    df = pd.DataFrame(data_dic)
    extracted_folder = DATASET/'extracted_files'
    full_df_path_csv = extracted_folder/"full_tunneling_J.csv"
    full_df_path_pkl =extracted_folder/"full_tunneling_J.pkl"
    df.to_csv(full_df_path_csv, index=False)
    df.to_pickle(full_df_path_pkl)

    grouped_file_path_csv = extracted_folder/"grouped_tunneling_J.csv"
    grouped_file_path_pkl = extracted_folder/"grouped_tunneling_J.pkl"

    df_grouped = pd.DataFrame(data_arranged_by_array)
    df_grouped.to_csv(grouped_file_path_csv, index=False)
    df_grouped.to_pickle(grouped_file_path_pkl)

    error_log_file_path = extracted_folder/"error_log.json"

    with error_log_file_path.open("w") as f:
        json.dump(error_log, f, indent=2)



run_extracting_data()



if any(np.isnan(value) for value in data_arranged_by_array['Magnification']):
    print("There are NaN values in 'date'.")
else:
    print("No NaN values in 'date'.")  


# file_path_for_data = r"C:\Users\sdehgha2\OneDrive - North Carolina State University\Desktop\PhD code\large area tunneling j\LAMoTuJ\datasets\raw\Ames, Iowa\Ag\C15\Substrate 1\5 2-21_data.txt"
# file_path_for_metadata = Path(r"C:\Users\sdehgha2\OneDrive - North Carolina State University\Desktop\PhD code\large area tunneling j\LAMoTuJ\datasets\raw\Ames, Iowa\Ag\C15\Substrate 1\5 2-21_data")
# lam_path = file_path_for_metadata.parts[file_path_for_metadata.parts.index('raw')+1:]
# relative_path_after_lamotuj = Path(*lam_path)

# print(relative_path_after_lamotuj)
# x_df= convert_txt_to_dataframe(file_path_for_data)
# x_df['log_absJ'] = np.log10(x_df['absJ'])
# x_df, agg_df = assign_pattern_id(x_df, "voltage", "scan ID", "scan V direction")


# aggregated_data = x_df.groupby(['scan ID','scan V direction']).agg({
#     'voltage': lambda x: list(x),
#     'absJ': lambda x: list(x),
#     'J': lambda x: list(x),
#     'time': lambda x: list(x),
#     'log_absJ': lambda x: list(x),
#     'current': lambda x: list(x),
# }).reset_index()

# print(aggregated_data)

# test = HERE/"test_grouped.csv"
# aggregated_data.to_csv(test, index=False)


# x_dic = x_df.to_dict(orient='list')
# # Print the dictionary with the parsed data
# di = parse_metadata(file_path=file_path_for_metadata)


# print(x_df)
# for key in x_df:
#     if key in data_dic:
#         data_dic[key].extend(x_dic[key])

# df_length = len(x_df)
# print(df_length)
# for key, value in di.items():
#     if key in data_dic:
#         data_dic[key].extend([value] * df_length)
#     else:
#         print(f"Key '{key}' not found in data_dic.")

# print(pd.DataFrame(data_dic))
# df = pd.DataFrame(data_dic)
# print(df)