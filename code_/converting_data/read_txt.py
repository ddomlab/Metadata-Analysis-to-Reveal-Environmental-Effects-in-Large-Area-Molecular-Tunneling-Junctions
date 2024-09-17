import os
from pathlib import Path
import re
from typing import Dict,List,Union
import pandas as pd
import numpy as np
import json


HERE: Path = Path(__file__).resolve().parent
RAW: Path = HERE.parent.parent/'datasets'/'raw'
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
                # 'temperature':[],
                # 'humidity':[],
                'scan ID':[],
                # 'file_path':[],
                # 'substrate_ID':[],
                'Path':[]
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
                                    # 'temperature':[],
                                    # 'humidity':[],
                                    'scan ID':[],
                                    # 'file_path':[],
                                    # 'substrate_ID':[],
                                    'Path':[]
                                    } 



def extract_C_number(s: str) -> int:
    # Define the pattern to find 'C' followed by digits
    match = re.search(r'C(\d+)', s, re.IGNORECASE)
    if match:
        return int(match.group(1))
    else:
        return None


def parse_metadata(file_path)->Dict:
    
    # Define regex patterns for date and time
    date_pattern = re.compile(r'\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s*\w+\s*\d{1,2},\s*\d{4}\b')
    time_pattern = re.compile(r'\b\d{1,2}:\d{2}\s*(?:AM|PM)\b')
    # Variable to track if we are before the first "Comments:" section
    before_comments = True
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
    
    
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()

            # Stop processing after the first "Comments:" line
            if line.startswith("Comments:"):
                before_comments = False
                break
            
            # Match date and time using regex patterns
            if before_comments:
                if date_pattern.match(line):
                    collected_metadata['date']=line
                    continue
                elif time_pattern.match(line):
                    collected_metadata['start time']=line
                    continue

            # Skip empty lines
            if not line:
                continue
            try:
                # Process key-value pairs
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

            
    return collected_metadata

def clean_column_names(column_name:str) -> str:
    # Remove numbers and extra underscores
    parts = column_name.strip().split("_") 
    return parts[0]


def convert_txt_to_dataframe(txt_file_path) -> pd.DataFrame:
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
            # Assign NaN to the current row and move one step forward
            ids.append(np.nan)
            scan_direction.append(np.nan)
            i += 1

    # For any remaining rows at the end that don't have enough to match the pattern, assign NaN
    padding = [np.nan] * (len(data) - len(ids))
    ids.extend(padding)
    scan_direction.extend(padding)

    data[new_column_name] = ids
    data[scan_direction_column] = scan_direction

    aggregated_data = data.groupby(new_column_name).agg({
        'voltage': list,
        'absJ': list,
        'J': list,
        'current': list,
        'time': list,
        'log_absJ': list,
        new_column_name: list,
        scan_direction_column: list
    })

    return data,aggregated_data



error_log = []
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
                        try:
                            C_number:int = extract_C_number(carbon_number)
                            df_extracted: pd.DataFrame = convert_txt_to_dataframe(data_file_path)
                            # Calculate the log of absJ
                            df_extracted['log_absJ'] = np.log10(df_extracted['absJ'])
                            df_extracted, grouped_pattern = assign_pattern_id(df_extracted, "voltage", "scan ID", "scan V direction")
                            extracted_data_dict: Dict = df_extracted.to_dict(orient='list')
                            grouped_pattern_dict: Dict = grouped_pattern.to_dict(orient='list')
                            meta_dict: Dict = parse_metadata(file_path=meta_file_path)

                        except Exception as e:
                            # Capture any error and add it to the error log
                            error_log.append({
                                "file": str(data_file_path),   # Store the file that caused the error
                                "error": str(e)           # Store the error message
                            })
                            continue

                        for key in extracted_data_dict:
                            if key in data_dic:
                                data_dic[key].extend(extracted_data_dict[key])

                        # for grouped
                        for key in grouped_pattern_dict:
                            if key in data_arranged_by_array:
                                data_arranged_by_array[key].extend(grouped_pattern_dict[key])

                        data_length = len(df_extracted)
                        grouped_length = len(grouped_pattern)

                        for key, value in meta_dict.items():
                            if key in data_dic:
                                data_dic[key].extend([value] * data_length)
                            else:
                                print(f"Key '{key}' not found in data_dic.")

                        # for grouped
                        for key, value in meta_dict.items():
                            if key in data_arranged_by_array:
                                data_arranged_by_array[key].extend([value] * grouped_length)
                            else:
                                print(f"Key '{key}' not found in data_arranged_by_array.")
                        
                        data_dic['carbon number'].extend([C_number] * data_length)
                        data_dic['electrode'].extend([electrode] * data_length)
                        data_dic['location'].extend([location] * data_length)
                        data_dic['Path'].extend([meta_file_path] * data_length)

                        # for grouped
                        data_arranged_by_array['carbon number'].extend([C_number] * grouped_length)
                        data_arranged_by_array['electrode'].extend([electrode] * grouped_length)
                        data_arranged_by_array['location'].extend([location] * grouped_length)
                        data_arranged_by_array['Path'].extend([meta_file_path] * grouped_length)



df = pd.DataFrame(data_dic)
test = HERE/"full_tunneling_J.csv"
df.to_csv(test)
df_grouped = pd.DataFrame(data_arranged_by_array)
test_grouped = HERE/"grouped_tunneling_J.csv"
df_grouped.to_csv(test_grouped)

error_log_file = HERE/"error_log.csv"

with error_log_file.open("w") as f:
    json.dump(error_log, f, indent=2)

# file_path_for_data = r"C:\Users\sdehgha2\OneDrive - North Carolina State University\Desktop\PhD code\large area tunneling j\LAMoTuJ\datasets\raw\Ames, Iowa\Ag\C15\Substrate 1\5 2-21_data.txt"
# file_path_for_metadata = r"C:\Users\sdehgha2\OneDrive - North Carolina State University\Desktop\PhD code\large area tunneling j\LAMoTuJ\datasets\raw\Ames, Iowa\Ag\C15\Substrate 1\5 2-21_data"

# x_df= convert_txt_to_csv(file_path_for_data)
# x_df['log_absJ'] = np.log10(x_df['absJ'])
# x_df, agg_df = assign_pattern_id(x_df, "voltage", "scan ID", "scan V direction")

# print(agg_df)

# test = HERE/"test_grouped.csv"
# agg_df.to_csv(test)


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