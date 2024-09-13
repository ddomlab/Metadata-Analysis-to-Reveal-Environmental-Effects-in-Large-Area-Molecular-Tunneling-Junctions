import os
from pathlib import Path
import re
from typing import Dict,List,Union
import pandas as pd

HERE: Path = Path(__file__).resolve().parent
RAW: Path = HERE.parent.parent/'datasets'/'raw'
print("here:",RAW)


data_dic: Dict = {
                'location':[],
                'carbon number':[],
                'electrode':[],
                'voltage':[], 
                'abs J':[], 
                'Log Abs Current Density':[], 
                'J':[],
                'test duration (s)':[],#below is metadata
                'date':[],
                'time':[],
                'V(high) (V)':[],
                'V(low) (V)':[],
                'V(start/end) (V)':[],
                'step (V)':[],
                'NPLC':[],
                'delay (s)':[],
                'autozero time (s)':[],
                'number of V(high) to V(low) spans':[],
                'Junction diameter':[],
                'Magnification':[],
                'temperature':[],
                'humidity':[],
                'scan_ID':[],
                'file_path':[],
                'substrate_ID':[],
                } 


data_arranged_by_loop_array: Dict ={
                                    'location':[],
                                    'carbon number':[],
                                    'electrode':[],
                                    'voltage':[], # list of array
                                    'abs J':[], # list of array
                                    'Log Abs Current Density':[], # list of array
                                    'J':[], # list of array
                                    'test duration (s)':[],#below is metadata # list of array
                                    'date':[],
                                    'time':[],
                                    'V(high) (V)':[],
                                    'V(low) (V)':[],
                                    'V(start/end) (V)':[],
                                    'step (V)':[],
                                    'NPLC':[],
                                    'delay (s)':[],
                                    'autozero time (s)':[],
                                    'number of V(high) to V(low) spans':[],
                                    'Junction diameter':[],
                                    'Magnification':[],
                                    'temperature':[],
                                    'humidity':[],
                                    'scan_ID':[],
                                    'scan_direction':[],
                                    'substrate_ID':[],

                                    } 



def parse_metadata(file_path)->Dict:
    
    # Define regex patterns for date and time
    date_pattern = re.compile(r'\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s*\w+\s*\d{1,2},\s*\d{4}\b')
    time_pattern = re.compile(r'\b\d{1,2}:\d{2}\s*(?:AM|PM)\b')
    # Variable to track if we are before the first "Comments:" section
    before_comments = True
    collected_metadata: Dict = {
    'date': None,
    'time': None,
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
                    collected_metadata['time']=line
                    continue

            # Skip empty lines
            if not line:
                continue

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
    return collected_metadata


def convert_txt_to_csv(txt_file_path):
    try:
        # Read the text file
        df = pd.read_csv(txt_file_path, sep='\t')
        return df 
    
    except Exception as e:
        raise RuntimeError(f"Failed to convert '{txt_file_path}': {e}")

# for location in os.listdir(RAW):
#     # print(location)
#     if location == 'Ames, Iowa':
#         # print(location)
#         print(f'location is {location}')
#         location_path = os.path.join(RAW, location)
#         for electrode in os.listdir(location_path):
#             print(f'electrode:{electrode}')
#             carbon_number_path = os.path.join(location_path,electrode)
#             for carbon_number in os.listdir(carbon_number_path):     
#                 print(f'carbon_number is {carbon_number}')    
#                 sub_path = os.path.join(carbon_number_path,carbon_number)
#                 for sub in os.listdir(sub_path):
#                     print(f'sub is : {sub}')
#                     device_path = Path(os.path.join(sub_path,sub))
#                     #     # two section: one is .text and another is wothout it
#                     metadata_file_paths= []
#                     data_file_paths:List[Path] = device_path.rglob("*_data.txt")
#                     for data_file in data_file_paths:
#                         # making the list of metadata
#                         meta_data_files = data_file.with_name(data_file.stem.replace("_data", ""))
#                         metadata_file_paths.append(meta_data_files)
                    
#                     # processing both files simultaneously 
#                     for meta , dat in zip(metadata_file_paths,data_file_paths):
#                         print(meta,  '\t'  , dat )



file_path = r"C:\Users\sdehgha2\OneDrive - North Carolina State University\Desktop\PhD code\large area tunneling j\LAMoTuJ\datasets\raw\Ames, Iowa\Au\C9\04142016\5 2-11_data.txt"


print(convert_txt_to_csv(file_path))

# Print the dictionary with the parsed data
print(parse_metadata(file_path=file_path))