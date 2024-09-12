import os
from pathlib import Path
import re
from typing import Dict,List

HERE: Path = Path(__file__).resolve().parent
RAW: Path = HERE.parent.parent/'datasets'/'raw'
print("here:",RAW)

V(high) =  0.500000 V
V(low) = -0.500000 V

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

for location in os.listdir(RAW):
    
    print(location)