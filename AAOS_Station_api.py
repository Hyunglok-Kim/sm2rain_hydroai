import os
import sys
import platform

import pandas as pd
import requests
import time
from datetime import datetime, timedelta
from tqdm import tqdm
import xml.etree.ElementTree as ET
from urllib.parse import unquote

if platform.system() == 'Darwin': # Mac Mini
    base_FP = '/Users/geobambus'
    cpuserver_data_FP = os.path.join(base_FP, 'cpuserver_data')
    george_FP = os.path.join(cpuserver_data_FP, 'python_modules', 'kunhee')
elif platform.system() == 'Linux': # Workstation
    base_FP = '/home/kunhee'
    if os.path.exists(base_FP):
        cpuserver_data_FP = os.path.join(base_FP, 'cpuserver_data')
        nas_FP = os.path.join(base_FP, 'NAS')
        das_FP = os.path.join(base_FP, 'DAS')
        george_FP = os.path.join(cpuserver_data_FP, 'python_modules', 'kunhee')
    else: # CPU Server
        base_FP = '/data'
        nas_FP = '/data'
        das_FP = '/data'
        cpuserver_data_FP = '/data'

# File Paths
raw_service_key = 'b2b8bcd4c1bdc3dba096adcdbe6f4b6703937346e53104d70c0fe258a020f933' 
output_FP = os.path.join(george_FP, 'Precipitation', 'Station_AAOS.csv')

service_key = unquote(raw_service_key)

# Request station list
url = 'http://apis.data.go.kr/1390802/AgriWeather/getObsrSpotList'
params = {
    'serviceKey': service_key,
    'Page_No': '1',
    'Page_Size': '1000',
}

response = requests.get(url, params=params)

# Parse and save data
try:
    root = ET.fromstring(response.text)
    items = root.findall('.//item')
    
    data_list = []
    
    for i, item in enumerate(items):
        row = {}
        # Extract all sub-tags in item
        for child in item:
            row[child.tag] = child.text # The tag name becomes the column name
            
        data_list.append(row)
        
        # Print the tag configuration of the first data for debugging
        if i == 0:
            print(f"\n[First data sample tag check]")
            print(row.keys())

    if data_list:
        df = pd.DataFrame(data_list)
        
        # Save data
        os.makedirs(os.path.dirname(output_FP), exist_ok=True)
        
        df.to_csv(output_FP, index=False, encoding='utf-8-sig')
        print(f"\nDone! Total {len(df)} stations saved.")
        print(f"File path: {output_FP}")
        print("\n[Preview]")
        print(df.head())
    else:
        print("[WARNING] Data (item) not found. Check the response content.")
        print("Response content:", response.text[:300])

except Exception as e:
    print(f"Parsing error: {e}")