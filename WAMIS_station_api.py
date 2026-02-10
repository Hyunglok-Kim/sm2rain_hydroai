import os
import sys
import platform

import pandas as pd
import requests
import time
from datetime import datetime, timedelta
from tqdm import tqdm
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

output_FP = os.path.join(george_FP, 'Precipitation', 'Station_WAMIS_raw.csv')

# URL
URL = "http://www.wamis.go.kr:8080/wamis/openapi/wkw/rf_dubrfobs"

print("--- STARTING WAMIS STATION DOWNLOAD ---")

# Request by basin (from Han River to Jeju Island)
all_stations = []
basins = {
    '1': '한강',
    '2': '낙동강',
    '3': '금강',
    '4': '섬진강',
    '5': '영산강',
    '6': '제주도'
}

for code, name in basins.items():
    print(f"   - {name}({code}) basin request...")
    
    params = {
        'basin': code,    # basin code
        'oper': 'y',      # only operating stations
        'output': 'json'  # JSON format
    }
    
    try:
        response = requests.get(URL, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'list' in data:
                for item in data['list']:
                    all_stations.append({
                        'obscd': item.get('obscd'),     # station code (core)
                        'obsnm': item.get('obsnm'),     # station name
                        'mngorg': item.get('mngorg'),   # management organization
                        'addr': item.get('addr', ''),   # address (may be empty)
                        'lat': item.get('lat', ''),     # latitude (to check)
                        'lon': item.get('lon', '')      # longitude (to check)
                    })
    except Exception as e:
        print(f"     [ERROR] {name} basin request failed: {e}")

# Save data
if all_stations:
    df = pd.DataFrame(all_stations)
    df.drop_duplicates(subset=['obscd'], inplace=True)
    
    os.makedirs(os.path.dirname(output_FP), exist_ok=True)
    df.to_csv(output_FP, index=False, encoding='utf-8-sig')
    
    print(f"\nDone! Total {len(df)} stations saved.")
    print(f"File path: {output_FP}")
    print(df.head())
else:
    print("[ERROR] No data downloaded. Check the API key or internet connection.")