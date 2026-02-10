import os
import sys
import platform

import pandas as pd
import requests
import time
from datetime import datetime, timedelta
from tqdm import tqdm
from urllib.parse import unquote
from xml.etree import ElementTree as ET

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

station_FP = os.path.join(george_FP, 'Precipitation', 'Station_WAMIS.csv')
output_FP  = os.path.join(george_FP, 'Precipitation', 'Data_WAMIS.csv')

# Hourly Rainfall Endpoint
URL = "http://www.wamis.go.kr:8080/wamis/openapi/wkw/rf_hrdata"

start_year = 2021
end_year   = 2021

print("--- WAMIS DATA DOWNLOAD (Hybrid Parsing) ---")

# =========================================================
# 2. LOAD STATIONS
# =========================================================
if not os.path.exists(station_FP):
    print(f"[ERROR] Fixed station file not found: {station_FP}")
    print("Please run the Coordinate Fix script first.")
    exit()

meta_df = pd.read_csv(station_FP, dtype={'obscd': str})
station_codes = meta_df['obscd'].tolist()
print(f"1. Target Stations: {len(station_codes)}")

# =========================================================
# 3. DOWNLOAD LOOP
# =========================================================
print("2. Downloading Data...")
all_data = []

# Loop by station first, then by year (WAMIS prefers this)
for idx, code in enumerate(station_codes):
    if idx % 10 == 0: print(f"   Processing Station {idx+1}/{len(station_codes)}: {code}")
    
    for year in range(start_year, end_year + 1):
        s_dt = f"{year}0101"
        e_dt = f"{year}1231"
        
        params = {
            'obscd': code,
            'startdt': s_dt,
            'enddt': e_dt,
            'output': 'json'
        }
        
        try:
            response = requests.get(URL, params=params, timeout=10)
            if response.status_code == 200:
                text = response.text.strip()
                items_found = False
                
                # STRATEGY 1: JSON
                try:
                    data = response.json()
                    if 'list' in data:
                        for item in data['list']:
                            # Only keep valid rainfall data
                            if item.get('rn') is not None:
                                all_data.append({
                                    '지점': code,
                                    '일시': item.get('ymdhm'),
                                    '강수량': item.get('rn')
                                })
                        items_found = True
                except:
                    pass
                
                # STRATEGY 2: XML Fallback
                if not items_found and text.startswith('<'):
                    try:
                        root = ET.fromstring(text)
                        # WAMIS XML structure usually has <list>...<list> or <item>
                        items = root.findall('.//list') 
                        if not items: items = root.findall('.//item')
                        
                        for item in items:
                            ymdhm = item.find('ymdhm')
                            rn = item.find('rn')
                            
                            if ymdhm is not None and rn is not None:
                                all_data.append({
                                    '지점': code,
                                    '일시': ymdhm.text,
                                    '강수량': rn.text
                                })
                    except:
                        pass
                        
            time.sleep(0.05)
            
        except Exception as e:
            print(f"      [Error] {code} ({year}): {e}")

# =========================================================
# 4. SAVE
# =========================================================
if all_data:
    df = pd.DataFrame(all_data)
    # Clean up dates (YYYYMMDDHH) -> standard datetime if needed
    # WAMIS sends '2021010101' (Hour) or '20210101' (Day)
    
    os.makedirs(os.path.dirname(output_FP), exist_ok=True)
    df.to_csv(output_FP, index=False, encoding='utf-8-sig')
    print(f"\n[SUCCESS] Saved {len(df)} rows to {output_FP}")
    print(df.head())
else:
    print("\n[ERROR] No data downloaded. Check the API key or internet connection.")