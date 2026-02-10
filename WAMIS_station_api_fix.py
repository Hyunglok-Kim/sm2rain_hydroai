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

input_FP  = os.path.join(george_FP, 'Precipitation', 'Station_WAMIS_raw.csv')
output_FP = os.path.join(george_FP, 'Precipitation', 'Station_WAMIS.csv')

# Info Endpoint (Gives Lat/Lon)
URL = "http://www.wamis.go.kr:8080/wamis/openapi/wkw/rf_obsinfo"

print("--- WAMIS STATION COORDINATE FIX ---")

# =========================================================
# 2. LOAD STATION LIST
# =========================================================
if not os.path.exists(input_FP):
    print(f"[ERROR] Input file not found: {input_FP}")
    exit()

df = pd.read_csv(input_FP, dtype={'obscd': str})
print(f"1. Loaded {len(df)} stations to fix.")

# =========================================================
# 3. FETCH COORDINATES LOOP
# =========================================================
print("2. Fetching Coordinates (Lat/Lon)...")

lats = []
lons = []
addrs = []

# Use tqdm for progress bar
for idx, row in tqdm(df.iterrows(), total=len(df)):
    code = row['obscd']
    
    params = {
        'obscd': code,
        'output': 'json'
    }
    
    found_lat = None
    found_lon = None
    found_addr = None
    
    try:
        response = requests.get(URL, params=params, timeout=5)
        
        if response.status_code == 200:
            # WAMIS sometimes returns list, sometimes dict
            data = response.json()
            
            item = None
            if 'list' in data and data['list']:
                item = data['list'][0]
            elif isinstance(data, dict):
                item = data
                
            if item:
                # WAMIS keys are often 'lat', 'lon', or 'aglat', 'aglon'
                found_lat = item.get('lat') or item.get('aglat')
                found_lon = item.get('lon') or item.get('aglon')
                found_addr = item.get('addr')
                
    except:
        pass
    
    # Store result (keep original if new one is None)
    lats.append(found_lat if found_lat else row.get('lat', ''))
    lons.append(found_lon if found_lon else row.get('lon', ''))
    addrs.append(found_addr if found_addr else row.get('addr', ''))
    
    # Be nice to the server
    if idx % 10 == 0: time.sleep(0.05)

# =========================================================
# 4. SAVE FIXED FILE
# =========================================================
df['lat'] = lats
df['lon'] = lons
df['addr'] = addrs

# Filter out failed ones (still empty lat/lon)
valid_df = df[df['lat'].notnull() & df['lon'].notnull() & (df['lat'] != '')]

# Save
os.makedirs(os.path.dirname(output_FP), exist_ok=True)
valid_df.to_csv(output_FP, index=False, encoding='utf-8-sig')

print(f"\nDone! Fixed coordinates for {len(valid_df)} stations.")
print(f"File saved to: {output_FP}")
print(valid_df[['obscd', 'obsnm', 'lat', 'lon']].head())