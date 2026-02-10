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

# File Paths
service_key_raw = 'b2b8bcd4c1bdc3dba096adcdbe6f4b6703937346e53104d70c0fe258a020f933'
service_key = unquote(service_key_raw)

station_FP = os.path.join(george_FP, 'Precipitation', 'Station_AAOS.csv')
output_FP  = os.path.join(george_FP, 'Precipitation', 'Data_AAOS.csv')

# Target Period
start_year = 2021
end_year   = 2021

print("--- STARTING AAOS DOWNLOAD (Hybrid Parsing) ---")

# =========================================================
# 2. LOAD STATION LIST
# =========================================================
print(f"1. Loading Station File: {station_FP}")
try:
    meta_df = pd.read_csv(station_FP)
    # Fix column names just in case
    meta_df.columns = meta_df.columns.str.strip()
    
    # Find the Code column dynamically
    code_col = next((c for c in meta_df.columns if 'code' in c.lower() and 'spot' in c.lower()), None)
    if not code_col: code_col = 'Obsr_Spot_Code' # Fallback
    
    station_ids = meta_df[code_col].astype(str).tolist()
    print(f"   - Loaded {len(station_ids)} stations (ID Col: {code_col})")

except Exception as e:
    print(f"[ERROR] Reading CSV failed: {e}")
    exit()

# =========================================================
# 3. DOWNLOAD LOOP
# =========================================================
URL = "http://apis.data.go.kr/1390802/AgriWeather/getRdaAws1HourList"
all_data = []

print("2. Downloading Data...")

for year in range(start_year, end_year + 1):
    for month in range(1, 13):
        # Date Logic
        if month == 12: next_month = datetime(year + 1, 1, 1)
        else: next_month = datetime(year, month + 1, 1)
        last_day = (next_month - timedelta(days=1)).day
        
        start_dt = f"{year}-{month:02d}-01"
        end_dt   = f"{year}-{month:02d}-{last_day:02d}"
        
        print(f"   Processing {year}-{month:02d} ...")

        for stn_code in station_ids:
            params = {
                'serviceKey': service_key,
                'Page_No': '1',
                'Page_Size': '999',
                'returnType': 'JSON', # Try JSON, but handle XML
                'search_Code': stn_code,
                'date_Gb': '1',
                'start_Date': start_dt,
                'end_Date': end_dt
            }

            try:
                response = requests.get(URL, params=params, timeout=10)
                if response.status_code == 200:
                    text = response.text.strip()
                    items_found = False
                    
                    # [STRATEGY 1] Try JSON
                    try:
                        data = response.json()
                        items = data['response']['body']['items']['item']
                        for item in items:
                            all_data.append({
                                '지점': stn_code,
                                '일시': item.get('Obsr_Tm'),
                                '강수량': item.get('Rain'),
                                '기온': item.get('Temp')
                            })
                        items_found = True
                    except:
                        # JSON failed. Fallback to Strategy 2.
                        pass

                    # [STRATEGY 2] Try XML (if JSON failed)
                    if not items_found and text.startswith('<'):
                        try:
                            root = ET.fromstring(text)
                            items = root.findall('.//item')
                            if items:
                                for item in items:
                                    # Safe extraction function
                                    def get_val(tag):
                                        node = item.find(tag)
                                        return node.text if node is not None else None
                                    
                                    all_data.append({
                                        '지점': stn_code,
                                        '일시': get_val('Obsr_Tm'),
                                        '강수량': get_val('Rain'),
                                        '기온': get_val('Temp')
                                    })
                                items_found = True
                        except:
                            pass
                    
                    # [DEBUG] If still nothing found, print Error from server
                    if not items_found and ('ERROR' in text or 'EXCEPTION' in text):
                        # Only print this once to avoid spamming
                        if len(all_data) == 0:
                            print(f"      [Server Error] {text[:200]}")
                            
                time.sleep(0.01) # Fast mode
            except Exception as e:
                pass

# =========================================================
# 4. SAVE RESULTS
# =========================================================
print("3. Saving Data...")
if all_data:
    df = pd.DataFrame(all_data)
    os.makedirs(os.path.dirname(output_FP), exist_ok=True)
    df.to_csv(output_FP, index=False, encoding='utf-8-sig')
    print(f"\n[SUCCESS] Saved {len(df)} rows to {output_FP}")
    print(df.head())
else:
    print("\n[ERROR] No data downloaded.")
    print("Possibilities:")
    print("1. 'AgriWeather Observation Data' API (15078057) is not applied/approved.")
    print("2. The daily traffic limit is exceeded.")
    print("3. The API server is down.")