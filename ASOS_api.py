import os
import sys
import platform

import pandas as pd
import requests
import time
from datetime import datetime, timedelta
from tqdm import tqdm

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
service_key = 'b2b8bcd4c1bdc3dba096adcdbe6f4b6703937346e53104d70c0fe258a020f933'
station_FP = os.path.join(george_FP, 'Precipitation', 'Station_ASOS.csv')
output_FP = os.path.join(george_FP, 'Precipitation', 'Data_ASOS.csv')

# Data Period: 2021.01.01 - 2025.12.31
start_year = 2021
end_year = 2025

# Data load
try:
    meta_df = pd.read_csv(station_FP, encoding='cp949')
except:
    meta_df = pd.read_csv(station_FP, encoding='utf-8')

# Extract station IDs
station_ids = meta_df['지점'].astype(str).str.strip().tolist()
print(f"Total {len(station_ids)} stations data requested.")

# Monthly data collection
url = 'https://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList'
all_data = []

total_months = (end_year - start_year + 1) * 12
current_progress = 0

for year in tqdm(range(start_year, end_year + 1)):
    for month in tqdm(range(1, 13)):
        if month == 12:
            next_month = datetime(year + 1, 1, 1)
        else:
            next_month = datetime(year, month + 1, 1)
        last_day = (next_month - timedelta(days=1)).day
        
        start_dt = f"{year}{month:02d}01"
        end_dt   = f"{year}{month:02d}{last_day:02d}"
        
        for stn_id in station_ids:
            params = {
                'serviceKey': service_key,
                'pageNo': '1',
                'numOfRows': '999',
                'dataType': 'JSON',
                'dataCd': 'ASOS',
                'dateCd': 'HR',
                'startDt': start_dt,
                'startHh': '00',
                'endDt': end_dt,
                'endHh': '23',
                'stnIds': stn_id
            }

            try:
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    try:
                        items = response.json()['response']['body']['items']['item']
                        if items:
                            for item in items:
                                all_data.append({
                                    '지점': item.get('stnId'),
                                    '지점명': item.get('stnNm'),
                                    '일시': item.get('tm'),
                                    '강수량': item.get('rn')
                                })
                    except:
                        pass
                
                time.sleep(0.02) 
            except Exception as e:
                print(f"   [Error] {stn_id} ({start_dt}): {e}")

# Save data
df = pd.DataFrame(all_data)
df.to_csv(output_FP, index=False, encoding='utf-8-sig')
print(f"\nDone! Total {len(df)} data saved.")
print(f"File path: {output_FP}")