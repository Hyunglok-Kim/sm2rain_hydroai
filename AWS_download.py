import os
import sys
import platform

import pandas as pd
import requests
import time
from datetime import datetime, timedelta
import io
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
URL = "https://apihub.kma.go.kr/api/typ01/url/sfc_aws_day.php?tm1=20210101&tm2=20251231&obs=rn_day&&stn=0&disp=0&help=1&authKey=zfxOjrUxRym8To61MVcpuQ"
output_FP = os.path.join(george_FP, 'Precipitation', 'AWS_Data.csv')

# Request data
try:
    print(f"Requesting data... (It may take 10-20 seconds)")
    response = requests.get(URL, timeout=60)
    
    if response.status_code != 200:
        print(f"[ERROR] Server response error: {response.status_code}")
        print("Key expired or address is wrong.")
        exit()
        
    raw_text = response.text
    
    # If the data is too short, it is likely an error message
    if len(raw_text) < 500:
        print(f"[WARNING] The data is too short. Check the content:\n{raw_text}")
        exit()

    print(f"Data received ({len(raw_text)/1024:.1f} KB)")

except Exception as e:
    print(f"[ERROR] Request failed: {e}")
    exit()

# Parse data
print("Parsing data...")

data_rows = []
lines = raw_text.split('\n')

for line in lines:
    line = line.strip()
    
    # Skip lines starting with # or empty lines
    if line.startswith('#') or not line:
        continue
    
    # Skip the last 'end' marker
    if line.startswith('*DATA'): 
        continue

    # Split by spaces (split() automatically handles multiple spaces)
    parts = line.split()
    
    # The data must have at least 6 elements (date, station, longitude, latitude, altitude, value)
    if len(parts) >= 6:
        # The station name may have spaces at the end (e.g. "Chungju(Gong) *" -> "Chungju(Gong)*")
        # However, we only need the value here, so we don't need to process the station name separately
        
        row = {
            '일시': parts[0],      # YYYYMMDD
            '지점': parts[1],      # STN ID
            '경도': parts[2],      # LON
            '위도': parts[3],      # LAT
            '고도': parts[4],      # HT
            '강수량': parts[5]     # VAL (강수량)
        }
        data_rows.append(row)

# =========================================================
# Save data
# =========================================================
if data_rows:
    df = pd.DataFrame(data_rows)
    
    # 날짜 형식 변환 (YYYYMMDD -> YYYY-MM-DD)
    df['일시'] = pd.to_datetime(df['일시'], format='%Y%m%d')
    
    # 숫자형 변환
    df['강수량'] = pd.to_numeric(df['강수량'], errors='coerce').fillna(0)
    df['위도'] = pd.to_numeric(df['위도'], errors='coerce')
    df['경도'] = pd.to_numeric(df['경도'], errors='coerce')
    
    # -99.0 같은 결측치는 0으로 처리 (강수량 특성상)
    df.loc[df['강수량'] < 0, '강수량'] = 0
    
    df.to_csv(output_FP, index=False, encoding='utf-8-sig')
    print(f"Done! Total {len(df)} data saved.")
    print(f"File path: {output_FP}")
    print("\n[Preview]")
    print(df.head())

else:
    print("[ERROR] 유효한 데이터를 찾지 못했습니다. 원본 텍스트 형식을 다시 확인하세요.")