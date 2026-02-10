import pandas as pd
import requests
import time
import os

# =============================================================================
# 1. 설정 (날짜와 파일 경로 확인)
# =============================================================================
# CSV File Path
station_file_path = 'Precipitation/Station_AWS.csv' 

# Time Period
TARGET_START_DATE = '20210101'
TARGET_END_DATE = '20241231'

# Service Key
SERVICE_KEY = 'b2b8bcd4c1bdc3dba096adcdbe6f4b6703937346e53104d70c0fe258a020f933'

# =============================================================================
# 2. Load Station List (Read ASOS.csv)
# =============================================================================
try:
    # Usually cp949 encoding for Korean Windows files
    stn_df = pd.read_csv(station_file_path, encoding='cp949')
except UnicodeDecodeError:
    # If not, try utf-8
    stn_df = pd.read_csv(station_file_path, encoding='utf-8')

# Remove column names with spaces (to avoid errors)
stn_df.columns = stn_df.columns.str.strip()

# Select only active stations ('End Date' is empty)
active_stations = stn_df[stn_df['종료일'].isna()]

# Extract station IDs
station_ids = active_stations['지점'].astype(str).tolist()

print(f"Total {len(station_ids)} stations data requested.")

# =============================================================================
# 3. API 호출 루프 (전국 돌기)
# =============================================================================
url = 'https://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList'
all_rain_data = []

for i, stn_id in enumerate(station_ids):
    print(f"[{i+1}/{len(station_ids)}] 지점번호 {stn_id} 요청 중...", end=" ")
    
    params = {
        'serviceKey': SERVICE_KEY,
        'pageNo': '1',
        'numOfRows': '999',     # 넉넉하게
        'dataType': 'JSON',
        'dataCd': 'ASOS',
        'dateCd': 'HR',         # 시간 단위
        'startDt': TARGET_START_DATE,
        'startHh': '00',
        'endDt': TARGET_END_DATE,
        'endHh': '23',
        'stnIds': stn_id
    }
    
    try:
        response = requests.get(url, params=params)
        
        # 데이터가 정상적으로 왔는지 확인
        if response.status_code == 200:
            data_json = response.json()
            
            # 응답 안에 실제 데이터(item)가 있는지 체크
            if 'response' in data_json and 'body' in data_json['response']:
                items = data_json['response']['body']['items']['item']
                
                # 데이터가 하나라도 있으면 리스트에 추가
                if items:
                    for item in items:
                        # 필요한 정보만 쏙쏙 뽑아서 저장 (지점, 시간, 강수량)
                        all_rain_data.append({
                            '지점': item.get('stnId'),
                            '지점명': item.get('stnNm'),
                            '일시': item.get('tm'),
                            '강수량': item.get('rn')  # 강수량이 없으면 빈칸일 수 있음
                        })
                    print("성공")
                else:
                    print("데이터 없음 (기간 내 관측값 없음)")
            else:
                print("API 응답 형식 오류")
        else:
            print("API 호출 실패")
            
    except Exception as e:
        print(f"에러 발생: {e}")
    
    # 기상청 서버에 너무 빨리 요청하면 차단당할 수 있으니 0.1초 쉬기
    time.sleep(0.1)

# =============================================================================
# 4. 결과 저장
# =============================================================================
if all_rain_data:
    result_df = pd.DataFrame(all_rain_data)
    
    # 강수량 빈칸('')을 0으로 채울지, 그냥 둘지 결정 (보통 분석할 땐 0으로 많이 채움)
    # result_df['강수량'] = result_df['강수량'].replace('', '0')
    
    file_name = f"AWS_{TARGET_START_DATE}_{TARGET_END_DATE}.csv"
    result_df.to_csv(file_name, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*50)
    print(f"수집 완료! 파일이 생성되었습니다: {file_name}")
    print(result_df.head())
    print("="*50)
else:
    print("\n수집된 데이터가 없습니다. 날짜나 키를 확인해보세요.")