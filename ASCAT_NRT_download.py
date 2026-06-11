import eumdac
import datetime

# 1. 인증 설정 (본인의 Key와 Secret 입력)
consumer_key = 'rUEgSuOfZTZ1vJItoR_swecO3Y0a'
consumer_secret = 'gSTn97P5_OJNOUv7WNHFysfdTOga'
credentials = (consumer_key, consumer_secret)
token = eumdac.AccessToken(credentials)
ds = eumdac.DataStore(token)

# 2. ASCAT 상품 선택 (토양 수분 NRT 예시)
# datastore_id = 'EO:EUM:DAT:METOP:SOMO12-NRT'
datastore_id = 'EO:EUM:DAT:METOP:SOMO12'
selected_collection = ds.get_collection(datastore_id)

# 3. 기간 설정 (2020년 1월 1일 ~ 2025년 현재)
start_date = datetime.datetime(2020, 1, 1)
end_date = datetime.datetime(2025, 3, 10) # 미래 날짜는 현재 시점까지 검색됨

# 4. 데이터 검색
products = selected_collection.search(dtstart=start_date, dtend=end_date)

print(f"Total products found: {len(products)}")

import os
os.chdir('/home/jaese/cpuserver_data/personal_data/project_KIHS/data/ASCAT_NRT')
# os.getcwd()
# sorted(os.listdir())[0]

# 5. 다운로드 루프
for product in products:
    print(f"Downloading: {product}")
    with product.open() as fsrc, open(fsrc.name, mode='wb') as fdst:
        fdst.write(fsrc.read())

