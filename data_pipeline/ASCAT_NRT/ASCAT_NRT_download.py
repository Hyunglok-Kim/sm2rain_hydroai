"""
ASCAT_NRT_download.py  (EUMETSAT Data Store에서 ASCAT 토양수분 다운로드)
=======================================================================
EUMETSAT Data Store(eumdac)에서 Metop ASCAT L2 25km 토양수분(SOMO12)
원본 산출물(.zip, 내부 .nat)을 기간 지정으로 내려받는다.

준비
  1) https://api.eumetsat.int/api-key/ 에서 Consumer Key/Secret 발급
  2) 환경변수로 등록 (키를 코드/깃에 절대 넣지 말 것):
       export EUMDAC_CONSUMER_KEY="..."
       export EUMDAC_CONSUMER_SECRET="..."
  3) pip install eumdac

입력 : 없음 (API에서 직접 검색)
출력 : raw/  (스크립트 위치 기준; 산출물별 .zip/.nat 파일)

* 이미 받은 파일(같은 이름)은 건너뛰므로 중단 후 재실행해도 이어받기 된다.
* 이후 단계: ASCAT_NRT_KST_process.py 가 raw/ 를 읽어
  한국 0.1° 격자 + KST 일경계 스택(ASCAT_daily_stack_KST.nc)을 만든다.
"""
import datetime
import os

import eumdac

# ==============================================================================
# 설정
# ==============================================================================
HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, "raw")          # 원본 저장 폴더

# 인증키는 환경변수에서 읽는다 (공개 저장소에 키 노출 금지)
CONSUMER_KEY = os.environ.get("EUMDAC_CONSUMER_KEY")
CONSUMER_SECRET = os.environ.get("EUMDAC_CONSUMER_SECRET")

# ASCAT L2 25km 토양수분 (SOMO12). NRT 컬렉션은 'EO:EUM:DAT:METOP:SOMO12-NRT'
DATASTORE_ID = "EO:EUM:DAT:METOP:SOMO12"

# 다운로드 기간
START_DATE = datetime.datetime(2021, 1, 1)
END_DATE = datetime.datetime(2025, 5, 1)


# ==============================================================================
# 메인
# ==============================================================================
def main():
    if not CONSUMER_KEY or not CONSUMER_SECRET:
        raise SystemExit(
            "환경변수 EUMDAC_CONSUMER_KEY / EUMDAC_CONSUMER_SECRET 를 먼저 설정하세요.\n"
            "  (발급: https://api.eumetsat.int/api-key/)"
        )

    os.makedirs(RAW_DIR, exist_ok=True)

    token = eumdac.AccessToken((CONSUMER_KEY, CONSUMER_SECRET))
    datastore = eumdac.DataStore(token)
    collection = datastore.get_collection(DATASTORE_ID)

    products = collection.search(dtstart=START_DATE, dtend=END_DATE)
    print(f"검색된 산출물: {len(products)}개  ({START_DATE.date()} ~ {END_DATE.date()})")

    n_new = n_skip = n_err = 0
    for product in products:
        try:
            with product.open() as fsrc:
                out_path = os.path.join(RAW_DIR, fsrc.name)
                if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                    n_skip += 1
                    continue
                print(f"다운로드: {fsrc.name}")
                with open(out_path, mode="wb") as fdst:
                    fdst.write(fsrc.read())
                n_new += 1
        except Exception as e:  # noqa: BLE001
            n_err += 1
            print(f"  [실패] {product}: {e}")

    print(f"완료: 신규 {n_new}, 건너뜀 {n_skip}, 실패 {n_err}  → {RAW_DIR}")


if __name__ == "__main__":
    main()
