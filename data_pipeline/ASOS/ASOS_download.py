"""
ASOS_download.py  (기상청 종관기상관측 일강수 다운로드)
========================================================
공공데이터포털(data.go.kr) 기상청 지상(종관, ASOS) 일자료 API로
관측소별 일강수량(sumRn)을 내려받아 CSV 하나로 저장한다.

ASOS 일자료는 이미 KST(00~24시 KST) 일경계이므로 추가 시간대 처리가 필요 없다.

준비
  1) data.go.kr 에서 "기상청_지상(종관, ASOS) 일자료 조회서비스" 활용신청
     → 서비스키 발급
  2) 환경변수로 등록 (키를 코드/깃에 절대 넣지 말 것):
       export KMA_SERVICE_KEY="..."
  3) 관측소 목록 CSV 준비 → data/Station_ASOS.csv
     ('지점' 컬럼 필수. 기상청 관측지점정보에서 내려받기)

입력 : data/Station_ASOS.csv    (지점 목록)
출력 : output/ASOS_daily_{시작}_{끝}.csv
       (컬럼: 지점, 지점명, 일시, 강수량[mm/day])

* 강수량 결측('')은 무강수로 0 처리한다.
"""
import os
import time
from datetime import datetime

import pandas as pd
import requests

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - 최소 환경 대비 편의 처리
    def tqdm(iterable, *args, **kwargs):
        return iterable

# ==============================================================================
# 설정
# ==============================================================================
HERE = os.path.dirname(os.path.abspath(__file__))
STATION_CSV = os.path.join(HERE, "data", "Station_ASOS.csv")
OUT_DIR = os.path.join(HERE, "output")

SERVICE_KEY = os.environ.get("KMA_SERVICE_KEY")

START_DT = "20210101"
END_DT = "20251231"

URL = "https://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList"
NUM_OF_ROWS = 999          # 페이지당 행 수 (일자료라 1년치도 한 페이지에 들어옴)
SLEEP_SEC = 0.05           # 서버 부하 방지


# ==============================================================================
# 다운로드
# ==============================================================================
def load_station_ids(path):
    try:
        meta = pd.read_csv(path, encoding="cp949")
    except UnicodeDecodeError:
        meta = pd.read_csv(path, encoding="utf-8")
    meta.columns = meta.columns.str.strip()
    ids = meta["지점"].astype(str).str.strip().tolist()
    print(f"관측소 {len(ids)}개 로드: {path}")
    return ids


def fetch_station_year(stn_id, year):
    """한 관측소의 1년치 일자료를 페이지 순회로 수집."""
    rows, page = [], 1
    start = max(f"{year}0101", START_DT)
    end = min(f"{year}1231", END_DT)
    while True:
        params = {
            "serviceKey": SERVICE_KEY,
            "pageNo": str(page),
            "numOfRows": str(NUM_OF_ROWS),
            "dataType": "JSON",
            "dataCd": "ASOS",
            "dateCd": "DAY",
            "startDt": start,
            "endDt": end,
            "stnIds": stn_id,
        }
        try:
            r = requests.get(URL, params=params, timeout=60)
            if r.status_code != 200:
                break
            body = r.json()["response"]["body"]
            items = body["items"]["item"]
            if not items:
                break
            for it in items:
                rows.append({
                    "지점": it.get("stnId"),
                    "지점명": it.get("stnNm"),
                    "일시": it.get("tm"),          # YYYY-MM-DD (KST)
                    "강수량": it.get("sumRn"),      # 일강수량 mm
                })
            if page * NUM_OF_ROWS >= int(body["totalCount"]):
                break
            page += 1
        except Exception:  # noqa: BLE001  (해당 연도에 자료 없으면 응답형식이 달라짐)
            break
        time.sleep(SLEEP_SEC)
    return rows


def main():
    if not SERVICE_KEY:
        raise SystemExit("환경변수 KMA_SERVICE_KEY 를 먼저 설정하세요. (data.go.kr 서비스키)")
    os.makedirs(OUT_DIR, exist_ok=True)

    station_ids = load_station_ids(STATION_CSV)
    y0, y1 = int(START_DT[:4]), int(END_DT[:4])

    all_rows = []
    for stn_id in tqdm(station_ids, desc="ASOS 지점"):
        for year in range(y0, y1 + 1):
            all_rows.extend(fetch_station_year(stn_id, year))

    df = pd.DataFrame(all_rows)
    if df.empty:
        raise SystemExit("수집된 자료가 없습니다. 서비스키/지점목록/기간을 확인하세요.")

    # 결측('') 강수량은 무강수 0 처리, 형 변환
    df["강수량"] = pd.to_numeric(df["강수량"], errors="coerce").fillna(0.0)
    df["일시"] = pd.to_datetime(df["일시"])

    out_path = os.path.join(OUT_DIR, f"ASOS_daily_{START_DT}_{END_DT}.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"완료: {len(df):,}행 저장 → {out_path}")
    print(f"  기간 {df['일시'].min().date()} ~ {df['일시'].max().date()}, "
          f"지점 {df['지점'].nunique()}개")


if __name__ == "__main__":
    main()
