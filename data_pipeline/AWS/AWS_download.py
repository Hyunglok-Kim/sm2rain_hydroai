"""
AWS_download.py  (기상청 방재기상관측 일강수 다운로드)
=======================================================
기상청 API허브(apihub.kma.go.kr)의 AWS 일자료(sfc_aws_day, rn_day)로
전국 AWS 관측소의 일강수량을 한 번에 내려받아 CSV 하나로 저장한다.

AWS 일자료는 이미 KST(00~24시 KST) 일경계이므로 추가 시간대 처리가 필요 없다.

준비
  1) https://apihub.kma.go.kr 회원가입 → API 인증키(authKey) 발급
  2) 환경변수로 등록 (키를 코드/깃에 절대 넣지 말 것):
       export KMA_APIHUB_KEY="..."

입력 : 없음 (전 지점 일괄 요청, stn=0)
출력 : output/AWS_daily_{시작}_{끝}.csv
       (컬럼: 일시, 지점, 경도, 위도, 고도, 강수량[mm/day])

* 응답 텍스트의 -99 등 음수 결측값은 무강수 0 으로 처리한다.
* 기간이 길면 응답이 커지므로 연 단위로 나눠 요청한다.
"""
import os
from datetime import datetime

import pandas as pd
import requests

# ==============================================================================
# 설정
# ==============================================================================
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "output")

AUTH_KEY = os.environ.get("KMA_APIHUB_KEY")

START_DT = "20210101"
END_DT = "20251231"

URL = "https://apihub.kma.go.kr/api/typ01/url/sfc_aws_day.php"


# ==============================================================================
# 다운로드
# ==============================================================================
def fetch_year(year):
    """1년치 전 지점 일강수 요청 → 파싱된 행 리스트."""
    tm1 = max(f"{year}0101", START_DT)
    tm2 = min(f"{year}1231", END_DT)
    params = {
        "tm1": tm1, "tm2": tm2,
        "obs": "rn_day",       # 일강수량
        "stn": "0",            # 전 지점
        "disp": "0",
        "help": "1",
        "authKey": AUTH_KEY,
    }
    print(f"  {year}: 요청 중... ", end="", flush=True)
    r = requests.get(URL, params=params, timeout=300)
    if r.status_code != 200:
        print(f"[실패] HTTP {r.status_code}")
        return []
    text = r.text
    if len(text) < 500:
        print(f"[경고] 응답이 너무 짧음 — 키/기간 확인 필요:\n{text[:200]}")
        return []
    print(f"수신 {len(text)/1024:.0f} KB")

    rows = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("*"):
            continue
        parts = line.split()
        # 형식: 일시(YYYYMMDD) 지점 경도 위도 고도 강수량 ...
        if len(parts) >= 6:
            rows.append({
                "일시": parts[0],
                "지점": parts[1],
                "경도": parts[2],
                "위도": parts[3],
                "고도": parts[4],
                "강수량": parts[5],
            })
    return rows


def main():
    if not AUTH_KEY:
        raise SystemExit("환경변수 KMA_APIHUB_KEY 를 먼저 설정하세요. (apihub.kma.go.kr 인증키)")
    os.makedirs(OUT_DIR, exist_ok=True)

    y0, y1 = int(START_DT[:4]), int(END_DT[:4])
    print(f"AWS 일강수 다운로드: {START_DT} ~ {END_DT}")

    all_rows = []
    for year in range(y0, y1 + 1):
        all_rows.extend(fetch_year(year))

    df = pd.DataFrame(all_rows)
    if df.empty:
        raise SystemExit("수집된 자료가 없습니다. 인증키/기간을 확인하세요.")

    # 형 변환 + 결측 처리 (-99 등 음수 → 무강수 0)
    df["일시"] = pd.to_datetime(df["일시"], format="%Y%m%d")
    for c in ("경도", "위도", "고도", "강수량"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.loc[df["강수량"] < 0, "강수량"] = 0.0
    df["강수량"] = df["강수량"].fillna(0.0)

    out_path = os.path.join(OUT_DIR, f"AWS_daily_{START_DT}_{END_DT}.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"완료: {len(df):,}행 저장 → {out_path}")
    print(f"  기간 {df['일시'].min().date()} ~ {df['일시'].max().date()}, "
          f"지점 {df['지점'].nunique()}개")


if __name__ == "__main__":
    main()
