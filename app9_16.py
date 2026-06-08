import os
import re
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
from dash import Dash, dcc, html, Input, Output, State, ctx, dash_table, no_update, ALL
from dash.exceptions import PreventUpdate


# -----------------------------
# .env 로더 (추가 패키지 불필요)
#   - 프로젝트 폴더의 .env 파일에서 KEY=VALUE 를 읽어 환경변수로 등록
#   - 이미 셸에 설정된 환경변수가 있으면 그 값을 우선 사용
# -----------------------------
def _load_dotenv(path=None):
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(path):
        return
    try:
        with open(path, encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = val
    except Exception:
        pass


_load_dotenv()


# 데이터 경로 자동 탐색 (루트 또는 data/ 하위 모두 지원)
_DATA_CANDIDATES = [
    "merged_data_v10.csv",
    os.path.join("data", "merged_data_v10.csv"),
    os.path.join("data", "merged_data_v11.csv"),
]
DATA_PATH = next((p for p in _DATA_CANDIDATES if os.path.exists(p)), _DATA_CANDIDATES[0])


# -----------------------------
# 0) Load
# -----------------------------
df = pd.read_csv(DATA_PATH)

# 좌표 없는 병원 제거(최소한의 지도 표시를 위해)
df = df.dropna(subset=["latitude", "longitude"]).copy()

# id/string 정리
for c in ["EMS_SN", "H_CODE", "H2_CODE"]:
    if c in df.columns:
        df[c] = df[c].astype(str)

# 연령 숫자화
df["H_AGE_NUM"] = pd.to_numeric(df.get("H_AGE", np.nan), errors="coerce")

# 컬럼 이름 안전 확보(데이터마다 다를 수 있어 자동선택)
COL_REGION_HOSP = "H_REGION_KOR" if "H_REGION_KOR" in df.columns else ("H_REGION" if "H_REGION" in df.columns else None)
COL_REGION_ADDR = "ADDS_SIDO_STD" if "ADDS_SIDO_STD" in df.columns else ("ADDS_SIDO" if "ADDS_SIDO" in df.columns else None)

if COL_REGION_HOSP is None:
    raise ValueError("병원 권역 컬럼(H_REGION_KOR 또는 H_REGION)을 찾지 못했습니다.")
if COL_REGION_ADDR is None:
    # 주소 컬럼 없으면 로컬/유입 계산은 0으로 처리
    COL_REGION_ADDR = "__NO_ADDR__"
    df[COL_REGION_ADDR] = np.nan


# -----------------------------
# 0-1) ICD (S/T 세부분류) 생성 (없으면 생성)
# -----------------------------
ICD_ST_DETAIL_RANGES = [
    ("S00", "S09", "머리의 손상 (S00–S09)"),
    ("S10", "S19", "목의 손상 (S10–S19)"),
    ("S20", "S29", "흉부의 손상 (S20–S29)"),
    ("S30", "S39", "복부/아래등/요추/골반의 손상 (S30–S39)"),
    ("S40", "S49", "어깨 및 위팔의 손상 (S40–S49)"),
    ("S50", "S59", "팔꿈치 및 아래팔의 손상 (S50–S59)"),
    ("S60", "S69", "손목 및 손의 손상 (S60–S69)"),
    ("S70", "S79", "고관절 및 대퇴의 손상 (S70–S79)"),
    ("S80", "S89", "무릎 및 아래다리의 손상 (S80–S89)"),
    ("S90", "S99", "발목 및 발의 손상 (S90–S99)"),
    ("T00", "T07", "여러 신체부위를 침범한 손상 (T00–T07)"),
    ("T08", "T14", "몸통/사지/여러 부위의 상세불명 손상 (T08–T14)"),
    ("T15", "T19", "자연개구를 통해 들어간 이물 (T15–T19)"),
    ("T20", "T25", "신체 표면의 화상 및 부식 (T20–T25)"),
    ("T26", "T28", "눈 및 내부기관의 화상 및 부식 (T26–T28)"),
    ("T29", "T32", "다발성 및 상세불명 부위의 화상 및 부식 (T29–T32)"),
    ("T33", "T35", "동상 (T33–T35)"),
    ("T36", "T50", "약물/약제/생물학적 물질에 의한 중독 (T36–T50)"),
    ("T51", "T65", "비의약품 물질의 독성효과 (T51–T65)"),
    ("T66", "T78", "외인의 기타 및 상세불명의 영향 (T66–T78)"),
    ("T79", "T79", "외상의 특정 조기합병증 (T79)"),
    ("T80", "T88", "달리 분류되지 않은 외과적/내과적 치료의 합병증 (T80–T88)"),
    ("T90", "T98", "손상/중독/외인의 기타 결과의 후유증 (T90–T98)"),
]


def extract_icd3(x):
    if pd.isna(x):
        return None
    s = str(x).strip().upper()
    m = re.match(r"^([A-Z]\d{2})", s)
    return m.group(1) if m else None


def _split_code(code3: str):
    return code3[0], int(code3[1:3])


def _in_range(code3: str, start: str, end: str) -> bool:
    cL, cN = _split_code(code3)
    sL, sN = _split_code(start)
    eL, eN = _split_code(end)

    if sL == eL == cL:
        return sN <= cN <= eN

    if sL <= cL <= eL:
        if cL == sL:
            return cN >= sN
        if cL == eL:
            return cN <= eN
        return True

    return False


def map_icd_st_detail(icd3: str):
    if not icd3:
        return None
    for start, end, label in ICD_ST_DETAIL_RANGES:
        if _in_range(icd3, start, end):
            return label
    return None


if "ICD_ST_DETAIL" not in df.columns:
    if "ICD01" in df.columns:
        df["ICD01_3"] = df["ICD01"].apply(extract_icd3)
        df["ICD_ST_DETAIL"] = df["ICD01_3"].apply(map_icd_st_detail)
    else:
        df["ICD_ST_DETAIL"] = np.nan


# -----------------------------
# 0-2) Transfer time (minutes): AD_D/T -> H2_D/T
# -----------------------------
def to_dt(date_series, time_series):
    d = pd.to_numeric(date_series, errors="coerce")
    t = pd.to_numeric(time_series, errors="coerce")
    out = pd.Series(pd.NaT, index=date_series.index, dtype="datetime64[ns]")

    idx = d.notna() & t.notna()
    if idx.any():
        dd = d[idx].astype(int)
        tt = t[idx].astype(int)
        hh = tt // 100
        mm = tt % 100
        s = dd.astype(str).str.zfill(8) + " " + hh.astype(str).str.zfill(2) + ":" + mm.astype(str).str.zfill(2)
        out.loc[idx] = pd.to_datetime(s, format="%Y%m%d %H:%M", errors="coerce")
    return out


if all(c in df.columns for c in ["AD_D", "AD_T", "H2_D", "H2_T"]):
    df["AD_DT"] = to_dt(df["AD_D"], df["AD_T"])
    df["H2_DT"] = to_dt(df["H2_D"], df["H2_T"])
    df["transfer_minutes"] = (df["H2_DT"] - df["AD_DT"]).dt.total_seconds() / 60.0
    df.loc[df["transfer_minutes"] < 0, "transfer_minutes"] = np.nan
else:
    df["transfer_minutes"] = np.nan


# -----------------------------
# 1) Aggregations
# -----------------------------
def agg_hospitals(df_):
    d = df_.dropna(subset=["latitude", "longitude", "H_CODE", "H_NM"]).copy()
    return (
        d.groupby(["H_CODE", "H_NM", COL_REGION_HOSP], as_index=False)
        .agg(
            patient_cnt=("EMS_SN", "nunique"),
            lat=("latitude", "mean"),
            lon=("longitude", "mean"),
        )
    )


def role_metrics(df_):
    d = df_.dropna(subset=["H_CODE", "H_NM", COL_REGION_HOSP]).copy()
    d = d.drop_duplicates(subset=["EMS_SN", "H_CODE"])

    total = d.groupby(["H_CODE", "H_NM", COL_REGION_HOSP], as_index=False).agg(
        total_patients=("EMS_SN", "nunique")
    )

    if COL_REGION_ADDR in d.columns and COL_REGION_ADDR != "__NO_ADDR__":
        local = d[d[COL_REGION_ADDR] == d[COL_REGION_HOSP]].groupby("H_CODE", as_index=False).agg(
            local_patients=("EMS_SN", "nunique")
        )
        inflow = d[d[COL_REGION_ADDR] != d[COL_REGION_HOSP]].groupby("H_CODE", as_index=False).agg(
            inflow_patients=("EMS_SN", "nunique")
        )
    else:
        local = total[["H_CODE"]].copy()
        local["local_patients"] = 0
        inflow = total[["H_CODE"]].copy()
        inflow["inflow_patients"] = 0

    transferred = d[d["H2_CODE"].notna() & (d["H2_CODE"] != "nan")].groupby("H_CODE", as_index=False).agg(
        transfer_patients=("EMS_SN", "nunique")
    )

    out = (
        total.merge(local, on="H_CODE", how="left")
        .merge(inflow, on="H_CODE", how="left")
        .merge(transferred, on="H_CODE", how="left")
    )
    out[["local_patients", "inflow_patients", "transfer_patients"]] = out[
        ["local_patients", "inflow_patients", "transfer_patients"]
    ].fillna(0)

    out["local_ratio"] = np.where(out["total_patients"] > 0, out["local_patients"] / out["total_patients"], 0.0)
    out["inflow_ratio"] = np.where(out["total_patients"] > 0, out["inflow_patients"] / out["total_patients"], 0.0)
    out["transfer_ratio"] = np.where(out["total_patients"] > 0, out["transfer_patients"] / out["total_patients"], 0.0)

    out["hub_score"] = np.log1p(out["total_patients"]) * (0.4 + 0.7*out["inflow_ratio"] + 0.7*out["transfer_ratio"])
    return out


def agg_regions(df_):
    """Region-level node aggregation (centroid = mean of hospital centroids)."""
    hosp_agg = agg_hospitals(df_)
    if len(hosp_agg) == 0:
        return hosp_agg.iloc[0:0].assign(
            total_patients=pd.Series(dtype=float),
            hosp_cnt=pd.Series(dtype=float),
        )

    reg = (
        hosp_agg
        .groupby(COL_REGION_HOSP, as_index=False)
        .agg(
            total_patients=("patient_cnt", "sum"),
            hosp_cnt=("H_CODE", "nunique"),
            lat=("lat", "mean"),
            lon=("lon", "mean"),
        )
    )
    return reg


def region_metrics(df_):
    """Region-level ratios (local/inflow/transfer)."""
    d = df_.dropna(subset=[COL_REGION_HOSP, "H_CODE"]).copy()
    d = d.drop_duplicates(subset=["EMS_SN", "H_CODE"])

    total = d.groupby(COL_REGION_HOSP, as_index=False).agg(total_patients=("EMS_SN", "nunique"))

    if COL_REGION_ADDR in d.columns and COL_REGION_ADDR != "__NO_ADDR__":
        local = d[d[COL_REGION_ADDR] == d[COL_REGION_HOSP]].groupby(COL_REGION_HOSP, as_index=False).agg(
            local_patients=("EMS_SN", "nunique")
        )
        inflow = d[d[COL_REGION_ADDR] != d[COL_REGION_HOSP]].groupby(COL_REGION_HOSP, as_index=False).agg(
            inflow_patients=("EMS_SN", "nunique")
        )
    else:
        local = total[[COL_REGION_HOSP]].copy()
        local["local_patients"] = 0
        inflow = total[[COL_REGION_HOSP]].copy()
        inflow["inflow_patients"] = 0

    transferred = d[d["H2_CODE"].notna() & (d["H2_CODE"] != "nan")].groupby(COL_REGION_HOSP, as_index=False).agg(
        transfer_patients=("EMS_SN", "nunique")
    )

    out = (
        total.merge(local, on=COL_REGION_HOSP, how="left")
             .merge(inflow, on=COL_REGION_HOSP, how="left")
             .merge(transferred, on=COL_REGION_HOSP, how="left")
    ).fillna(0)

    out["local_ratio"] = np.where(out["total_patients"] > 0, out["local_patients"] / out["total_patients"], 0.0)
    out["inflow_ratio"] = np.where(out["total_patients"] > 0, out["inflow_patients"] / out["total_patients"], 0.0)
    out["transfer_ratio"] = np.where(out["total_patients"] > 0, out["transfer_patients"] / out["total_patients"], 0.0)
    out["hub_score"] = np.log1p(out["total_patients"]) * (0.4 + 0.7*out["inflow_ratio"] + 0.7*out["transfer_ratio"])
    return out



def _build_dt_from_cols(d: pd.DataFrame, dcol: str, tcol: str) -> pd.Series:
    """Build pandas datetime from separate date/time columns.
    Accepts YYYYMMDD and HHMM (or HMM) as strings/numbers. Returns NaT on parse failure.
    """
    if dcol not in d.columns or tcol not in d.columns:
        return pd.Series(pd.NaT, index=d.index)
    date_s = d[dcol].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    time_s = d[tcol].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()

    # normalize empty/nan
    date_s = date_s.replace({"nan": "", "NaN": "", "None": ""})
    time_s = time_s.replace({"nan": "", "NaN": "", "None": ""})

    # pad: date to 8, time to 4
    date_s = date_s.str.zfill(8)
    time_s = time_s.str.zfill(4)

    dt = pd.to_datetime(date_s + time_s, format="%Y%m%d%H%M", errors="coerce")
    return dt


def _compute_transfer_minutes(d: pd.DataFrame) -> pd.Series:
    """Compute transfer minutes using preferred columns.
    Priority:
      1) existing 'transfer_minutes' numeric column
      2) (DC_D, DC_T) -> (H2_D, H2_T)
      3) fallback: NaN
    """
    if "transfer_minutes" in d.columns:
        tm = pd.to_numeric(d["transfer_minutes"], errors="coerce")
        if tm.notna().any():
            return tm

    # Preferred per user: discharge/decision time -> arrival at H2
    if set(["DC_D", "DC_T", "H2_D", "H2_T"]).issubset(d.columns):
        src = _build_dt_from_cols(d, "DC_D", "DC_T")
        dst = _build_dt_from_cols(d, "H2_D", "H2_T")
        tm = (dst - src).dt.total_seconds() / 60.0
        # remove obviously wrong negatives
        tm = tm.where(tm >= 0)
        return tm

    return pd.Series(np.nan, index=d.index)


def transfer_edges(df_, min_cnt=30):
    """
    Hospital -> Hospital edges based on H2_* columns.

    - Edge weight: unique patient count (cnt)
    - avg_transfer_min: mean of transfer_minutes (AD_DT -> H2_DT), if available
    - severe_ratio: ratio of severe patients on the edge (default: ISS>=16), if ISS exists
    """
    needed = ["H_CODE", "H_NM", COL_REGION_HOSP, "latitude", "longitude",
              "H2_CODE", "H2_ORG_NAME", "H2_latitude", "H2_longitude"]
    needed = [c for c in needed if c in df_.columns]
    d = df_.dropna(subset=needed).copy()
    d = d.drop_duplicates(subset=["EMS_SN", "H_CODE", "H2_CODE"])

    # ---- Normalize destination hospital code (H2_CODE) when the same hospital appears
    # in both H_* and H2_* columns but with different codes.
    # If H2_ORG_NAME matches a known H_NM, we map it to that H_CODE so in/out degrees align.
    if "H_NM" in d.columns and "H2_ORG_NAME" in d.columns:
        name_to_code = (
            d.dropna(subset=["H_NM", "H_CODE"])
             .drop_duplicates(subset=["H_NM"])
             .set_index("H_NM")["H_CODE"]
             .astype(str)
             .to_dict()
        )
        d["H_CODE"] = d["H_CODE"].astype(str)
        d["H2_CODE"] = d["H2_CODE"].astype(str)
        mapped = d["H2_ORG_NAME"].map(name_to_code)
        d.loc[mapped.notna(), "H2_CODE"] = mapped[mapped.notna()].astype(str)

    # transfer time (preferred: DC_D/DC_T -> H2_D/H2_T)
    d["_tm"] = _compute_transfer_minutes(d)

# severe flag (ISS>=16 is a common definition; if ISS not available, keep NaN)
    if "ISS" in d.columns:
        iss = pd.to_numeric(d["ISS"], errors="coerce")
        d["_severe"] = (iss >= 16).astype(int)
    else:
        d["_severe"] = np.nan

    e = (
        d.groupby(["H_CODE", "H_NM", COL_REGION_HOSP, "H2_CODE", "H2_ORG_NAME"], as_index=False)
        .agg(
            cnt=("EMS_SN", "nunique"),
            severe_cnt=("_severe", "sum"),
            src_lat=("latitude", "mean"),
            src_lon=("longitude", "mean"),
            dst_lat=("H2_latitude", "mean"),
            dst_lon=("H2_longitude", "mean"),
            avg_transfer_min=("_tm", "mean"),
        )
    )
    e["avg_transfer_min"] = e["avg_transfer_min"].round(1)

    # severe ratio
    if "severe_cnt" in e.columns:
        e["severe_cnt"] = pd.to_numeric(e["severe_cnt"], errors="coerce")
        e["severe_ratio"] = (e["severe_cnt"] / e["cnt"]).replace([np.inf, -np.inf], np.nan)
    else:
        e["severe_ratio"] = np.nan

    return e[e["cnt"] >= min_cnt].copy()


def transfer_edges_region(df_, min_cnt=30, region_centroids=None, hosp_to_region=None):
    """
    Region -> Region edges aggregated from hospital edges.
    Destination region is resolved by H2_CODE lookup in hosp_to_region.
    """
    if region_centroids is None or hosp_to_region is None:
        return pd.DataFrame(columns=[
            "SRC_REGION", "DST_REGION", "cnt", "avg_transfer_min",
            "src_lat", "src_lon", "dst_lat", "dst_lon"
        ])

    needed = ["H_CODE", COL_REGION_HOSP, "H2_CODE"]
    if not set(needed).issubset(df_.columns):
        return pd.DataFrame(columns=[
            "SRC_REGION", "DST_REGION", "cnt", "avg_transfer_min",
            "src_lat", "src_lon", "dst_lat", "dst_lon"
        ])

    d = df_.dropna(subset=["H_CODE", COL_REGION_HOSP, "H2_CODE"]).copy()
    d = d[d["H2_CODE"].astype(str) != "nan"].copy()
    d = d.drop_duplicates(subset=["EMS_SN", "H_CODE", "H2_CODE"])

    d["SRC_REGION"] = d[COL_REGION_HOSP].astype(str)
    d["DST_REGION"] = d["H2_CODE"].map(hosp_to_region).fillna("기타/미상")

    d["_tm"] = _compute_transfer_minutes(d)

    e = (
        d.groupby(["SRC_REGION", "DST_REGION"], as_index=False)
         .agg(
            cnt=("EMS_SN", "nunique"),
            avg_transfer_min=("_tm", "mean"),
         )
    )
    e = e[e["cnt"] >= min_cnt].copy()
    if len(e) == 0:
        return e.assign(src_lat=np.nan, src_lon=np.nan, dst_lat=np.nan, dst_lon=np.nan)

    e["avg_transfer_min"] = e["avg_transfer_min"].round(1)

    # attach centroid coords
    # region_centroids may be dict or DataFrame
    if isinstance(region_centroids, dict):
        region_centroids = pd.DataFrame.from_dict(region_centroids, orient="index").reset_index().rename(columns={"index": COL_REGION_HOSP})
    reg = region_centroids.set_index(COL_REGION_HOSP)[["lat", "lon"]]
    e["src_lat"] = e["SRC_REGION"].map(reg["lat"])
    e["src_lon"] = e["SRC_REGION"].map(reg["lon"])
    e["dst_lat"] = e["DST_REGION"].map(reg["lat"])
    e["dst_lon"] = e["DST_REGION"].map(reg["lon"])

    e = e.dropna(subset=["src_lat", "src_lon", "dst_lat", "dst_lon"])
    return e

def transfer_edges_for_table(df_, min_cnt=1):
    """
    Scenario E 테이블용 전원 edge.
    - 좌표(H2_latitude/H2_longitude) 없어도 포함
    - cnt, avg_transfer_min, severe_ratio만 계산
    """
    needed = ["EMS_SN", "H_CODE", "H_NM", "H2_CODE", "H2_ORG_NAME"]
    needed = [c for c in needed if c in df_.columns]
    d = df_.dropna(subset=["EMS_SN", "H_CODE", "H_NM"]).copy()

    # 전원 병원 이름/코드 둘 중 하나라도 있으면 포함
    if "H2_CODE" in d.columns:
        d["_h2code_ok"] = d["H2_CODE"].notna() & (d["H2_CODE"].astype(str).str.lower() != "nan")
    else:
        d["_h2code_ok"] = False

    if "H2_ORG_NAME" in d.columns:
        d["_h2name_ok"] = d["H2_ORG_NAME"].notna() & (d["H2_ORG_NAME"].astype(str).str.strip() != "")
    else:
        d["_h2name_ok"] = False

    d = d[d["_h2code_ok"] | d["_h2name_ok"]].copy()

    # 1인 1출발-도착 중복 제거
    key_cols = ["EMS_SN", "H_CODE"]
    if "H2_CODE" in d.columns:
        key_cols.append("H2_CODE")
    if "H2_ORG_NAME" in d.columns:
        key_cols.append("H2_ORG_NAME")
    d = d.drop_duplicates(subset=key_cols)

    # transfer time (DC->H2 preferred)
    d["_tm"] = _compute_transfer_minutes(d)

    # severe flag
    if "ISS" in d.columns:
        iss = pd.to_numeric(d["ISS"], errors="coerce")
        d["_severe"] = (iss >= 16).astype(int)
    else:
        d["_severe"] = np.nan

    # group
    grp_cols = ["H_CODE", "H_NM"]
    if "H2_CODE" in d.columns:
        grp_cols.append("H2_CODE")
    else:
        d["H2_CODE"] = np.nan
        grp_cols.append("H2_CODE")

    if "H2_ORG_NAME" in d.columns:
        grp_cols.append("H2_ORG_NAME")
    else:
        d["H2_ORG_NAME"] = "Unknown"
        grp_cols.append("H2_ORG_NAME")

    e = (
        d.groupby(grp_cols, as_index=False)
         .agg(
            cnt=("EMS_SN", "nunique"),
            severe_cnt=("_severe", "sum"),
            avg_transfer_min=("_tm", "mean"),
         )
    )

    e["avg_transfer_min"] = pd.to_numeric(e["avg_transfer_min"], errors="coerce").round(1)

    e["severe_cnt"] = pd.to_numeric(e["severe_cnt"], errors="coerce")
    e["severe_ratio"] = (e["severe_cnt"] / e["cnt"]).replace([np.inf, -np.inf], np.nan)

    return e[e["cnt"] >= min_cnt].copy()


def transfer_inout_tables(df_, selected_hcode: str, cnt_range=None, top_k=10):
    """
    Scenario E: 선택 병원 기준 Outflow / Inflow 테이블
    - 좌표 없는 H2도 포함 (테이블용 edges 사용)
    - Inflow는 (H2_CODE==H_CODE) 매칭 + (H2_ORG_NAME==H_NM) 이름 매칭 fallback
    """
    lo, hi = (cnt_range or [1, 10**9])
    lo = int(lo); hi = int(hi)

    # 좌표 없는 병원도 포함하도록 테이블용 edge 사용
    e_all = transfer_edges_for_table(df_, min_cnt=1)

    diag = {
        "edges_total": int(len(e_all)),
        "cnt_lo": lo,
        "cnt_hi": hi,
        "edges_in_range": 0,
        "edges_out_selected": 0,
        "edges_in_selected": 0,
    }

    if len(e_all) == 0:
        return (pd.DataFrame(columns=["dst_name","cnt","avg_transfer_min","severe_ratio"]),
                pd.DataFrame(columns=["src_name","cnt","avg_transfer_min","severe_ratio"]),
                diag)

    e = e_all[(e_all["cnt"] >= lo) & (e_all["cnt"] <= hi)].copy()
    diag["edges_in_range"] = int(len(e))

    selected_hcode = str(selected_hcode)

    # 선택 병원 이름(동일 병원 이름 매칭 fallback에 필요)
    sel_name = None
    try:
        sel_name = df_.loc[df_["H_CODE"].astype(str) == selected_hcode, "H_NM"].dropna().astype(str).iloc[0]
    except Exception:
        sel_name = None

    # Outflow: 출발 병원 코드 기준
    out_e = e[e["H_CODE"].astype(str) == selected_hcode].copy()

    # Inflow: (1) H2_CODE가 selected_hcode인 경우 + (2) H2_ORG_NAME이 selected 병원명인 경우
    in_mask = (e["H2_CODE"].astype(str) == selected_hcode)
    if sel_name is not None and "H2_ORG_NAME" in e.columns:
        in_mask = in_mask | (e["H2_ORG_NAME"].astype(str) == str(sel_name))
    in_e = e[in_mask].copy()

    diag["edges_out_selected"] = int(len(out_e))
    diag["edges_in_selected"] = int(len(in_e))

    out_df = (out_e.sort_values("cnt", ascending=False)
              .head(top_k)
              .assign(dst_name=lambda x: x["H2_ORG_NAME"].astype(str))
              [["dst_name", "cnt", "avg_transfer_min", "severe_ratio"]])

    in_df = (in_e.sort_values("cnt", ascending=False)
             .head(top_k)
             .assign(src_name=lambda x: x["H_NM"].astype(str))
             [["src_name", "cnt", "avg_transfer_min", "severe_ratio"]])

    for dff in (out_df, in_df):
        if "avg_transfer_min" in dff.columns:
            dff["avg_transfer_min"] = pd.to_numeric(dff["avg_transfer_min"], errors="coerce").round(1)
        if "severe_ratio" in dff.columns:
            dff["severe_ratio"] = (pd.to_numeric(dff["severe_ratio"], errors="coerce") * 100).round(1)

    return out_df, in_df, diag



def outcome_counts_for_hospital(df_, hospital_code: str):
    d = df_[df_["H_CODE"] == hospital_code].copy()
    out = {}
    for col in ["ER_RESULT_KOR", "AD_RESULT_KOR", "H2_ER_RESULT_KOR", "H2_ADM_RESULT_KOR"]:
        if col in d.columns:
            out[col] = d[col].fillna("NA").value_counts(dropna=False).head(6).to_dict()
        else:
            out[col] = {}
    return out


def outcome_counts_for_region(df_, region_name: str):
    d = df_[df_[COL_REGION_HOSP] == region_name].copy()
    out = {}
    for col in ["ER_RESULT_KOR", "AD_RESULT_KOR", "H2_ER_RESULT_KOR", "H2_ADM_RESULT_KOR"]:
        if col in d.columns:
            out[col] = d[col].fillna("NA").value_counts(dropna=False).head(6).to_dict()
        else:
            out[col] = {}
    return out


def hospital_shot_data(df_, hospital_code: str):
    d = df_[df_["H_CODE"] == hospital_code].copy()
    info = {
        "NAME": d["H_NM"].iloc[0] if len(d) else "",
        "REGION": d[COL_REGION_HOSP].iloc[0] if len(d) else "",
        "total": int(d["EMS_SN"].nunique()) if len(d) else 0,
    }

    sex = d["H_SEX"].fillna("NA").value_counts() if "H_SEX" in d.columns else pd.Series(dtype=int)
    ages = d["H_AGE_NUM"].dropna()
    er = d["ER_RESULT_KOR"].fillna("NA").value_counts() if "ER_RESULT_KOR" in d.columns else pd.Series(dtype=int)

    return info, sex, ages, er


def region_shot_data(df_, region_name: str):
    d = df_[df_[COL_REGION_HOSP] == region_name].copy()
    info = {
        "NAME": region_name,
        "REGION": region_name,
        "total": int(d["EMS_SN"].nunique()) if len(d) else 0,
        "hosp_cnt": int(d["H_CODE"].nunique()) if len(d) else 0,
    }
    sex = d["H_SEX"].fillna("NA").value_counts() if "H_SEX" in d.columns else pd.Series(dtype=int)
    ages = d["H_AGE_NUM"].dropna()
    er = d["ER_RESULT_KOR"].fillna("NA").value_counts() if "ER_RESULT_KOR" in d.columns else pd.Series(dtype=int)
    return info, sex, ages, er


# -----------------------------
# 2) Global slider ranges
# -----------------------------
all_edges = transfer_edges(df, min_cnt=1)
if len(all_edges):
    min_cnt_default = int(min(50, all_edges["cnt"].quantile(0.90)))
    min_cnt_default = max(1, min_cnt_default)
    max_cnt = int(max(10, all_edges["cnt"].max()))
    max_cnt = min(max_cnt, 500)
else:
    min_cnt_default = 1
    max_cnt = 100

cnt_range_default = [int(min_cnt_default), int(max_cnt)]


# -----------------------------
# 3) Filter helper (transfer-only / transfer-time filters removed)
# -----------------------------
def apply_filters(
    df_,
    hosp_regions=None,
    addr_regions=None,
    classes_kor=None,
    icd_st_detail=None,
    sex=None,
    age_range=None,
    iss_range=None,
    include_iss_na=True,
):
    d = df_

    if hosp_regions:
        d = d[d[COL_REGION_HOSP].isin(hosp_regions)]

    if addr_regions:
        d = d[d[COL_REGION_ADDR].isin(addr_regions)]

    if classes_kor and ("CLASSES_KOR" in d.columns):
        d = d[d["CLASSES_KOR"].isin(classes_kor)]

    if icd_st_detail and ("ICD_ST_DETAIL" in d.columns):
        d = d[d["ICD_ST_DETAIL"].isin(icd_st_detail)]

    if sex and ("H_SEX" in d.columns):
        d = d[d["H_SEX"].isin(sex)]

    if age_range and len(age_range) == 2:
        lo, hi = age_range
        d = d[(d["H_AGE_NUM"].notna()) & (d["H_AGE_NUM"] >= lo) & (d["H_AGE_NUM"] <= hi)]

    if ("ISS" in d.columns) and iss_range and len(iss_range) == 2:
        lo, hi = iss_range
        iss = pd.to_numeric(d["ISS"], errors="coerce")
        if include_iss_na:
            d = d[(iss.isna()) | ((iss >= lo) & (iss <= hi))]
        else:
            d = d[(iss.notna()) & (iss >= lo) & (iss <= hi)]

    return d


# -----------------------------
# 4) Arrowhead (line segments)
# -----------------------------
def _arrowhead_segments(src_lat, src_lon, dst_lat, dst_lon, head_len=0.08, head_angle_deg=25):
    dx = dst_lon - src_lon
    dy = dst_lat - src_lat
    norm = np.hypot(dx, dy)
    if norm == 0:
        return []

    ux, uy = dx / norm, dy / norm
    theta = np.deg2rad(head_angle_deg)

    def rot(u_x, u_y, ang):
        return (
            u_x*np.cos(ang) - u_y*np.sin(ang),
            u_x*np.sin(ang) + u_y*np.cos(ang),
        )

    bx, by = -ux, -uy
    r1x, r1y = rot(bx, by, +theta)
    r2x, r2y = rot(bx, by, -theta)

    p1_lon = dst_lon + head_len * r1x
    p1_lat = dst_lat + head_len * r1y
    p2_lon = dst_lon + head_len * r2x
    p2_lat = dst_lat + head_len * r2y

    return [
        ([dst_lat, p1_lat], [dst_lon, p1_lon]),
        ([dst_lat, p2_lat], [dst_lon, p2_lon]),
    ]



# Graph config for right-shot panel (disable scroll-zoom so mousewheel scroll works)
GRAPH_CFG_SHOT = {
    "displayModeBar": False,
    "scrollZoom": False,
    "doubleClick": "reset",
}
# -----------------------------
# 5) Shot charts
# -----------------------------
def build_sex_pie(sex_counts: pd.Series) -> go.Figure:
    fig = go.Figure()
    if sex_counts is not None and len(sex_counts) > 0:
        fig.add_trace(go.Pie(labels=sex_counts.index.tolist(), values=sex_counts.values.tolist()))
    fig.update_layout(title="성별 비율", margin=dict(l=10, r=10, t=40, b=10))
    return fig


def build_age_hist(ages: pd.Series) -> go.Figure:
    fig = go.Figure()
    if ages is not None and len(ages) > 0:
        fig.add_trace(go.Histogram(x=ages, xbins=dict(start=0, end=100, size=10)))
    fig.update_layout(
        title="나이대 분포", xaxis_title="나이", yaxis_title="환자 수",
        margin=dict(l=10, r=10, t=40, b=10)
    )
    return fig


def build_er_bar(er_counts: pd.Series) -> go.Figure:
    fig = go.Figure()
    if er_counts is not None and len(er_counts) > 0:
        top = er_counts.head(8)
        fig.add_trace(go.Bar(x=top.index.tolist(), y=top.values.tolist()))
    fig.update_layout(
        title="ER_RESULT_KOR 분포(상위)", xaxis_title="ER 결과", yaxis_title="환자 수",
        margin=dict(l=10, r=10, t=40, b=10)
    )
    return fig


# -----------------------------
# 6) Map figure builders
# -----------------------------
def _add_mapbox_dummy(fig, center_lat, center_lon):
    """Force mapbox subplot creation even when data is empty."""
    fig.add_trace(go.Scattermapbox(
        lat=[center_lat],
        lon=[center_lon],
        mode="markers",
        marker=dict(size=1, opacity=0),
        hoverinfo="skip",
        showlegend=False
    ))



# -----------------------------
# 2-1) Network centrality (Scenario C)
# -----------------------------
def compute_network_centrality(df_):
    """
    Build a directed hospital transfer graph and compute:
    - weighted in-degree / out-degree (strength)
    - PageRank (weighted)
    - community id (greedy modularity on undirected projection)
    - top-3 connected hospitals by total flow (in+out)

    Returns DataFrame keyed by H_CODE.
    """
    e = transfer_edges(df_, min_cnt=1)
    if e is None or len(e) == 0:
        return pd.DataFrame(columns=[
            "H_CODE", "in_degree_w", "out_degree_w", "pagerank", "pagerank_rank",
            "community", "top3_links"
        ])

    G = nx.DiGraph()
    # add edges
    for _, r in e.iterrows():
        src = str(r["H_CODE"])
        dst = str(r["H2_CODE"])
        w = float(r["cnt"]) if pd.notna(r["cnt"]) else 0.0
        if w <= 0:
            continue
        G.add_edge(src, dst, weight=w)

    if G.number_of_nodes() == 0:
        return pd.DataFrame(columns=[
            "H_CODE", "in_degree_w", "out_degree_w", "pagerank", "pagerank_rank",
            "community", "top3_links"
        ])

    # weighted in/out degree (strength)
    in_w = {n: 0.0 for n in G.nodes()}
    out_w = {n: 0.0 for n in G.nodes()}
    for u, v, data in G.edges(data=True):
        w = float(data.get("weight", 0.0))
        out_w[u] += w
        in_w[v] += w

    # pagerank (weighted)
    try:
        pr = nx.pagerank(G, weight="weight")
    except Exception:
        pr = {n: 0.0 for n in G.nodes()}

    # community detection (undirected projection)
    und = nx.Graph()
    for u, v, data in G.edges(data=True):
        w = float(data.get("weight", 0.0))
        if und.has_edge(u, v):
            und[u][v]["weight"] += w
        else:
            und.add_edge(u, v, weight=w)

    comm_map = {n: -1 for n in G.nodes()}
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        comms = list(greedy_modularity_communities(und, weight="weight"))
        for cid, nodes in enumerate(comms):
            for n in nodes:
                comm_map[n] = cid
    except Exception:
        pass

    # top-3 connected hospitals (by total flow with neighbor)
    # build neighbor weights using e (hospital names available)
    code_to_name_src = dict(zip(e["H_CODE"].astype(str), e["H_NM"].astype(str)))
    code_to_name_dst = dict(zip(e["H2_CODE"].astype(str), e["H2_ORG_NAME"].astype(str)))
    code_to_name = {**code_to_name_dst, **code_to_name_src}

    neigh_w = {n: {} for n in G.nodes()}
    for _, r in e.iterrows():
        u = str(r["H_CODE"])
        v = str(r["H2_CODE"])
        w = float(r["cnt"]) if pd.notna(r["cnt"]) else 0.0
        if u not in neigh_w:
            neigh_w[u] = {}
        if v not in neigh_w:
            neigh_w[v] = {}
        neigh_w[u][v] = neigh_w[u].get(v, 0.0) + w
        neigh_w[v][u] = neigh_w[v].get(u, 0.0) + w  # undirected total

    top3_links = {}
    for n in G.nodes():
        items = sorted(neigh_w.get(n, {}).items(), key=lambda kv: kv[1], reverse=True)[:3]
        parts = []
        for nb, w in items:
            nm = code_to_name.get(nb, nb)
            parts.append(f"{nm}({int(w):,})")
        top3_links[n] = ", ".join(parts) if parts else "-"

    # rank pagerank descending
    pr_series = pd.Series(pr).sort_values(ascending=False)
    rank_map = {k: int(i + 1) for i, k in enumerate(pr_series.index)}

    out = pd.DataFrame({
        "H_CODE": list(G.nodes()),
        "in_degree_w": [in_w.get(n, 0.0) for n in G.nodes()],
        "out_degree_w": [out_w.get(n, 0.0) for n in G.nodes()],
        "pagerank": [pr.get(n, 0.0) for n in G.nodes()],
        "pagerank_rank": [rank_map.get(n, None) for n in G.nodes()],
        "community": [comm_map.get(n, -1) for n in G.nodes()],
        "top3_links": [top3_links.get(n, "-") for n in G.nodes()],
    })

    return out



def build_network_nodes(df_):
    """
    Build node metadata for transfer network from BOTH:
      - primary hospital columns (H_CODE, H_NM, region, latitude/longitude)
      - secondary hospital columns (H2_CODE, H2_ORG_NAME, H2_latitude/H2_longitude)
    This prevents in-degree from being "all zeros" when the dataset only stores the 2nd hospital as H2_*.
    """
    parts = []

    if set(["H_CODE", "latitude", "longitude"]).issubset(df_.columns):
        a = df_.dropna(subset=["H_CODE", "latitude", "longitude"]).copy()
        a["H_CODE"] = a["H_CODE"].astype(str)
        a["H_NM"] = a["H_NM"] if "H_NM" in a.columns else a["H_CODE"]
        if COL_REGION_HOSP in a.columns:
            a[COL_REGION_HOSP] = a[COL_REGION_HOSP]
        else:
            a[COL_REGION_HOSP] = np.nan
        a = a.rename(columns={"latitude": "lat", "longitude": "lon"})
        parts.append(a[["H_CODE", "H_NM", COL_REGION_HOSP, "lat", "lon"]])

    if set(["H2_CODE", "H2_latitude", "H2_longitude"]).issubset(df_.columns):
        b = df_.dropna(subset=["H2_CODE", "H2_latitude", "H2_longitude"]).copy()

        # Normalize H2_CODE using primary hospital name->code mapping when possible
        if "H_NM" in df_.columns and "H_CODE" in df_.columns and "H2_ORG_NAME" in b.columns:
            prim = df_.dropna(subset=["H_NM", "H_CODE"]).drop_duplicates(subset=["H_NM"])
            name_to_code = dict(zip(prim["H_NM"].astype(str), prim["H_CODE"].astype(str)))
            mapped = b["H2_ORG_NAME"].astype(str).map(name_to_code)
            b.loc[mapped.notna(), "H2_CODE"] = mapped[mapped.notna()].astype(str)

        b["H_CODE"] = b["H2_CODE"].astype(str)
        b["H_NM"] = b["H2_ORG_NAME"] if "H2_ORG_NAME" in b.columns else b["H_CODE"]
        # region for H2: try lookup by code in primary hospital mapping if possible
        if COL_REGION_HOSP in df_.columns and "H_CODE" in df_.columns:
            look = df_.dropna(subset=["H_CODE", COL_REGION_HOSP]).copy()
            reg_map = dict(zip(look["H_CODE"].astype(str), look[COL_REGION_HOSP]))
            b[COL_REGION_HOSP] = b["H_CODE"].map(reg_map)
        else:
            b[COL_REGION_HOSP] = np.nan
        b = b.rename(columns={"H2_latitude": "lat", "H2_longitude": "lon"})
        parts.append(b[["H_CODE", "H_NM", COL_REGION_HOSP, "lat", "lon"]])

    if not parts:
        return pd.DataFrame(columns=["H_CODE", "H_NM", COL_REGION_HOSP, "lat", "lon"])

    n = pd.concat(parts, ignore_index=True)
    # keep first non-null name/region, and average coords if multiple
    n["H_NM"] = n["H_NM"].astype(str)
    n = (
        n.groupby("H_CODE", as_index=False)
        .agg(
            H_NM=("H_NM", "first"),
            **{COL_REGION_HOSP: (COL_REGION_HOSP, "first")},
            lat=("lat", "mean"),
            lon=("lon", "mean"),
        )
    )
    return n


def make_fig_hospital(df_f, hosp_f, roles_f, scenario, layers, cnt_range, selected_hcode, choro_metric="injury_cnt"):
    show_hosp = "hosp" in (layers or [])
    show_edges = "edges" in (layers or [])

    if len(hosp_f):
        center_lat = float(hosp_f["lat"].mean())
        center_lon = float(hosp_f["lon"].mean())
    else:
        center_lat, center_lon = 36.3, 127.8
    zoom = 6.2

    fig = go.Figure()
    _add_mapbox_dummy(fig, center_lat, center_lon)

    # 시군구 단계구분도 (맨 아래 레이어; 마커/엣지는 위에 렌더)
    if "choro" in (layers or []):
        add_choropleth(fig, df_f, choro_metric)

    if len(hosp_f):
        h = hosp_f.merge(
            roles_f[["H_CODE", "local_ratio", "inflow_ratio", "transfer_ratio", "hub_score"]],
            on="H_CODE", how="left"
        ).fillna(0)
        if scenario == "C":
            cmet = compute_network_centrality(df_f)
            if len(cmet):
                h = h.merge(cmet, on="H_CODE", how="left")
            # fill missing
            for col in ["in_degree_w", "out_degree_w", "pagerank", "pagerank_rank", "community", "top3_links"]:
                if col not in h.columns:
                    h[col] = np.nan
            h["in_degree_w"] = pd.to_numeric(h["in_degree_w"], errors="coerce").fillna(0.0)
            h["out_degree_w"] = pd.to_numeric(h["out_degree_w"], errors="coerce").fillna(0.0)
            h["pagerank"] = pd.to_numeric(h["pagerank"], errors="coerce").fillna(0.0)
            h["pagerank_rank"] = pd.to_numeric(h["pagerank_rank"], errors="coerce")
            h["community"] = pd.to_numeric(h["community"], errors="coerce").fillna(-1).astype(int)
            h["top3_links"] = h["top3_links"].fillna("-").astype(str)

    else:
        h = hosp_f.copy()

    # ---- Nodes ----
    if show_hosp and len(h):
        hover = (
            "병원: " + h["H_NM"].astype(str)
            + "<br>권역: " + h[COL_REGION_HOSP].astype(str)
            + "<br>환자수: " + h["patient_cnt"].astype(int).astype(str)
            + "<br>로컬비중: " + (h.get("local_ratio", 0) * 100).round(1).astype(str) + "%"
            + "<br>유입비중: " + (h.get("inflow_ratio", 0) * 100).round(1).astype(str) + "%"
            + "<br>전원비중: " + (h.get("transfer_ratio", 0) * 100).round(1).astype(str) + "%"
            + "<br>Hub score: " + h.get("hub_score", 0).round(2).astype(str)
        )

        if scenario == "C":
            # Network metrics hover
            hover = (
                hover
                + "<br><br><b>[Network Centrality]</b>"
                + "<br>In-degree(w): " + h["in_degree_w"].round(0).astype(int).astype(str)
                + " / Out-degree(w): " + h["out_degree_w"].round(0).astype(int).astype(str)
                + "<br>PageRank rank: " + h["pagerank_rank"].fillna(-1).astype(int).astype(str)
                + "<br>Community ID: " + h["community"].astype(int).astype(str)
                + "<br>주요 연결 병원 Top-3: " + h["top3_links"].astype(str)
            )

        hover = hover + "<br><br><b>클릭하면 우측 샷 갱신</b>"

        base_size = np.clip(np.sqrt(h["patient_cnt"]), 4, 28)
        hi_size = np.clip(base_size * 1.4, 6, 36)
        lo_size = np.clip(base_size * 0.7, 3, 18)

        # Scenario logic
        if scenario == "B":
            thr = h["hub_score"].quantile(0.8) if len(h) else 0
            mask_hi = h["hub_score"] >= thr
        elif scenario == "C":
            # Network modeling view: show all nodes, but encode centrality
            mask_hi = np.ones(len(h), dtype=bool)
        else:
            mask_hi = np.ones(len(h), dtype=bool)

        if scenario == "C":
            # Visual encodings (following your slide concept):
            # - Node size: PageRank (bigger = higher PR)
            # - Border thickness: PageRank bins (thicker border for higher PR)
            # - Node color: Community ID

            pr = h["pagerank"].to_numpy(dtype=float)
            pr_max = np.nanmax(pr) if np.isfinite(np.nanmax(pr)) else 0.0  # ✅ nanamax -> nanmax

            if pr_max > 0:
                pr_norm = pr / pr_max
            else:
                pr_norm = np.zeros_like(pr)

            # PageRank 기반 노드 크기 크게 (10~70)
            inner_size = 10 + 55 * (pr_norm ** 0.5)
            inner_size = np.clip(inner_size, 5, 22)

            # NOTE: Scattermapbox marker.line is not supported in some Plotly versions.
            # We draw border via 2-layer markers (outer black + inner colored).
            # Border thickness is approximated by PageRank binning (4 groups).
            pr_pos = pr[np.isfinite(pr) & (pr > 0)]
            if pr_pos.size >= 4:
                qs = np.quantile(pr_pos, [0.25, 0.50, 0.75])
                pr_bin = np.digitize(np.nan_to_num(pr, nan=-1.0), bins=qs, right=True)  # 0..3
            elif pr_pos.size > 0:
                pr_bin = np.where(pr > 0, 3, 0)
            else:
                pr_bin = np.zeros_like(pr, dtype=int)

            # 테두리 두께(outer-inner 차이)도 좀 더 키워도 됨
            outer_delta_by_bin = {0: 1.0, 1: 2.5, 2: 4.0, 3: 5.5}

            palette = [
                "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
                "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
            ]
            comm_ids = h["community"].astype(int).to_numpy()
            comm_colors = [palette[c % len(palette)] if c >= 0 else "#999999" for c in comm_ids]

            for b in [0, 1, 2, 3]:
                msk = (pr_bin == b)
                if not np.any(msk):
                    continue

                outer_delta = outer_delta_by_bin[b]

                # Outer (border) trace: hover disabled
                fig.add_trace(go.Scattermapbox(
                    lat=h.loc[msk, "lat"],
                    lon=h.loc[msk, "lon"],
                    mode="markers",
                    marker=dict(
                        size=(inner_size[msk] + outer_delta),  
                        color="#111111",
                        opacity=0.9,
                    ),
                    hoverinfo="skip",
                    showlegend=False,
                ))

                # Inner (fill) trace: actual hover
                fig.add_trace(go.Scattermapbox(
                    lat=h.loc[msk, "lat"],
                    lon=h.loc[msk, "lon"],
                    mode="markers",
                    marker=dict(
                        size=inner_size[msk],  
                        color=[comm_colors[i] for i in np.where(msk)[0]],
                        opacity=0.85,
                    ),
                    hovertext=hover[msk],
                    hoverinfo="text",
                    customdata=h.loc[msk, "H_CODE"],
                    name=f"Hospitals (Network, PR bin {b})",
                    showlegend=False,
                ))


        else:
            if scenario == "B":
                lo = h[~mask_hi]
                if len(lo):
                    fig.add_trace(go.Scattermapbox(
                        lat=lo["lat"], lon=lo["lon"],
                        mode="markers",
                        marker=dict(size=lo_size[~mask_hi], opacity=0.25),
                        hovertext=hover[~mask_hi],
                        hoverinfo="text",
                        customdata=lo["H_CODE"],
                        name="Hospitals (Others)"
                    ))

                hi = h[mask_hi]
                if len(hi):
                    fig.add_trace(go.Scattermapbox(
                        lat=hi["lat"], lon=hi["lon"],
                        mode="markers",
                        marker=dict(size=hi_size[mask_hi], opacity=0.95),
                        hovertext=hover[mask_hi],
                        hoverinfo="text",
                        customdata=hi["H_CODE"],
                        name="Hospitals (Highlighted)"
                    ))
            else:
                fig.add_trace(go.Scattermapbox(
                    lat=h["lat"], lon=h["lon"],
                    mode="markers",
                    marker=dict(size=base_size, opacity=0.85),
                    hovertext=hover,
                    hoverinfo="text",
                    customdata=h["H_CODE"],
                    name="Hospitals"
                ))

    # ---- Edges ----
    # NOTE: Plotly mapbox becomes very slow / can freeze when we add one trace per edge.
    # We therefore draw *all* edges in a few aggregated traces:
    #   1) one lines trace (with None separators),
    #   2) one invisible midpoint-marker trace for hover tooltips,
    #   3) one arrowhead trace (optional) for direction.
    if show_edges:
        lo, hi = (cnt_range or [1, 10**9])
        lo = int(lo); hi = int(hi)
        e = transfer_edges(df_f, min_cnt=1)
        e = e[(e["cnt"] >= lo) & (e["cnt"] <= hi)].copy()
        e = e[(e["cnt"] >= lo) & (e["cnt"] <= hi)].copy()
        if len(e):
            lat_lines, lon_lines = [], []
            mid_lat, mid_lon, mid_hover = [], [], []
            lat_arr, lon_arr = [], []

            for _, r in e.iterrows():
                src_lat, src_lon = float(r["src_lat"]), float(r["src_lon"])
                dst_lat, dst_lon = float(r["dst_lat"]), float(r["dst_lon"])

                lat_lines += [src_lat, dst_lat, None]
                lon_lines += [src_lon, dst_lon, None]

                avg_tm = r.get("avg_transfer_min", np.nan)
                tm_txt = f"{avg_tm:.1f}분" if pd.notna(avg_tm) else "NA"
                hover_txt = (
                    f'{r["H_NM"]} → {r["H2_ORG_NAME"]}'
                    f'<br>전원 건수: {int(r["cnt"]):,}'
                    f'<br>평균 전원시간(AD→H2): {tm_txt}'
                )
                mid_lat.append((src_lat + dst_lat) / 2.0)
                mid_lon.append((src_lon + dst_lon) / 2.0)
                mid_hover.append(hover_txt)

                segs = _arrowhead_segments(src_lat, src_lon, dst_lat, dst_lon, head_len=0.08, head_angle_deg=25)
                for (alat, alon) in segs:
                    lat_arr += [alat[0], alat[1], None]
                    lon_arr += [alon[0], alon[1], None]

            fig.add_trace(go.Scattermapbox(
                lat=lat_lines,
                lon=lon_lines,
                mode="lines",
                line=dict(width=2),
                hoverinfo="skip",
                showlegend=False,
                name="Transfer"
            ))

            fig.add_trace(go.Scattermapbox(
                lat=mid_lat,
                lon=mid_lon,
                mode="markers",
                marker=dict(size=8, opacity=0.0),
                hovertext=mid_hover,
                hoverinfo="text",
                showlegend=False,
                name="Transfer Hover"
            ))

            fig.add_trace(go.Scattermapbox(
                lat=lat_arr,
                lon=lon_arr,
                mode="lines",
                line=dict(width=2),
                hoverinfo="skip",
                showlegend=False,
                name="Arrow"
            ))

    fig.update_layout(
        mapbox=dict(style="carto-positron", center=dict(lat=center_lat, lon=center_lon), zoom=zoom),
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(orientation="h"),
    )
    return fig


def make_fig_region(df_f, region_f, rmet_f, scenario, layers, cnt_range, selected_region, choro_metric="injury_cnt"):
    show_regions = "hosp" in (layers or [])
    show_edges = "edges" in (layers or [])

    if len(region_f):
        center_lat = float(region_f["lat"].mean())
        center_lon = float(region_f["lon"].mean())
    else:
        center_lat, center_lon = 36.3, 127.8
    zoom = 6.2

    fig = go.Figure()
    _add_mapbox_dummy(fig, center_lat, center_lon)

    # 시군구 단계구분도 (맨 아래 레이어)
    if "choro" in (layers or []):
        add_choropleth(fig, df_f, choro_metric)

    reg = region_f.merge(rmet_f[[COL_REGION_HOSP, "local_ratio", "inflow_ratio", "transfer_ratio", "hub_score"]],
                         on=COL_REGION_HOSP, how="left").fillna(0)

    if show_regions and len(reg):
        hover = (
            "지역: " + reg[COL_REGION_HOSP].astype(str)
            + "<br>환자수: " + reg["total_patients"].astype(int).astype(str)
            + "<br>병원수: " + reg["hosp_cnt"].astype(int).astype(str)
            + "<br>로컬비중: " + (reg["local_ratio"] * 100).round(1).astype(str) + "%"
            + "<br>유입비중: " + (reg["inflow_ratio"] * 100).round(1).astype(str) + "%"
            + "<br>전원비중: " + (reg["transfer_ratio"] * 100).round(1).astype(str) + "%"
            + "<br>Hub score: " + reg["hub_score"].round(2).astype(str)
            + "<br><b>클릭하면 우측 지역 샷 갱신</b>"
        )
        size = np.clip(np.sqrt(reg["total_patients"]), 8, 55)

        fig.add_trace(go.Scattermapbox(
            lat=reg["lat"], lon=reg["lon"],
            mode="markers",
            marker=dict(size=size, opacity=0.85),
            hovertext=hover,
            hoverinfo="text",
            customdata=reg[COL_REGION_HOSP],  # region name
            name="Regions"
        ))

    # Region edges (aggregated)
    if show_edges and len(reg):
        hosp_map = agg_hospitals(df_f)[["H_CODE", COL_REGION_HOSP]].dropna()
        hosp_to_region = dict(zip(hosp_map["H_CODE"].astype(str), hosp_map[COL_REGION_HOSP].astype(str)))
        region_centroids = reg.set_index(COL_REGION_HOSP)[["lat", "lon"]].to_dict("index")

        lo, hi = (cnt_range or [1, 10**9])
        lo = int(lo); hi = int(hi)

        e = transfer_edges_region(
            df_f,
            min_cnt=1,
            region_centroids=region_centroids,
            hosp_to_region=hosp_to_region
        )
        if len(e):
            lat_lines, lon_lines = [], []
            mid_lat, mid_lon, mid_hover = [], [], []
            lat_arr, lon_arr = [], []

            for _, r in e.iterrows():
                src_lat, src_lon = float(r["src_lat"]), float(r["src_lon"])
                dst_lat, dst_lon = float(r["dst_lat"]), float(r["dst_lon"])

                lat_lines += [src_lat, dst_lat, None]
                lon_lines += [src_lon, dst_lon, None]

                avg_tm = r.get("avg_transfer_min", np.nan)
                tm_txt = f"{avg_tm:.1f}분" if pd.notna(avg_tm) else "NA"
                hover_txt = (
                    f'{r["SRC_REGION"]} → {r["DST_REGION"]}'
                    f'<br>전원 건수: {int(r["cnt"]):,}'
                    f'<br>평균 전원시간(AD→H2): {tm_txt}'
                )
                mid_lat.append((src_lat + dst_lat) / 2.0)
                mid_lon.append((src_lon + dst_lon) / 2.0)
                mid_hover.append(hover_txt)

                segs = _arrowhead_segments(src_lat, src_lon, dst_lat, dst_lon, head_len=0.10, head_angle_deg=25)
                for (alat, alon) in segs:
                    lat_arr += [alat[0], alat[1], None]
                    lon_arr += [alon[0], alon[1], None]

            fig.add_trace(go.Scattermapbox(
                lat=lat_lines,
                lon=lon_lines,
                mode="lines",
                line=dict(width=2),
                hoverinfo="skip",
                showlegend=False,
                name="Region Transfer"
            ))
            fig.add_trace(go.Scattermapbox(
                lat=mid_lat,
                lon=mid_lon,
                mode="markers",
                marker=dict(size=10, opacity=0.0),
                hovertext=mid_hover,
                hoverinfo="text",
                showlegend=False,
                name="Region Transfer Hover"
            ))
            fig.add_trace(go.Scattermapbox(
                lat=lat_arr,
                lon=lon_arr,
                mode="lines",
                line=dict(width=2),
                hoverinfo="skip",
                showlegend=False,
                name="Arrow"
            ))

    # selected region zoom
    if selected_region and len(reg) and selected_region in set(reg[COL_REGION_HOSP]):
        row = reg[reg[COL_REGION_HOSP] == selected_region].iloc[0]
        fig.add_trace(go.Scattermapbox(
            lat=[row["lat"]], lon=[row["lon"]],
            mode="markers",
            marker=dict(size=60, opacity=0.95),
            hoverinfo="text",
            hovertext=f"SELECTED: {row[COL_REGION_HOSP]}",
            showlegend=False
        ))
        center_lat, center_lon = float(row["lat"]), float(row["lon"])
        zoom = 7.2
#open-street-map
    fig.update_layout(
        mapbox=dict(style="carto-positron", center=dict(lat=center_lat, lon=center_lon), zoom=zoom),
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(orientation="h"),
    )
    return fig


# -----------------------------
# 7) Dash layout
# -----------------------------

# -----------------------------
# 1-3) Patient state transition Sankey (Graph Panel)
# -----------------------------
def _pick_col(df_, candidates):
    for c in candidates:
        if c in df_.columns:
            return c
    return None


def build_patient_state_sankey(df_subset, title="Patient State Transitions", top_n=8):
    """
    Build a Sankey diagram for patient state transitions.

    States (per your slide):
      - Pre: CLSS (or CLASS/CLSS)
      - ER1: ER_RESULT_KOR
      - AD1: AD_RESULT_KOR
      - ER2: H2_ER_RESULT_KOR
      - AD2: H2_ADM_RESULT_KOR

    Logic:
      - Always build Pre -> ER1 -> AD1
      - If transferred (H2_CODE exists or any H2 result exists):
            AD1 -> ER2 -> AD2
        else:
            AD1 -> END(No Transfer)

    We bin rare categories into "Other" per stage to keep the Sankey readable.
    """
    if df_subset is None or len(df_subset) == 0 or "EMS_SN" not in df_subset.columns:
        fig = go.Figure()
        fig.update_layout(title=title, height=260, margin=dict(l=10, r=10, t=30, b=10))
        fig.add_annotation(text="No data", x=0.5, y=0.5, showarrow=False)
        return fig

    # column resolution (robust)
    col_pre = _pick_col(df_subset, ["CLASSES_KOR", "CLASSES", "H_CLASS", "H_CLSS"])
    col_er1 = _pick_col(df_subset, ["ER_RESULT_KOR", "ER_RESULT"])
    col_ad1 = _pick_col(df_subset, ["AD_RESULT_KOR", "AD_RESULT"])
    col_er2 = _pick_col(df_subset, ["H2_ER_RESULT_KOR", "H2_ER_RESULT"])
    col_ad2 = _pick_col(df_subset, ["H2_ADM_RESULT_KOR", "H2_ADM_RESULT", "H2_AD_RESULT_KOR", "H2_AD_RESULT"])
    col_h2code = _pick_col(df_subset, ["H2_CODE", "H2_H_CODE"])

    # if essential cols missing, fallback
    if col_pre is None and col_er1 is None and col_ad1 is None:
        fig = go.Figure()
        fig.update_layout(title=title, height=260, margin=dict(l=10, r=10, t=30, b=10))
        fig.add_annotation(text="Missing required columns", x=0.5, y=0.5, showarrow=False)
        return fig

    cols = [c for c in [col_pre, col_er1, col_ad1, col_er2, col_ad2, col_h2code] if c is not None]
    d = df_subset[["EMS_SN"] + cols].copy()

    # patient-level (one row per patient)
    agg_map = {c: "first" for c in cols}
    pt = d.groupby("EMS_SN", as_index=False).agg(agg_map)

    # transferred flag
    if col_h2code is not None:
        transferred = pt[col_h2code].notna()
    else:
        transferred = False
        if col_er2 is not None:
            transferred = transferred | pt[col_er2].notna()
        if col_ad2 is not None:
            transferred = transferred | pt[col_ad2].notna()

    # Helper to keep only top_n categories per stage
    def top_map(series, top_n_):
        vc = series.value_counts(dropna=True)
        top_vals = set(vc.head(top_n_).index.astype(str).tolist())
        def f(x):
            if pd.isna(x):
                return "Unknown"
            s = str(x)
            return s if s in top_vals else "Other"
        return series.map(f)

    # Build stage labels with prefixes to avoid node merging across stages
    def prefixed(prefix, series):
        return prefix + ": " + series.astype(str)

    pre_s = top_map(pt[col_pre], top_n) if col_pre is not None else pd.Series(["Unknown"] * len(pt))
    er1_s = top_map(pt[col_er1], top_n) if col_er1 is not None else pd.Series(["Unknown"] * len(pt))
    ad1_s = top_map(pt[col_ad1], top_n) if col_ad1 is not None else pd.Series(["Unknown"] * len(pt))

    # For ER2/AD2, compute top categories among transferred patients only (less clutter)
    if col_er2 is not None:
        er2_raw = pt.loc[transferred, col_er2]
        er2_map = top_map(er2_raw, top_n).reindex(pt.index, fill_value="Unknown")
        er2_s = er2_map
    else:
        er2_s = pd.Series(["Unknown"] * len(pt))

    if col_ad2 is not None:
        ad2_raw = pt.loc[transferred, col_ad2]
        ad2_map = top_map(ad2_raw, top_n).reindex(pt.index, fill_value="Unknown")
        ad2_s = ad2_map
    else:
        ad2_s = pd.Series(["Unknown"] * len(pt))

    pre_lbl = prefixed("Pre", pre_s)
    er1_lbl = prefixed("ER1", er1_s)
    ad1_lbl = prefixed("AD1", ad1_s)
    er2_lbl = prefixed("ER2", er2_s)
    ad2_lbl = prefixed("AD2", ad2_s)

    end_lbl = pd.Series(["END: No Transfer"] * len(pt))

    edges = []

    # Pre -> ER1
    edges.append(pd.DataFrame({"src": pre_lbl, "dst": er1_lbl}))
    # ER1 -> AD1
    edges.append(pd.DataFrame({"src": er1_lbl, "dst": ad1_lbl}))

    # AD1 -> (ER2 or END)
    edges.append(pd.DataFrame({"src": ad1_lbl[~transferred], "dst": end_lbl[~transferred]}))
    edges.append(pd.DataFrame({"src": ad1_lbl[transferred], "dst": er2_lbl[transferred]}))
    # ER2 -> AD2 (only transferred)
    edges.append(pd.DataFrame({"src": er2_lbl[transferred], "dst": ad2_lbl[transferred]}))

    ed = pd.concat(edges, ignore_index=True)
    ed = ed.dropna()
    if len(ed) == 0:
        fig = go.Figure()
        fig.update_layout(title=title, height=260, margin=dict(l=10, r=10, t=30, b=10))
        fig.add_annotation(text="No transitions", x=0.5, y=0.5, showarrow=False)
        return fig

    cnt = ed.groupby(["src", "dst"], as_index=False).size().rename(columns={"size": "value"})

    # nodes
    labels = pd.Index(pd.unique(cnt[["src", "dst"]].values.ravel("K"))).tolist()
    idx = {lab: i for i, lab in enumerate(labels)}

    sources = cnt["src"].map(idx).tolist()
    targets = cnt["dst"].map(idx).tolist()
    values = cnt["value"].astype(int).tolist()

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=12,
            thickness=14,
            label=labels,
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
        )
    ))
    fig.update_layout(
        title=title,
        height=260,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig


# =========================================================
# Design system (PRD 7. 디자인 가이드)
# =========================================================
CLR = {
    "primary": "#10B981",   # Emerald
    "navy": "#0F172A",      # Deep navy
    "slate": "#1E293B",
    "danger": "#EF4444",
    "amber": "#F59E0B",
    "bg": "#F8FAFC",
    "panel": "#FFFFFF",
    "border": "#E2E8F0",
    "text": "#0F172A",
    "muted": "#64748B",
}

CARD_STYLE = {
    "background": CLR["panel"],
    "borderRadius": "12px",
    "border": f"1px solid {CLR['border']}",
    "boxShadow": "0 1px 3px rgba(15,23,42,0.06), 0 1px 2px rgba(15,23,42,0.04)",
    "padding": "14px 16px",
    "boxSizing": "border-box",
}


# =========================================================
# 시군구 단계구분도 (PRD 필수기능 2: Web-GIS Choropleth)
# =========================================================
SIDO_SHORT = {
    "서울특별시": "서울", "부산광역시": "부산", "대구광역시": "대구", "인천광역시": "인천",
    "광주광역시": "광주", "대전광역시": "대전", "울산광역시": "울산", "세종특별자치시": "세종",
    "세종시": "세종", "경기도": "경기", "강원도": "강원", "강원특별자치도": "강원",
    "충청북도": "충북", "충청남도": "충남", "전라북도": "전북", "전북특별자치도": "전북",
    "전라남도": "전남", "경상북도": "경북", "경상남도": "경남", "제주특별자치도": "제주", "제주도": "제주",
}


def _sido_short(s):
    return SIDO_SHORT.get(str(s).strip(), str(s).strip())


# 데이터→GeoJSON 키 보정 (행정구역 개편 대응)
SGG_ALIAS = {
    "경기_부천시 원미구": "경기_부천시", "경기_부천시 소사구": "경기_부천시", "경기_부천시 오정구": "경기_부천시",
    "대구_군위군": "경북_군위군",
}

_GEOJSON_PATH = next((p for p in ["data/sigungu_merged.geojson", "sigungu_merged.geojson"]
                      if os.path.exists(p)), None)
SIGUNGU_GEOJSON = None
if _GEOJSON_PATH:
    try:
        with open(_GEOJSON_PATH, encoding="utf-8") as _fp:
            SIGUNGU_GEOJSON = json.load(_fp)
    except Exception:
        SIGUNGU_GEOJSON = None

_SGG_NAME_MAP = ({f["properties"]["join_key"]: f["properties"]["sigungu"]
                  for f in SIGUNGU_GEOJSON["features"]} if SIGUNGU_GEOJSON else {})

CHORO_METRICS = {
    "injury_cnt": ("손상 환자수", "Reds", "명"),
    "transfer_rate": ("전원율", "OrRd", "%"),
    "severe_rate": ("중증손상률(ISS≥16)", "OrRd", "%"),
    "elderly_rate": ("고령자손상률(65+)", "YlOrBr", "%"),
}


def _data_join_key(sido, sigu):
    k = f"{_sido_short(sido)}_{str(sigu).strip()}"
    return SGG_ALIAS.get(k, k)


def sigungu_metrics(df_f):
    """거주지 시군구(ADDS_SIDO+ADDS_SIGU)별 손상/전원/중증/고령 지표."""
    cols = {"ADDS_SIDO", "ADDS_SIGU", "EMS_SN"}
    if SIGUNGU_GEOJSON is None or not cols.issubset(df_f.columns) or len(df_f) == 0:
        return pd.DataFrame(columns=["join_key", "injury_cnt", "transfer_rate", "severe_rate", "elderly_rate"])

    d = df_f.dropna(subset=["ADDS_SIDO", "ADDS_SIGU"]).copy()
    d = d[d["ADDS_SIGU"].astype(str).str.strip() != "미상"]
    if len(d) == 0:
        return pd.DataFrame(columns=["join_key", "injury_cnt", "transfer_rate", "severe_rate", "elderly_rate"])
    d["join_key"] = [_data_join_key(s, g) for s, g in zip(d["ADDS_SIDO"], d["ADDS_SIGU"])]

    inj = d.groupby("join_key")["EMS_SN"].nunique()
    tr = (d[d["H2_CODE"].notna() & (d["H2_CODE"].astype(str).str.lower() != "nan")]
          .groupby("join_key")["EMS_SN"].nunique() if "H2_CODE" in d.columns else pd.Series(dtype=float))
    if "ISS" in d.columns:
        iss = pd.to_numeric(d["ISS"], errors="coerce")
        sev = d[iss >= 16].groupby("join_key")["EMS_SN"].nunique()
    else:
        sev = pd.Series(dtype=float)
    eld = (d[d["H_AGE_NUM"] >= 65].groupby("join_key")["EMS_SN"].nunique()
           if "H_AGE_NUM" in d.columns else pd.Series(dtype=float))

    out = pd.DataFrame({"injury_cnt": inj})
    out["transfer_rate"] = (tr / inj * 100)
    out["severe_rate"] = (sev / inj * 100)
    out["elderly_rate"] = (eld / inj * 100)
    return out.fillna(0).reset_index()


def add_choropleth(fig, df_f, metric="injury_cnt"):
    """지도 figure에 시군구 단계구분도 레이어 추가 (맨 아래 레이어)."""
    if SIGUNGU_GEOJSON is None:
        return
    m = sigungu_metrics(df_f)
    if len(m) == 0:
        return
    label, scale, unit = CHORO_METRICS.get(metric, CHORO_METRICS["injury_cnt"])
    m["name"] = m["join_key"].map(_SGG_NAME_MAP).fillna(m["join_key"])
    z = m[metric].astype(float)
    hov = ("<b>" + m["name"].astype(str) + "</b><br>"
           + label + ": " + z.round(1).astype(str) + unit
           + "<br>손상 환자수: " + m["injury_cnt"].astype(int).astype(str) + "명<extra></extra>")
    fig.add_trace(go.Choroplethmapbox(
        geojson=SIGUNGU_GEOJSON,
        locations=m["join_key"],
        featureidkey="properties.join_key",
        z=z,
        colorscale=scale,
        zmin=float(z.min()) if len(z) else 0,
        zmax=float(np.nanpercentile(z, 97)) if len(z) else 1,
        marker=dict(line=dict(width=0.3, color="rgba(255,255,255,0.55)"), opacity=0.62),
        colorbar=dict(title=dict(text=label, side="right"), thickness=12, len=0.55,
                      x=0.0, xanchor="left", y=0.5),
        hovertemplate=hov,
        name=f"시군구 {label}",
        showscale=True,
    ))


# =========================================================
# KPI cards + sparklines (PRD 필수기능 5)
# =========================================================
def _monthly_series(df_f, mask=None):
    """ER_D(YYYYMMDD) 기준 월별 고유 환자수 시계열(최근 12개월)."""
    if "ER_D" not in df_f.columns or "EMS_SN" not in df_f.columns or len(df_f) == 0:
        return []
    d = df_f if mask is None else df_f[mask]
    if len(d) == 0:
        return []
    ym = pd.to_numeric(d["ER_D"], errors="coerce")
    ym = (ym // 100).dropna().astype("int64")  # YYYYMM
    tmp = pd.DataFrame({"ym": ym.values, "sn": d.loc[ym.index, "EMS_SN"].values})
    g = tmp.groupby("ym")["sn"].nunique().sort_index()
    return g.tail(12).tolist()


def build_sparkline(values, color):
    fig = go.Figure()
    if values and len(values) >= 2:
        fig.add_trace(go.Scatter(
            y=values, mode="lines", line=dict(color=color, width=2),
            fill="tozeroy", fillcolor=color.replace(")", ",0.12)").replace("rgb", "rgba") if color.startswith("rgb") else "rgba(16,185,129,0.10)",
            hoverinfo="skip",
        ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=38,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
    )
    return fig


def compute_kpis(df_f):
    out = {"total": 0, "transfer_rate": 0.0, "severe_rate": 0.0, "elderly_rate": 0.0, "vuln": 0}
    if df_f is None or len(df_f) == 0 or "EMS_SN" not in df_f.columns:
        return out
    total = int(df_f["EMS_SN"].nunique())
    out["total"] = total
    if total == 0:
        return out

    if "H2_CODE" in df_f.columns:
        tr = df_f[df_f["H2_CODE"].notna() & (df_f["H2_CODE"].astype(str).str.lower() != "nan")]["EMS_SN"].nunique()
        out["transfer_rate"] = tr / total * 100

    if "ISS" in df_f.columns:
        iss = pd.to_numeric(df_f["ISS"], errors="coerce")
        sev = df_f.loc[iss >= 16, "EMS_SN"].nunique()
        out["severe_rate"] = sev / total * 100

    if "H_AGE_NUM" in df_f.columns:
        eld = df_f.loc[df_f["H_AGE_NUM"] >= 65, "EMS_SN"].nunique()
        out["elderly_rate"] = eld / total * 100

    try:
        rmet = region_metrics(df_f)
        out["vuln"] = int((rmet["transfer_ratio"] >= 0.20).sum()) if len(rmet) else 0
    except Exception:
        out["vuln"] = 0
    return out


def build_kpi_cards(df_f):
    k = compute_kpis(df_f)

    # 스파크라인용 월별 시계열
    spark_total = _monthly_series(df_f)
    spark_sev = None
    spark_eld = None
    spark_tr = None
    if "ISS" in df_f.columns and len(df_f):
        spark_sev = _monthly_series(df_f, pd.to_numeric(df_f["ISS"], errors="coerce") >= 16)
    if "H_AGE_NUM" in df_f.columns and len(df_f):
        spark_eld = _monthly_series(df_f, df_f["H_AGE_NUM"] >= 65)
    if "H2_CODE" in df_f.columns and len(df_f):
        spark_tr = _monthly_series(df_f, df_f["H2_CODE"].notna() & (df_f["H2_CODE"].astype(str).str.lower() != "nan"))

    cards = [
        ("전체 손상 환자 / 전원율", f"{k['total']:,}", f"전원율 {k['transfer_rate']:.1f}%", CLR["primary"], spark_total or spark_tr),
        ("중증 손상률 (ISS≥16)", f"{k['severe_rate']:.1f}%", "중증 손상 환자 비중", CLR["danger"], spark_sev),
        ("고령자 손상률 (65세+)", f"{k['elderly_rate']:.1f}%", "고령 손상 환자 비중", CLR["amber"], spark_eld),
    ]

    children = []
    for title, value, sub, color, spark in cards:
        children.append(html.Div(
            style={**CARD_STYLE, "flex": "1 1 0", "minWidth": "0",
                   "borderTop": f"3px solid {color}", "display": "flex", "flexDirection": "column"},
            children=[
                html.Div(title, style={"fontSize": "12px", "color": CLR["muted"], "fontWeight": 600,
                                       "whiteSpace": "nowrap", "overflow": "hidden", "textOverflow": "ellipsis"}),
                html.Div(value, style={"fontSize": "26px", "fontWeight": 800, "color": CLR["text"],
                                       "lineHeight": "1.1", "marginTop": "2px"}),
                html.Div(sub, style={"fontSize": "11px", "color": CLR["muted"], "marginTop": "1px"}),
                dcc.Graph(figure=build_sparkline(spark, color),
                          config={"displayModeBar": False, "staticPlot": True},
                          style={"height": "38px", "marginTop": "6px"}),
            ],
        ))

    return html.Div(
        style={"display": "flex", "flexDirection": "row", "gap": "12px", "width": "100%"},
        children=children,
    )


# =========================================================
# 대화형 AI 분석 인터페이스 (PRD 필수기능 4 - OpenAI Function Calling)
# =========================================================
_OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# AI가 제어 가능한 필터 옵션(시스템 프롬프트 주입용) - 모듈 로드 시 채워짐
AI_FILTER_VOCAB = {"hosp_regions": [], "addr_regions": [], "classes": [], "sex": []}

AI_TOOLS = [{
    "type": "function",
    "function": {
        "name": "set_dashboard_filters",
        "description": "사용자의 자연어 요청에 따라 대시보드의 필터, 시나리오, 지도 레이어를 변경한다. 변경이 필요한 항목만 포함한다.",
        "parameters": {
            "type": "object",
            "properties": {
                "scenario": {
                    "type": "string",
                    "enum": ["A", "B", "C", "D", "E"],
                    "description": "A=전체보기, B=허브강조, C=네트워크 거점분석, D=지역 집계보기, E=전원 환자만",
                },
                "hosp_regions": {"type": "array", "items": {"type": "string"},
                                 "description": "내원 병원 권역 목록"},
                "addr_regions": {"type": "array", "items": {"type": "string"},
                                 "description": "환자 거주지 시도 목록"},
                "classes": {"type": "array", "items": {"type": "string"}, "description": "환자 유형(CLASSES_KOR)"},
                "sex": {"type": "array", "items": {"type": "string"}, "description": "성별 코드"},
                "age_min": {"type": "integer"}, "age_max": {"type": "integer"},
                "iss_min": {"type": "integer"}, "iss_max": {"type": "integer"},
                "highlight_transfer": {"type": "boolean",
                                       "description": "전원율/전원 흐름을 강조할 때 true (시나리오 E + 전원 연결선 표시)"},
                "show_edges": {"type": "boolean", "description": "전원 연결선 레이어 표시 여부"},
                "show_choropleth": {"type": "boolean",
                                    "description": "시군구 단계구분도(손상률 색상 채우기) 표시 여부"},
                "reset": {"type": "boolean", "description": "모든 필터 초기화 여부"},
            },
        },
    },
}, {
    "type": "function",
    "function": {
        "name": "query_data",
        "description": (
            "데이터에서 지역(권역/시도)별 손상·전원 통계를 실제로 계산해 표 또는 그래프로 보여준다. "
            "지역 비교(예: '강원과 제주 전원율 비교'), 순위(예: '전원율 높은 지역 Top 5'), "
            "특정 지표 조회 질문에 사용한다."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "dimension": {
                    "type": "string", "enum": ["region", "addr"],
                    "description": "region=병원 권역(H_REGION_KOR), addr=환자 거주지 시도. 기본 region",
                },
                "groups": {
                    "type": "array", "items": {"type": "string"},
                    "description": "비교할 지역명 목록(예: ['강원','제주']). 비우면 전체에서 상위 top_n",
                },
                "metric": {
                    "type": "string",
                    "enum": ["transfer_rate", "injury_count", "severe_rate", "elderly_rate", "avg_transfer_min"],
                    "description": "transfer_rate=전원율, injury_count=손상 환자수, severe_rate=중증손상률, elderly_rate=고령자손상률, avg_transfer_min=평균 전원시간(분)",
                },
                "chart": {"type": "string", "enum": ["bar", "table"], "description": "bar=막대그래프, table=표. 기본 bar"},
                "top_n": {"type": "integer", "description": "groups 미지정 시 상위 개수. 기본 10"},
            },
            "required": ["metric"],
        },
    },
}]


def _ai_system_prompt(state_summary: str) -> str:
    return (
        "당신은 '한국형 시군구 손상 정밀예방 플랫폼'의 데이터 분석 어시스턴트입니다.\n"
        "사용자는 보건정책 담당자, 응급의료 관제 담당자, 일반 시민입니다.\n"
        "역할: (1) 제공된 통계 요약을 근거로 한국어로 간결하고 실무적인 인사이트를 제시한다. "
        "(2) 사용자가 화면/필터 변경을 요청하면 set_dashboard_filters 함수를 호출한다. "
        "(3) 지역별 수치 비교·순위·통계 질문(예: '강원과 제주 전원율 비교')에는 반드시 query_data 함수를 호출해 "
        "실제 데이터를 계산하고, 반환된 수치를 근거로 답한다. 절대 수치를 임의로 지어내지 말 것.\n"
        "절대 개인정보(환자 식별정보)를 추측하거나 생성하지 말 것. 집계 통계만 다룬다.\n\n"
        f"선택 가능한 병원 권역: {AI_FILTER_VOCAB['hosp_regions']}\n"
        f"선택 가능한 거주지 시도: {AI_FILTER_VOCAB['addr_regions']}\n"
        f"선택 가능한 환자 유형: {AI_FILTER_VOCAB['classes']}\n"
        f"선택 가능한 성별 코드: {AI_FILTER_VOCAB['sex']}\n\n"
        f"[현재 대시보드 상태 요약]\n{state_summary}\n"
    )


def _tool_args_to_updates(args: dict) -> dict:
    """OpenAI 함수 인자 -> Dash 컴포넌트 값 업데이트 dict."""
    u = {}
    if args.get("reset"):
        return {
            "scenario": "A", "layers": ["hosp"], "filter_hosp_region": [],
            "filter_addr_region": [], "filter_classes": [], "filter_icd_detail": [],
            "filter_sex": [], "filter_age": [0, 100], "filter_iss": [0, 75],
        }

    if args.get("highlight_transfer"):
        u["scenario"] = "E"
        u["layers"] = ["hosp", "edges"]
    if "scenario" in args and args["scenario"]:
        u["scenario"] = args["scenario"]
    if "show_edges" in args:
        base = u.get("layers", ["hosp"])
        if args["show_edges"]:
            u["layers"] = list(dict.fromkeys(base + ["edges"]))
        else:
            u["layers"] = [x for x in base if x != "edges"] or ["hosp"]
    if "show_choropleth" in args:
        base = u.get("layers", ["hosp"])
        if args["show_choropleth"]:
            u["layers"] = list(dict.fromkeys(base + ["choro"]))
        else:
            u["layers"] = [x for x in base if x != "choro"] or ["hosp"]

    def _valid(vals, vocab):
        return [v for v in (vals or []) if v in vocab] or [v for v in (vals or [])]

    if "hosp_regions" in args:
        u["filter_hosp_region"] = _valid(args["hosp_regions"], AI_FILTER_VOCAB["hosp_regions"])
    if "addr_regions" in args:
        u["filter_addr_region"] = _valid(args["addr_regions"], AI_FILTER_VOCAB["addr_regions"])
    if "classes" in args:
        u["filter_classes"] = _valid(args["classes"], AI_FILTER_VOCAB["classes"])
    if "sex" in args:
        u["filter_sex"] = _valid([str(s) for s in args["sex"]], AI_FILTER_VOCAB["sex"])

    amin = args.get("age_min"); amax = args.get("age_max")
    if amin is not None or amax is not None:
        u["filter_age"] = [int(amin) if amin is not None else 0, int(amax) if amax is not None else 100]
    imin = args.get("iss_min"); imax = args.get("iss_max")
    if imin is not None or imax is not None:
        u["filter_iss"] = [int(imin) if imin is not None else 0, int(imax) if imax is not None else 75]

    return u


def _describe_updates(u: dict) -> str:
    if not u:
        return ""
    parts = []
    name = {"A": "전체보기", "B": "허브강조", "C": "네트워크 거점분석", "D": "지역 집계보기", "E": "전원 환자만"}
    if "scenario" in u:
        parts.append(f"시나리오={name.get(u['scenario'], u['scenario'])}")
    if "layers" in u:
        lyr = u["layers"]
        lbl = []
        if "edges" in lyr:
            lbl.append("전원 연결선")
        if "choro" in lyr:
            lbl.append("시군구 단계구분도")
        parts.append("레이어=" + (", ".join(lbl) if lbl else "마커만"))
    if u.get("filter_hosp_region"):
        parts.append(f"권역={', '.join(map(str, u['filter_hosp_region']))}")
    if u.get("filter_addr_region"):
        parts.append(f"거주지={', '.join(map(str, u['filter_addr_region']))}")
    if u.get("filter_classes"):
        parts.append(f"환자유형={', '.join(map(str, u['filter_classes']))}")
    if u.get("filter_sex"):
        parts.append(f"성별={', '.join(map(str, u['filter_sex']))}")
    if "filter_age" in u:
        parts.append(f"연령={u['filter_age'][0]}~{u['filter_age'][1]}")
    if "filter_iss" in u:
        parts.append(f"ISS={u['filter_iss'][0]}~{u['filter_iss'][1]}")
    return " · ".join(parts)


# ---- 데이터 질의(query_data) 백엔드 ----
QUERY_METRICS = {
    "transfer_rate": ("전원율", "%"),
    "injury_count": ("손상 환자수", "명"),
    "severe_rate": ("중증손상률(ISS≥16)", "%"),
    "elderly_rate": ("고령자손상률(65+)", "%"),
    "avg_transfer_min": ("평균 전원시간", "분"),
}


def _region_stat_table(df_q, col, metric):
    """지역(col)별 지표 계산 -> DataFrame[name, value, n]."""
    if col not in df_q.columns or "EMS_SN" not in df_q.columns:
        return pd.DataFrame(columns=["name", "value", "n"])
    d = df_q.dropna(subset=[col, "EMS_SN"]).copy()
    d = d[d[col].astype(str).str.strip() != "미상"]
    if len(d) == 0:
        return pd.DataFrame(columns=["name", "value", "n"])

    n = d.groupby(col)["EMS_SN"].nunique()

    if metric == "injury_count":
        val = n.astype(float)
    elif metric == "transfer_rate":
        tr = (d[d["H2_CODE"].notna() & (d["H2_CODE"].astype(str).str.lower() != "nan")]
              .groupby(col)["EMS_SN"].nunique() if "H2_CODE" in d.columns else pd.Series(dtype=float))
        val = tr.reindex(n.index).fillna(0) / n * 100
    elif metric == "severe_rate":
        if "ISS" in d.columns:
            iss = pd.to_numeric(d["ISS"], errors="coerce")
            sev = d[iss >= 16].groupby(col)["EMS_SN"].nunique()
        else:
            sev = pd.Series(dtype=float)
        val = sev.reindex(n.index).fillna(0) / n * 100
    elif metric == "elderly_rate":
        eld = (d[d["H_AGE_NUM"] >= 65].groupby(col)["EMS_SN"].nunique()
               if "H_AGE_NUM" in d.columns else pd.Series(dtype=float))
        val = eld.reindex(n.index).fillna(0) / n * 100
    elif metric == "avg_transfer_min":
        d["_tm"] = _compute_transfer_minutes(d)
        val = d.groupby(col)["_tm"].mean().reindex(n.index)
    else:
        val = n.astype(float)

    out = pd.DataFrame({
        "name": n.index.astype(str),
        "value": pd.to_numeric(val.reindex(n.index), errors="coerce").fillna(0).values,
        "n": n.values,
    })
    return out


def _run_data_query(df_q, args):
    """query_data 실행 -> (payload_for_llm, viz_dict|None)."""
    dim = args.get("dimension", "region")
    metric = args.get("metric", "transfer_rate")
    if metric not in QUERY_METRICS:
        metric = "transfer_rate"
    chart = args.get("chart", "bar")
    groups = args.get("groups") or []
    try:
        top_n = int(args.get("top_n", 10) or 10)
    except Exception:
        top_n = 10
    col = COL_REGION_ADDR if dim == "addr" else COL_REGION_HOSP
    label, unit = QUERY_METRICS[metric]
    dim_label = "거주지 시도" if dim == "addr" else "병원 권역"

    tbl = _region_stat_table(df_q, col, metric)
    if len(tbl) == 0:
        return {"ok": False, "message": "현재 필터 조건에서 계산할 데이터가 없습니다."}, None

    if groups:
        norm = {_sido_short(g) for g in groups}
        tbl["_short"] = tbl["name"].map(_sido_short)
        sel = tbl[tbl["_short"].isin(norm) | tbl["name"].isin(groups)]
        if len(sel) == 0:
            return ({"ok": False, "message": f"요청한 지역 {groups} 을(를) 데이터에서 찾지 못했습니다."}, None)
        tbl = sel.sort_values("value", ascending=False).drop(columns=["_short"])
    else:
        tbl = tbl.sort_values("value", ascending=False).head(top_n)

    tbl = tbl.reset_index(drop=True)
    nd = 0 if metric == "injury_count" else 1
    tbl["value_r"] = tbl["value"].round(nd)

    payload = {
        "ok": True, "dimension": dim_label, "metric": label, "unit": unit,
        "rows": [{"지역": r["name"], "값": (int(r["value_r"]) if nd == 0 else round(float(r["value_r"]), 1)),
                  "환자수": int(r["n"])} for _, r in tbl.iterrows()],
    }

    if chart == "table":
        viz = {
            "type": "table",
            "columns": [{"name": "지역", "id": "name"},
                        {"name": f"{label}({unit})", "id": "value"},
                        {"name": "환자수(명)", "id": "n"}],
            "data": [{"name": r["name"],
                      "value": (int(r["value_r"]) if nd == 0 else round(float(r["value_r"]), 1)),
                      "n": int(r["n"])} for _, r in tbl.iterrows()],
        }
    else:
        fig = go.Figure(go.Bar(
            x=tbl["name"].tolist(), y=tbl["value_r"].tolist(),
            marker_color=CLR["primary"],
            text=[f"{v:,.0f}" if nd == 0 else f"{v:.1f}" for v in tbl["value_r"]],
            textposition="outside", cliponaxis=False,
        ))
        fig.update_layout(
            title=dict(text=f"{label} 비교 · {dim_label}", font=dict(size=13, color=CLR["navy"])),
            margin=dict(l=10, r=10, t=38, b=28), height=270,
            yaxis_title=f"{label}({unit})", paper_bgcolor="white", plot_bgcolor="white",
            font=dict(size=11), showlegend=False,
        )
        viz = {"type": "bar", "figure": json.loads(fig.to_json())}

    return payload, viz


def ai_chat(history, user_msg, state_summary, df_q=None):
    """OpenAI 호출 -> (assistant_text, updates_dict, viz|None). 키 없거나 실패 시 graceful 메시지."""
    if not os.environ.get("OPENAI_API_KEY"):
        return ("⚠️ OpenAI API 키가 설정되지 않았습니다. 환경변수 OPENAI_API_KEY(또는 .env)를 설정한 뒤 다시 시도하세요.\n"
                "(설정 전에도 좌측 필터/지도는 정상 동작합니다.)", {}, None)
    if df_q is None:
        df_q = df
    try:
        from openai import OpenAI
        client = OpenAI()
        messages = [{"role": "system", "content": _ai_system_prompt(state_summary)}]
        for h in (history or [])[-8:]:
            messages.append({"role": h["role"], "content": h.get("content", "")})
        messages.append({"role": "user", "content": user_msg})

        resp = client.chat.completions.create(
            model=_OPENAI_MODEL, messages=messages,
            tools=AI_TOOLS, tool_choice="auto", temperature=0.3, max_tokens=700,
        )
        msg = resp.choices[0].message
        updates = {}
        viz = None
        text = msg.content or ""

        if msg.tool_calls:
            tool_results = []
            did_query = False
            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    a = json.loads(tc.function.arguments or "{}")
                except Exception:
                    a = {}
                if name == "set_dashboard_filters":
                    updates.update(_tool_args_to_updates(a))
                    tool_results.append({"role": "tool", "tool_call_id": tc.id,
                                         "content": "대시보드 필터를 적용했습니다."})
                elif name == "query_data":
                    payload, v = _run_data_query(df_q, a)
                    if v is not None:
                        viz = v
                    did_query = True
                    tool_results.append({"role": "tool", "tool_call_id": tc.id,
                                         "content": json.dumps(payload, ensure_ascii=False)})

            desc = _describe_updates(updates)

            if did_query:
                # 계산 결과를 LLM에 되돌려 자연어 답변 생성 (2차 호출)
                assistant_dump = {
                    "role": "assistant", "content": msg.content or "",
                    "tool_calls": [{"id": tc.id, "type": "function",
                                    "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                                   for tc in msg.tool_calls],
                }
                messages2 = messages + [assistant_dump] + tool_results
                resp2 = client.chat.completions.create(
                    model=_OPENAI_MODEL, messages=messages2, temperature=0.3, max_tokens=600,
                )
                text = resp2.choices[0].message.content or ""
                if desc:
                    text = (text + f"\n\n✅ 대시보드도 업데이트했습니다 — {desc}.").strip()
            else:
                confirm = f"✅ 대시보드를 업데이트했습니다 — {desc}." if desc else "✅ 요청을 반영했습니다."
                text = (text + "\n\n" + confirm).strip() if text else confirm

        if not text:
            text = "응답을 생성하지 못했습니다. 다시 질문해 주세요."
        return text, updates, viz
    except Exception as e:
        return (f"⚠️ AI 응답 생성 중 오류가 발생했습니다: {e}", {}, None)


def render_chat(history):
    """채팅 메시지 타임라인 렌더링."""
    if not history:
        return [html.Div(
            "안녕하세요! 지역 손상·전원 데이터에 대해 무엇이든 물어보세요.\n"
            "예) \"서울 65세 이상 고령자만 보여줘\", \"전원율 강조해줘\"",
            style={"color": CLR["muted"], "fontSize": "13px", "whiteSpace": "pre-line",
                   "padding": "12px", "lineHeight": "1.6"},
        )]
    bubbles = []
    for h in history:
        is_user = h["role"] == "user"
        item_children = [html.Div(
            h.get("content", ""),
            style={
                "maxWidth": "85%", "padding": "9px 12px", "borderRadius": "12px",
                "fontSize": "13px", "lineHeight": "1.55", "whiteSpace": "pre-line",
                "background": CLR["primary"] if is_user else "#F1F5F9",
                "color": "white" if is_user else CLR["text"],
                "border": "none" if is_user else f"1px solid {CLR['border']}",
            },
        )]

        # 데이터 질의 결과(차트/표) 렌더링
        viz = h.get("viz")
        if viz and not is_user:
            if viz.get("type") == "bar" and viz.get("figure"):
                item_children.append(html.Div(
                    dcc.Graph(figure=viz["figure"], config={"displayModeBar": False},
                              style={"height": "270px"}),
                    style={**CARD_STYLE, "width": "100%", "padding": "6px", "marginTop": "6px"},
                ))
            elif viz.get("type") == "table" and viz.get("data") is not None:
                item_children.append(html.Div(
                    dash_table.DataTable(
                        columns=viz["columns"], data=viz["data"],
                        page_size=10,
                        style_table={"overflowX": "auto"},
                        style_cell={"fontSize": "12px", "padding": "6px", "textAlign": "left"},
                        style_header={"fontWeight": "bold", "background": "#F1F5F9"},
                    ),
                    style={**CARD_STYLE, "width": "100%", "padding": "8px", "marginTop": "6px"},
                ))

        bubbles.append(html.Div(
            style={"display": "flex", "flexDirection": "column",
                   "alignItems": "flex-end" if is_user else "flex-start",
                   "marginBottom": "10px"},
            children=item_children,
        ))
    return bubbles


app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "NPIPP · 손상 정밀예방 플랫폼"

# 전역 폰트/스크롤바 스타일
app.index_string = """<!DOCTYPE html>
<html>
<head>
{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>
  * { font-family: 'Pretendard','Noto Sans KR',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; }
  body { margin:0; background:#F8FAFC; }
  ::-webkit-scrollbar { width:8px; height:8px; }
  ::-webkit-scrollbar-thumb { background:#CBD5E1; border-radius:4px; }
  ::-webkit-scrollbar-track { background:transparent; }
  .npipp-chip { background:#fff; border:1px solid #E2E8F0; border-radius:16px; padding:5px 11px;
    font-size:11.5px; color:#0F172A; cursor:pointer; white-space:nowrap; }
  .npipp-chip:hover { border-color:#10B981; color:#10B981; }
  .Select-control, .is-focused .Select-control { border-radius:8px !important; }
</style>
</head>
<body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body>
</html>"""

def _uniq_sorted(series):
    return sorted([x for x in series.dropna().unique().tolist() if str(x) != "nan"])

region_hosp_opts = _uniq_sorted(df[COL_REGION_HOSP])
region_addr_opts = _uniq_sorted(df[COL_REGION_ADDR]) if COL_REGION_ADDR in df.columns else []
classes_opts = _uniq_sorted(df["CLASSES_KOR"]) if "CLASSES_KOR" in df.columns else []
icd_detail_opts = _uniq_sorted(df["ICD_ST_DETAIL"]) if "ICD_ST_DETAIL" in df.columns else []
sex_opts = _uniq_sorted(df["H_SEX"]) if "H_SEX" in df.columns else []

# AI 어시스턴트가 참조할 필터 어휘 채우기
AI_FILTER_VOCAB["hosp_regions"] = [str(x) for x in region_hosp_opts]
AI_FILTER_VOCAB["addr_regions"] = [str(x) for x in region_addr_opts]
AI_FILTER_VOCAB["classes"] = [str(x) for x in classes_opts]
AI_FILTER_VOCAB["sex"] = [str(x) for x in sex_opts]

def make_hospital_options(hosp_df):
    if hosp_df is None or len(hosp_df) == 0:
        return []
    return [
        {"label": f"{r.H_NM} ({getattr(r, COL_REGION_HOSP)}) - {int(r.patient_cnt):,}명", "value": r.H_CODE}
        for r in hosp_df.sort_values("patient_cnt", ascending=False).itertuples()
    ]

def make_region_options(reg_df):
    if reg_df is None or len(reg_df) == 0:
        return []
    return [
        {"label": f"{getattr(r, COL_REGION_HOSP)} - {int(r.total_patients):,}명 / 병원 {int(r.hosp_cnt):,}개", "value": getattr(r, COL_REGION_HOSP)}
        for r in reg_df.sort_values("total_patients", ascending=False).itertuples()
    ]

hosp0 = agg_hospitals(df)
hospital_options0 = make_hospital_options(hosp0)

_LEFT_PANEL_STYLE = {
    "flex": "0 0 390px", "width": "390px", "minWidth": "390px", "maxWidth": "390px",
    "flexShrink": 0, "padding": "14px", "borderRight": f"1px solid {CLR['border']}",
    "overflowY": "auto", "height": "100%", "boxSizing": "border-box", "background": CLR["panel"],
}
_CHAT_PANEL_STYLE = {
    "flex": "0 0 460px", "width": "460px", "minWidth": "460px", "maxWidth": "460px",
    "flexShrink": 0, "height": "100%", "minHeight": 0, "boxSizing": "border-box",
    "borderLeft": f"1px solid {CLR['border']}", "background": CLR["panel"],
    "display": "flex", "flexDirection": "column",
}
_SECTION_LABEL = {"fontWeight": "bold", "marginTop": "10px", "color": CLR["navy"], "fontSize": "13px"}

_SUGGEST_CHIPS = [
    "전원율 강조해서 보여줘",
    "서울 65세 이상 고령자만",
    "중증 손상(ISS 16+)만 보기",
    "네트워크 거점 분석 보여줘",
    "필터 초기화",
]

app.layout = html.Div(
    style={"display": "flex", "flexDirection": "column", "height": "100vh", "width": "100vw",
           "overflow": "hidden", "background": CLR["bg"]},
    children=[
        dcc.Store(id="selected_store", data=None),
        dcc.Store(id="sankey_modal_store", data={"open": False}),
        dcc.Store(id="chat_history", data=[]),

        # -----------------
        # Sankey Modal (Popup)
        # -----------------
        html.Div(
            id="sankey_modal",
            style={"display": "none", "position": "fixed", "left": 0, "top": 0,
                   "width": "100vw", "height": "100vh", "background": "rgba(0,0,0,0.45)",
                   "zIndex": 5000, "padding": "24px", "boxSizing": "border-box"},
            children=[
                html.Div(
                    style={"background": "white", "borderRadius": "12px", "maxWidth": "1100px",
                           "margin": "0 auto", "height": "calc(100vh - 48px)", "display": "flex",
                           "flexDirection": "column", "boxShadow": "0 12px 36px rgba(0,0,0,0.25)"},
                    children=[
                        html.Div(
                            style={"display": "flex", "alignItems": "center", "padding": "10px 12px",
                                   "borderBottom": "1px solid #eee"},
                            children=[
                                html.H4("Patient State Sankey", style={"margin": 0}),
                                html.Button("✕", id="close_sankey_modal", n_clicks=0,
                                            style={"marginLeft": "auto", "border": "none",
                                                   "background": "transparent", "fontSize": "20px",
                                                   "cursor": "pointer"}),
                            ],
                        ),
                        html.Div(
                            style={"flex": "1 1 auto", "minHeight": 0, "padding": "10px 10px 16px 10px"},
                            children=[dcc.Graph(id="sankey_modal_graph", style={"height": "100%"})],
                        ),
                    ],
                )
            ],
        ),

        # -----------------
        # TOP: Title + KPI summary cards
        # -----------------
        html.Div(
            style={"flex": "0 0 auto", "padding": "12px 16px",
                   "borderBottom": f"1px solid {CLR['border']}", "background": CLR["panel"]},
            children=[
                html.Div(
                    style={"display": "flex", "alignItems": "baseline", "gap": "10px", "marginBottom": "10px"},
                    children=[
                        html.Div("NPIPP", style={"fontSize": "20px", "fontWeight": 800, "color": CLR["primary"]}),
                        html.Div("한국형 시군구 손상 정밀예방 플랫폼",
                                 style={"fontSize": "15px", "fontWeight": 700, "color": CLR["navy"]}),
                        html.Div("National Precision Injury Prevention Platform",
                                 style={"fontSize": "11px", "color": CLR["muted"]}),
                    ],
                ),
                dcc.Loading(html.Div(id="kpi_cards"), type="circle", color=CLR["primary"]),
            ],
        ),

        # -----------------
        # MAIN ROW: Left filters / Center map+profile / Right AI chat
        # -----------------
        html.Div(
            style={"flex": "1 1 auto", "display": "flex", "flexDirection": "row",
                   "minHeight": 0, "overflow": "hidden"},
            children=[
                # ===== LEFT: Global filter panel (390px) =====
                html.Div(
                    style=_LEFT_PANEL_STYLE,
                    children=[
                        html.H3("글로벌 필터", style={"marginTop": "0px", "color": CLR["navy"]}),

                        html.Div("시나리오", style=_SECTION_LABEL),
                        dcc.Dropdown(
                            id="scenario",
                            options=[
                                {"label": "Perspective A (전체 보기)", "value": "A"},
                                {"label": "Perspective B (허브 강조)", "value": "B"},
                                {"label": "Perspective C (네트워크 거점/권역 분석)", "value": "C"},
                                {"label": "Perspective D (지역 집계 보기)", "value": "D"},
                                {"label": "Perspective E (전원 환자만 분석)", "value": "E"},
                            ],
                            value="A", clearable=False,
                        ),
                        html.Hr(),

                        html.Div("표시 레이어", style=_SECTION_LABEL),
                        dcc.Checklist(
                            id="layers",
                            options=[
                                {"label": " 마커(병원/지역)", "value": "hosp"},
                                {"label": " 전원 연결선", "value": "edges"},
                                {"label": " 시군구 단계구분도", "value": "choro"},
                            ],
                            value=["hosp"],
                            labelStyle={"display": "inline-block", "marginRight": "12px"},
                        ),

                        html.Div("단계구분도 지표", style={"marginTop": "10px", "fontSize": "13px"}),
                        dcc.Dropdown(
                            id="choro_metric",
                            options=[
                                {"label": "손상 환자수", "value": "injury_cnt"},
                                {"label": "전원율", "value": "transfer_rate"},
                                {"label": "중증손상률(ISS≥16)", "value": "severe_rate"},
                                {"label": "고령자손상률(65+)", "value": "elderly_rate"},
                            ],
                            value="injury_cnt", clearable=False,
                        ),

                        html.Div("전원 연결선 건수 필터 (Range)", style={"marginTop": "10px", "fontSize": "13px"}),
                        dcc.RangeSlider(
                            id="cnt_range", min=1, max=max_cnt, step=1, value=cnt_range_default,
                            marks={1: "1", 10: "10", 30: "30", 50: "50", max_cnt: str(max_cnt)},
                            updatemode="mouseup",
                        ),

                        html.Hr(),
                        html.H4("Filters", style={"marginBottom": "8px", "color": CLR["navy"]}),

                        html.Div("내원 병원 권역", style=_SECTION_LABEL),
                        dcc.Dropdown(id="filter_hosp_region",
                                     options=[{"label": x, "value": x} for x in region_hosp_opts],
                                     value=[], multi=True, placeholder="예: 서울, 경기, ..."),

                        html.Div("거주지 시도", style=_SECTION_LABEL),
                        dcc.Dropdown(id="filter_addr_region",
                                     options=[{"label": x, "value": x} for x in region_addr_opts],
                                     value=[], multi=True, placeholder="예: 서울특별시, 경상북도, ..."),

                        html.Div("환자 유형 (CLASSES_KOR)", style=_SECTION_LABEL),
                        dcc.Dropdown(id="filter_classes",
                                     options=[{"label": x, "value": x} for x in classes_opts],
                                     value=[], multi=True, placeholder="예: 다수사상, 중증손상, ..."),

                        html.Div("ICD 손상/중독 세부분류 (S/T)", style=_SECTION_LABEL),
                        dcc.Dropdown(id="filter_icd_detail",
                                     options=[{"label": x, "value": x} for x in icd_detail_opts],
                                     value=[], multi=True, placeholder="예: 머리 손상(S00–S09) ..."),

                        html.Div("성별", style=_SECTION_LABEL),
                        dcc.Dropdown(id="filter_sex",
                                     options=[{"label": x, "value": x} for x in sex_opts],
                                     value=[], multi=True, placeholder="예: 1: M, 2 : F"),

                        html.Div("연령 범위", style=_SECTION_LABEL),
                        dcc.RangeSlider(id="filter_age", min=0, max=100, step=1, value=[0, 100],
                                        marks={0: "0", 20: "20", 40: "40", 60: "60", 80: "80", 100: "100"}),

                        html.Div("ISS 범위", style=_SECTION_LABEL),
                        dcc.RangeSlider(id="filter_iss", min=0, max=75, step=1, value=[0, 75],
                                        marks={0: "0", 15: "15", 25: "25", 40: "40", 75: "75"}),
                        dcc.Checklist(id="filter_iss_na",
                                      options=[{"label": " ISS 결측 포함", "value": "include"}],
                                      value=["include"], style={"marginTop": "6px"}),

                        html.Hr(),
                        html.Div("샷(병원/지역 선택)", style=_SECTION_LABEL),
                        dcc.Dropdown(id="entity", options=hospital_options0,
                                     placeholder="예: 경북대병원 (또는 Scenario D에서 지역 선택)",
                                     value=None, clearable=True),
                        html.Div(
                            style={"marginTop": "8px", "fontSize": "12px", "color": CLR["muted"]},
                            children="Tip: 지도에서 마커를 클릭해도 선택과 동일하게 상세 프로파일이 갱신됩니다.",
                        ),

                        # 하위 호환: 콜백이 참조하는 shot_width (숨김)
                        html.Div(dcc.Slider(id="shot_width", min=360, max=720, step=10, value=460),
                                 style={"display": "none"}),

                        html.Div(id="kpi_panel",
                                 style={"marginTop": "12px", "fontSize": "13px", "lineHeight": "1.55"}),
                        html.Hr(),
                        html.Details(
                            open=False,
                            children=[
                                html.Summary("Rankings (Top 20)",
                                             style={"fontWeight": "bold", "cursor": "pointer", "color": CLR["navy"]}),
                                html.Div(id="rank_panel", style={"marginTop": "8px"}),
                            ],
                            style={"marginTop": "6px"},
                        ),
                    ],
                ),

                # ===== CENTER: Map (top) + Detail profile (bottom) =====
                html.Div(
                    style={"flex": "1 1 auto", "display": "flex", "flexDirection": "column",
                           "minWidth": 0, "height": "100%"},
                    children=[
                        html.Div(
                            style={"flex": "1 1 auto", "minHeight": 0, "position": "relative"},
                            children=[dcc.Graph(id="map", style={"height": "100%"})],
                        ),
                        # Bottom detail/profile panel (toggled via callback)
                        html.Div(
                            id="shot_container",
                            style={"display": "none"},
                            children=[
                                html.Div(
                                    style={"display": "flex", "alignItems": "center", "gap": "8px",
                                           "position": "sticky", "top": "0px", "zIndex": 1000,
                                           "background": CLR["panel"], "padding": "8px 12px",
                                           "borderBottom": f"1px solid {CLR['border']}"},
                                    children=[
                                        html.H4(id="shot_title", children="",
                                                style={"margin": 0, "fontSize": "16px", "color": CLR["navy"]}),
                                        html.Button("Sankey 크게 보기", id="open_sankey_btn", n_clicks=0,
                                                    style={"display": "none", "marginLeft": "auto",
                                                           "fontSize": "12px", "cursor": "pointer"}),
                                        html.Button("✕", id="close_shot_btn", n_clicks=0,
                                                    style={"border": "none", "background": "transparent",
                                                           "fontSize": "18px", "cursor": "pointer",
                                                           "lineHeight": "18px"}),
                                    ],
                                ),
                                html.Div(id="shot_body", style={"padding": "12px"}),
                            ],
                        ),
                    ],
                ),

                # ===== RIGHT: AI analysis chat (460px) =====
                html.Div(
                    style=_CHAT_PANEL_STYLE,
                    children=[
                        html.Div(
                            style={"padding": "12px 14px", "borderBottom": f"1px solid {CLR['border']}",
                                   "display": "flex", "alignItems": "center", "gap": "8px"},
                            children=[
                                html.Div("💬", style={"fontSize": "18px"}),
                                html.Div("AI 분석 어시스턴트",
                                         style={"fontWeight": 700, "color": CLR["navy"], "fontSize": "15px"}),
                            ],
                        ),
                        # 스크롤 영역 (flex 자식이 직접 overflow 담당 -> 길어지면 스크롤)
                        html.Div(
                            style={"flex": "1 1 auto", "minHeight": 0, "overflowY": "auto",
                                   "padding": "12px"},
                            children=[
                                dcc.Loading(
                                    html.Div(id="chat_window", children=render_chat([])),
                                    type="dot", color=CLR["primary"],
                                ),
                            ],
                        ),
                        html.Div(
                            style={"padding": "8px 12px", "display": "flex", "flexWrap": "wrap", "gap": "6px",
                                   "borderTop": f"1px solid {CLR['border']}"},
                            children=[html.Button(c, id={"type": "chip", "index": i}, n_clicks=0,
                                                  className="npipp-chip")
                                      for i, c in enumerate(_SUGGEST_CHIPS)],
                        ),
                        html.Div(
                            style={"padding": "10px 12px", "borderTop": f"1px solid {CLR['border']}",
                                   "display": "flex", "gap": "8px"},
                            children=[
                                dcc.Input(id="chat_input", type="text", value="", n_submit=0,
                                          placeholder="질문을 입력하세요...",
                                          style={"flex": "1 1 auto", "padding": "9px 12px",
                                                 "borderRadius": "8px", "border": f"1px solid {CLR['border']}",
                                                 "fontSize": "13px", "outline": "none"}),
                                html.Button("전송", id="chat_send", n_clicks=0,
                                            style={"background": CLR["primary"], "color": "white",
                                                   "border": "none", "borderRadius": "8px",
                                                   "padding": "9px 16px", "fontWeight": 700,
                                                   "cursor": "pointer", "fontSize": "13px"}),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)




# Right panel base styles (we toggle display only via callback)
SHOT_STYLE_BASE = {
    "flexShrink": 0,
    "height": "100vh",
    "maxHeight": "100vh",
    "overflowY": "auto",
    "overflowX": "auto",   # ✅ 테이블/그래프가 폭을 넘으면 가로 스크롤
    "borderLeft": "1px solid #ddd",
    "boxSizing": "border-box",
    "padding": "12px",
}
SHOT_STYLE_HIDDEN = {**SHOT_STYLE_BASE, "display": "none"}
SHOT_STYLE_VISIBLE = {**SHOT_STYLE_BASE, "display": "block"}

# -----------------------------
# 8) Callback
# -----------------------------
@app.callback(
    Output("map", "figure"),
    Output("kpi_panel", "children"),
    Output("rank_panel", "children"),
    Output("entity", "value"),
    Output("shot_title", "children"),
    Output("shot_body", "children"),
    Output("open_sankey_btn", "style"),
    Output("shot_container", "style"),
    Output("entity", "options"),
    Output("selected_store", "data"),
    Output("kpi_cards", "children"),
    Input("scenario", "value"),
    Input("layers", "value"),
    Input("cnt_range", "value"),
    Input("entity", "value"),
    Input("map", "clickData"),
    Input("close_shot_btn", "n_clicks"),
    # filters
    Input("filter_hosp_region", "value"),
    Input("filter_addr_region", "value"),
    Input("filter_classes", "value"),
    Input("filter_icd_detail", "value"),
    Input("filter_sex", "value"),
    Input("filter_age", "value"),
    Input("filter_iss", "value"),
    Input("filter_iss_na", "value"),
    Input("shot_width", "value"),
    Input("choro_metric", "value"),
    State("selected_store", "data"),
)
def update(
    scenario, layers, cnt_range, entity_value, clickData, close_clicks,
    f_hosp_region, f_addr_region, f_classes, f_icd_detail, f_sex,
    f_age, f_iss, f_iss_na, shot_width, choro_metric,
    selected_store
):
    # -------------------------------------------------
    # Selection persistence (fix: closing shot should persist across scenario changes)
    # We keep the last selected entity in dcc.Store and only update it when:
    #  - user picks from dropdown
    #  - user clicks on the map
    #  - user clicks the close (X) button
    # Otherwise (scenario/layer/filter changes) we reuse stored selection.
    # -------------------------------------------------
    trig = ctx.triggered_id
    if trig == "close_shot_btn":
        selected = None
    elif trig == "map" and clickData and "points" in clickData and len(clickData["points"]) > 0:
        p = clickData["points"][0]
        if "customdata" in p and p["customdata"] is not None:
            selected = str(p["customdata"])
        else:
            selected = selected_store
    elif trig == "entity":
        selected = entity_value
    else:
        selected = selected_store

    # Open Sankey button style (button exists in layout; only toggle display)
    open_btn_style = {"display": "none", "marginLeft": "auto", "fontSize": "12px", "cursor": "pointer"}



    # -------------------------------------------------
    # Bottom detail/profile panel style (hide when nothing selected)
    # PRD: 중앙 하단 상세 프로파일 영역
    # -------------------------------------------------
    if selected:
        shot_style = {
            "display": "block",
            "flex": "0 0 auto",
            "height": "42%",
            "maxHeight": "42%",
            "overflowY": "auto",
            "overflowX": "auto",
            "borderTop": f"1px solid {CLR['border']}",
            "boxSizing": "border-box",
            "background": CLR["panel"],
        }
    else:
        shot_style = {"display": "none"}

    # Toggle Sankey button visibility with selection (button always exists in layout)
    if selected:
        open_btn_style["display"] = "inline-block"
    else:
        open_btn_style["display"] = "none"


    include_iss_na = "include" in (f_iss_na or [])

    df_f = apply_filters(
        df,
        hosp_regions=f_hosp_region,
        addr_regions=f_addr_region,
        classes_kor=f_classes,
        icd_st_detail=f_icd_detail,
        sex=f_sex,
        age_range=f_age,
        iss_range=f_iss,
        include_iss_na=include_iss_na,
    )


    # Scenario E: transfer-only (전원 환자만)
    if scenario == "E":
        if "H2_CODE" in df_f.columns:
            df_f = df_f[df_f["H2_CODE"].notna() & (df_f["H2_CODE"].astype(str).str.lower() != "nan")]
    hosp_f = agg_hospitals(df_f) if len(df_f) else agg_hospitals(df.iloc[0:0])
    roles_f = role_metrics(df_f) if len(df_f) else role_metrics(df.iloc[0:0])

    region_f = agg_regions(df_f) if len(df_f) else agg_regions(df.iloc[0:0])
    rmet_f = region_metrics(df_f) if len(df_f) else region_metrics(df.iloc[0:0])

    if scenario == "D":
        entity_options = make_region_options(region_f)
        if selected and (len(region_f) == 0 or selected not in set(region_f[COL_REGION_HOSP].astype(str))):
            selected = None
    else:
        entity_options = make_hospital_options(hosp_f)
        if selected and (len(hosp_f) == 0 or selected not in set(hosp_f["H_CODE"].astype(str))):
            selected = None

    if scenario == "D":
        fig = make_fig_region(df_f, region_f, rmet_f, scenario, layers, cnt_range, selected, choro_metric)
    else:
        fig = make_fig_hospital(df_f, hosp_f, roles_f, scenario, layers, cnt_range, selected, choro_metric)

    if len(df_f):
        summary = html.Div(
            f"필터 결과: rows={len(df_f):,}, patients={df_f['EMS_SN'].nunique():,}, hospitals={df_f['H_CODE'].nunique():,}",
            style={"color": "#444"}
        )
    else:
        summary = html.Div("필터 결과가 비었습니다. (조건을 완화해 보세요)", style={"color": "crimson"})

    # Top KPI summary cards (PRD 필수기능 5)
    try:
        kpi_cards = build_kpi_cards(df_f)
    except Exception as _ek:
        kpi_cards = html.Div(f"KPI 계산 오류: {_ek}", style={"color": "crimson", "fontSize": "12px"})


    # -------------------------------------------------
    # Rankings (Top 20) panel: depends on scenario & filters
    # -------------------------------------------------
    rank_children = html.Div("선택된 시나리오에서 표시할 랭킹이 없습니다.", style={"fontSize":"12px","color":"#666"})
    try:
        if scenario == "B" and len(roles_f):
            # roles_f already contains H_NM and region (from role_metrics). Avoid merging duplicate columns.
            rr = roles_f.copy()
            # If some datasets miss name/region inside roles_f, backfill from hosp_f safely.
            if "H_NM" not in rr.columns and "H_NM" in hosp_f.columns:
                rr = rr.merge(hosp_f[["H_CODE","H_NM"]], on="H_CODE", how="left")
            if COL_REGION_HOSP not in rr.columns and COL_REGION_HOSP in hosp_f.columns:
                rr = rr.merge(hosp_f[["H_CODE", COL_REGION_HOSP]], on="H_CODE", how="left")

            rr = rr.sort_values("hub_score", ascending=False).head(20).copy()
            rr["rank"] = np.arange(1, len(rr)+1)
            rank_table = dash_table.DataTable(
                columns=[
                    {"name":"Rank","id":"rank","type":"numeric"},
                    {"name":"병원","id":"H_NM"},
                    {"name":"권역","id":COL_REGION_HOSP},
                    {"name":"Hub score","id":"hub_score","type":"numeric"},
                    {"name":"환자수","id":"total_patients","type":"numeric"},
                ],
                data=rr.reindex(columns=["rank","H_NM", COL_REGION_HOSP, "hub_score", "total_patients"]).round({"hub_score":2}).to_dict("records"),
                page_size=20,
                style_table={"overflowX":"auto"},
                style_cell={"fontSize":"12px","padding":"6px","whiteSpace":"normal","height":"auto"},
                style_header={"fontWeight":"bold"},
            )

            # 대표 기록(샘플): 현재 필터 + Top Hub 병원 기준
            rep_h = (rr["H_CODE"].iloc[0] if len(rr) and "H_CODE" in rr.columns else None)
            df_rep = df_f[df_f["H_CODE"] == rep_h] if rep_h else df_f
            cols_rep = [c for c in ["EMS_SN","H_NM", COL_REGION_HOSP, "ISS", "AD_RESULT_KOR", "ER_RESULT_KOR"] if c in df_rep.columns]
            rep_rows = []
            if len(df_rep) and "EMS_SN" in df_rep.columns:
                tmp = df_rep.sort_values("ISS", ascending=False) if "ISS" in df_rep.columns else df_rep
                tmp = tmp.drop_duplicates("EMS_SN").head(10)
                rep_rows = tmp[cols_rep].to_dict("records")
            rep_table = dash_table.DataTable(
                columns=[{"name":c, "id":c} for c in cols_rep],
                data=rep_rows,
                page_size=10,
                style_table={"overflowX":"auto"},
                style_cell={"fontSize":"12px","padding":"6px","whiteSpace":"normal","height":"auto"},
                style_header={"fontWeight":"bold"},
            )
            rank_children = html.Div(
                children=[
                    html.Div("전원 네트워크 Top 20 (Hub score)", style={"fontWeight":"bold","marginBottom":"6px"}),
                    rank_table,
                    html.Hr(),
                    html.Div("대표 기록(샘플 10) - Top Hub 병원 기준", style={"fontWeight":"bold","marginBottom":"6px"}),
                    rep_table,
                ]
            )
        elif scenario == "C":
            cmet = compute_network_centrality(df_f)
            if len(cmet):
                nodes_f = build_network_nodes(df_f)
                full = nodes_f.merge(cmet, on="H_CODE", how="left").fillna({
                    "pagerank":0.0, "in_degree_w":0.0, "out_degree_w":0.0, "community":-1
                })
                if COL_REGION_HOSP in full.columns:
                    full[COL_REGION_HOSP] = full[COL_REGION_HOSP].fillna("-")


                # PageRank Top 20
                rr_pr = full.sort_values("pagerank", ascending=False).head(20).copy()
                rr_pr["rank"] = np.arange(1, len(rr_pr)+1)
                rank_table_pr = dash_table.DataTable(
                    columns=[
                        {"name":"Rank","id":"rank","type":"numeric"},
                        {"name":"병원","id":"H_NM"},
                        {"name":"권역","id":COL_REGION_HOSP},
                        {"name":"PageRank","id":"pagerank","type":"numeric"},
                        {"name":"In(w)","id":"in_degree_w","type":"numeric"},
                        {"name":"Out(w)","id":"out_degree_w","type":"numeric"},
                        {"name":"Comm","id":"community","type":"numeric"},
                    ],
                    data=rr_pr[["rank","H_NM", COL_REGION_HOSP, "pagerank","in_degree_w","out_degree_w","community"]]
                        .round({"pagerank":6, "in_degree_w":0, "out_degree_w":0}).to_dict("records"),
                    page_size=20,
                    style_table={"overflowX":"auto"},
                    style_cell={"fontSize":"12px","padding":"6px","whiteSpace":"normal","height":"auto"},
                    style_header={"fontWeight":"bold"},
                )

                # Out(w) Top 20 (송신 병원)
                rr_out = full.sort_values("out_degree_w", ascending=False).head(20).copy()
                rr_out["rank"] = np.arange(1, len(rr_out)+1)
                rank_table_out = dash_table.DataTable(
                    columns=[
                        {"name":"Rank","id":"rank","type":"numeric"},
                        {"name":"병원","id":"H_NM"},
                        {"name":"권역","id":COL_REGION_HOSP},
                        {"name":"Out(w)","id":"out_degree_w","type":"numeric"},
                        {"name":"In(w)","id":"in_degree_w","type":"numeric"},
                    ],
                    data=rr_out[["rank","H_NM", COL_REGION_HOSP, "out_degree_w","in_degree_w"]]
                        .round({"in_degree_w":0, "out_degree_w":0}).to_dict("records"),
                    page_size=20,
                    style_table={"overflowX":"auto"},
                    style_cell={"fontSize":"12px","padding":"6px","whiteSpace":"normal","height":"auto"},
                    style_header={"fontWeight":"bold"},
                )

                # In(w) Top 20 (수신 병원)
                rr_in = full.sort_values("in_degree_w", ascending=False).head(20).copy()
                rr_in["rank"] = np.arange(1, len(rr_in)+1)
                rank_table_in = dash_table.DataTable(
                    columns=[
                        {"name":"Rank","id":"rank","type":"numeric"},
                        {"name":"병원","id":"H_NM"},
                        {"name":"권역","id":COL_REGION_HOSP},
                        {"name":"In(w)","id":"in_degree_w","type":"numeric"},
                        {"name":"Out(w)","id":"out_degree_w","type":"numeric"},
                    ],
                    data=rr_in[["rank","H_NM", COL_REGION_HOSP, "in_degree_w","out_degree_w"]]
                        .round({"in_degree_w":0, "out_degree_w":0}).to_dict("records"),
                    page_size=20,
                    style_table={"overflowX":"auto"},
                    style_cell={"fontSize":"12px","padding":"6px","whiteSpace":"normal","height":"auto"},
                    style_header={"fontWeight":"bold"},
                )

                rank_children = html.Div(
                    children=[
                        html.Div("전원 네트워크 Top 20 (PageRank)", style={"fontWeight":"bold","marginBottom":"6px"}),
                        rank_table_pr,
                        html.Hr(),
                        html.Div("전원 네트워크 Top 20 (Out(w))", style={"fontWeight":"bold","marginBottom":"6px"}),
                        rank_table_out,
                        html.Hr(),
                        html.Div("전원 네트워크 Top 20 (In(w))", style={"fontWeight":"bold","marginBottom":"6px"}),
                        rank_table_in,
                    ]
                )
        elif scenario in ["A","E"] and len(hosp_f):
            rr = hosp_f.sort_values("patient_cnt", ascending=False).head(20).copy()
            rr["rank"] = np.arange(1, len(rr)+1)
            rank_children = dash_table.DataTable(
                columns=[
                    {"name":"Rank","id":"rank","type":"numeric"},
                    {"name":"병원","id":"H_NM"},
                    {"name":"권역","id":COL_REGION_HOSP},
                    {"name":"환자수","id":"patient_cnt","type":"numeric"},
                ],
                data=rr[["rank","H_NM", COL_REGION_HOSP, "patient_cnt"]].to_dict("records"),
                page_size=20,
                style_table={"overflowX":"auto"},
                style_cell={"fontSize":"12px","padding":"6px","whiteSpace":"normal","height":"auto"},
                style_header={"fontWeight":"bold"},
            )


        elif scenario == "D" and len(rmet_f):
            # Region-level ranking (consistent with hover/shot):
            # - total_patients: filtered df_f 기준 지역별 EMS_SN nunique
            # - hosp_cnt: filtered df_f에 등장하는 병원코드(H_CODE/H2_CODE 등) 기준 지역별 unique 병원 수
            rr = rmet_f.copy()

            # --- normalize region labels to align across tables ---
            REGION_MAP = {
                "서울특별시":"서울","부산광역시":"부산","대구광역시":"대구","인천광역시":"인천","광주광역시":"광주",
                "대전광역시":"대전","울산광역시":"울산","세종특별자치시":"세종",
                "경기도":"경기","강원도":"강원","강원특별자치도":"강원",
                "충청북도":"충북","충청남도":"충남","전라북도":"전북","전라남도":"전남",
                "경상북도":"경북","경상남도":"경남","제주특별자치도":"제주","제주도":"제주",
            }
            def norm_reg(x):
                if x is None or (isinstance(x, float) and np.isnan(x)):
                    return np.nan
                s = str(x).strip()
                return REGION_MAP.get(s, s)

            rr["_REG_N"] = rr[COL_REGION_HOSP].map(norm_reg) if COL_REGION_HOSP in rr.columns else np.nan

            # Build H_CODE -> region map from hospital master (avoid type mismatch)
            hdict = {}
            if len(hosp_f) and ("H_CODE" in hosp_f.columns) and (COL_REGION_HOSP in hosp_f.columns):
                _hm = hosp_f[["H_CODE", COL_REGION_HOSP]].dropna(subset=["H_CODE"]).drop_duplicates("H_CODE").copy()
                _hm["H_CODE"] = _hm["H_CODE"].astype(str)
                _hm["_REG_N"] = _hm[COL_REGION_HOSP].map(norm_reg)
                hdict = dict(zip(_hm["H_CODE"], _hm["_REG_N"]))

            # --- per-row region label for filtered rows ---
            _grp = df_f.copy()

            if COL_REGION_HOSP not in _grp.columns or _grp[COL_REGION_HOSP].isna().all():
                # fallback: map from H_CODE -> region
                if "H_CODE" in _grp.columns and len(hdict):
                    _grp["_REG_N"] = _grp["H_CODE"].astype(str).map(hdict).map(norm_reg)
                else:
                    _grp["_REG_N"] = np.nan
            else:
                _grp["_REG_N"] = _grp[COL_REGION_HOSP].map(norm_reg)

            # total patients by region (filtered)
            if "EMS_SN" in _grp.columns:
                pat_by_reg = _grp.groupby("_REG_N", dropna=False)["EMS_SN"].nunique()
            else:
                pat_by_reg = pd.Series(dtype=int)

            # hospital count by region (filtered)
            code_cols = [c for c in ["H_CODE", "H2_CODE", "SRC_H_CODE", "DST_H_CODE", "src_code", "dst_code", "src", "dst", "to_code", "TO_H_CODE"] if c in _grp.columns]
            if len(code_cols) and len(hdict):
                long = []
                for c in dict.fromkeys(code_cols):
                    s = _grp[c].astype(str).replace({"nan": np.nan, "None": np.nan, "": np.nan})
                    long.append(pd.DataFrame({"_HCODE": s, "_REG_N": s.map(hdict)}))
                long_df = pd.concat(long, ignore_index=True)
                hosp_cnt_by_reg = long_df.dropna(subset=["_REG_N", "_HCODE"]).groupby("_REG_N")["_HCODE"].nunique()
            elif "H_CODE" in _grp.columns and _grp["H_CODE"].notna().any():
                _grp["_HCODE"] = _grp["H_CODE"].astype(str).replace({"nan": np.nan, "None": np.nan, "": np.nan})
                hosp_cnt_by_reg = _grp.dropna(subset=["_REG_N", "_HCODE"]).groupby("_REG_N")["_HCODE"].nunique()
            else:
                hosp_cnt_by_reg = pd.Series(dtype=int)

            rr["total_patients"] = rr["_REG_N"].map(pat_by_reg).fillna(0).astype(int) if len(pat_by_reg) else 0
            rr["hosp_cnt"] = rr["_REG_N"].map(hosp_cnt_by_reg).fillna(0).astype(int) if len(hosp_cnt_by_reg) else 0

            rr = rr.sort_values("hub_score", ascending=False).head(20).copy()
            rr["rank"] = np.arange(1, len(rr)+1)

            rank_children = dash_table.DataTable(
                columns=[
                    {"name":"Rank","id":"rank","type":"numeric"},
                    {"name":"지역","id":COL_REGION_HOSP},
                    {"name":"Hub score","id":"hub_score","type":"numeric"},
                    {"name":"환자수","id":"total_patients","type":"numeric"},
                    {"name":"병원수","id":"hosp_cnt","type":"numeric"},
                ],
                data=rr.reindex(columns=["rank", COL_REGION_HOSP, "hub_score", "total_patients", "hosp_cnt"]).round({"hub_score":2}).to_dict("records"),
                page_size=20,
                style_table={"overflowX":"auto"},
                style_cell={"fontSize":"12px","padding":"6px","whiteSpace":"normal","height":"auto"},
                style_header={"fontWeight":"bold"},
            )
    except Exception as _e:
        rank_children = html.Div(f"랭킹 계산 중 오류: {_e}", style={"color":"crimson","fontSize":"12px"})

    # Shot panel content placeholders (header is fixed in layout)
    shot_title = ""
    shot_body = []
        # open_btn_style is defined earlier (keep stable component; only update its display)


    def fmt_dist(dct):
        if not dct:
            return "-"
        items = list(dct.items())[:4]
        return ", ".join([f"{k}:{v}" for k, v in items])

    if scenario == "D":
        if selected and len(region_f) and selected in set(region_f[COL_REGION_HOSP].astype(str)):
            rowm = rmet_f[rmet_f[COL_REGION_HOSP].astype(str) == selected]
            rown = region_f[region_f[COL_REGION_HOSP].astype(str) == selected]
            if len(rowm) and len(rown):
                rm = rowm.iloc[0]
                rn = rown.iloc[0]
                outs = outcome_counts_for_region(df_f, selected)

                kpi_panel = [
                    summary,
                    html.Hr(),
                    html.Div([html.B("선택 지역: "), f"{selected}"]),
                    html.Div([html.B("총 환자수(필터 적용): "), f"{int(rn.get('total_patients',0)):,}"]),
                    html.Div([html.B("병원 수: "), f"{int(rn.get('hosp_cnt',0)):,}"]),
                    html.Div([html.B("로컬 흡수: "), f"{int(rm['local_patients']):,} ({rm['local_ratio']*100:.1f}%)"]),
                    html.Div([html.B("타지역 유입: "), f"{int(rm['inflow_patients']):,} ({rm['inflow_ratio']*100:.1f}%)"]),
                    html.Div([html.B("전원 발생: "), f"{int(rm['transfer_patients']):,} ({rm['transfer_ratio']*100:.1f}%)"]),
                    html.Div([html.B("Hub score: "), f"{rm['hub_score']:.2f}"]),
                    html.Hr(),
                    html.Div([html.B("ER 결과(1차): "), fmt_dist(outs.get("ER_RESULT_KOR", {}))]),
                    html.Div([html.B("입원 결과(1차): "), fmt_dist(outs.get("AD_RESULT_KOR", {}))]),
                    html.Div([html.B("ER 결과(2차): "), fmt_dist(outs.get("H2_ER_RESULT_KOR", {}))]),
                    html.Div([html.B("입원 결과(2차): "), fmt_dist(outs.get("H2_ADM_RESULT_KOR", {}))]),
                ]

                info, sex_s, ages, er = region_shot_data(df_f, selected)

                shot_header = html.Div(
                    [
                        html.Div([html.B("지역: "), info["NAME"]]),
                        html.Div([html.B("총 환자 수(필터 적용): "), f"{info['total']:,}"]),
                        html.Div([html.B("병원 수: "), f"{info['hosp_cnt']:,}"]),
                    ],
                    style={"lineHeight": "1.55"},
                )

                shot_container = html.Div(
                    style={"boxSizing": "border-box"},
                    children=[
                        html.Div(
                            [
                                html.H4("지역 통합 샷", style={"marginTop": "0px", "marginBottom": "0px"}),
                                html.Button(
                                    "✕",
                                    id="close_shot_btn__dup",
                                    n_clicks=0,
                                    style={
                                        "marginLeft": "auto",
                                        "border": "none",
                                        "background": "transparent",
                                        "fontSize": "18px",
                                        "cursor": "pointer",
                                        "lineHeight": "18px",
                                    },
                                ),
                            ],
                            style={"display": "flex", "alignItems": "center", "marginBottom": "8px", "position": "sticky", "top": "0px", "zIndex": 1000, "background": "white", "paddingTop": "4px", "paddingBottom": "4px"},
                        ),
                        html.Div(shot_header, style={"marginBottom": "10px", "fontSize": "14px"}),
                        dcc.Graph(figure=build_sex_pie(sex_s), style={"height": "260px"}, config=GRAPH_CFG_SHOT),
                        dcc.Graph(figure=build_age_hist(ages), style={"height": "260px"}, config=GRAPH_CFG_SHOT),
                        dcc.Graph(figure=build_er_bar(er), style={"height": "260px"}, config=GRAPH_CFG_SHOT),
                        dcc.Graph(figure=build_patient_state_sankey(df_f[df_f[COL_REGION_HOSP].astype(str) == str(selected)], title=f"State transitions (Region: {selected})"), style={"height": "300px"}),
                    ],
                )
            else:
                kpi_panel = [summary]
        else:
            kpi_panel = [summary]
    else:
        if selected and len(roles_f) and selected in set(roles_f["H_CODE"].astype(str)):
            r = roles_f[roles_f["H_CODE"].astype(str) == selected].iloc[0]

            # --- Scenario E: Transfer-focused panel (no ER_RESULT distributions) ---
            if scenario == "E":
                out_df, in_df, diag = transfer_inout_tables(df_f, selected, cnt_range=cnt_range, top_k=10)

                kpi_panel = [
                    summary,
                    html.Hr(),
                    html.Div([html.B("선택 병원: "), f"{r['H_NM']} ({r[COL_REGION_HOSP]})"]),
                    html.Div([html.B("총 환자수(필터 적용): "), f"{int(r['total_patients']):,}"]),
                    html.Div([html.B("전원 발생 환자(병원 기준): "), f"{int(r['transfer_patients']):,} ({r['transfer_ratio']*100:.1f}%)"]),
                    html.Div([html.B("전원 edge 수(전체/필터 범위): "), f"{diag['edges_total']:,} / {diag['edges_in_range']:,}"]),
                    html.Div([html.B("선택 병원 Out edges / In edges: "), f"{diag['edges_out_selected']:,} / {diag['edges_in_selected']:,}"]),
                    html.Div(
                        "※ 전원 연결선이 안 보이면 (1) 건수 필터 범위가 너무 높거나, (2) 전원 대상 병원이 병원정보(좌표/코드)에 없어 edge 생성에서 제외된 경우일 수 있습니다.",
                        style={"fontSize": "12px", "color": "#666", "marginTop": "6px"}
                    ),
                ]

                # Shot highlights: Top outgoing / incoming transfer partners
                info, sex_s, ages, _ = hospital_shot_data(df_f, selected)

                shot_header = html.Div(
                    [
                        html.Div([html.B("병원: "), info["NAME"]]),
                        html.Div([html.B("권역: "), info["REGION"]]),
                        html.Div([html.B("총 환자 수(필터 적용): "), f"{info['total']:,}"]),
                        html.Div([html.B("전원 건수 필터 범위: "), f"{diag['cnt_lo']} ~ {diag['cnt_hi']}"]),
                    ],
                    style={"lineHeight": "1.55"},
                )

                table_style = {
                    "style_table": {"overflowX": "auto"},
                    "style_cell": {"fontSize": "12px", "padding": "6px", "whiteSpace": "normal", "height": "auto"},
                    "style_header": {"fontWeight": "bold"},
                    "page_size": 10,
                }

                shot_container = html.Div(
                    style={"boxSizing": "border-box"},
                    children=[
                        html.Div(
                            [
                                html.H4("전원 네트워크 샷 (Scenario E)", style={"marginTop": "0px", "marginBottom": "0px"}),
                                html.Button(
                                    "✕",
                                    id="close_shot_btn__dup",
                                    n_clicks=0,
                                    style={
                                        "marginLeft": "auto",
                                        "border": "none",
                                        "background": "transparent",
                                        "fontSize": "18px",
                                        "cursor": "pointer",
                                        "lineHeight": "18px",
                                    },
                                ),
                            ],
                            style={"display": "flex", "alignItems": "center", "marginBottom": "8px", "position": "sticky", "top": "0px", "zIndex": 1000, "background": "white", "paddingTop": "4px", "paddingBottom": "4px"},
                        ),
                        html.Div(shot_header, style={"marginBottom": "10px", "fontSize": "14px"}),

                        html.Div("Top 목적지 병원 (Outflow)", style={"fontWeight": "bold", "marginTop": "8px"}),
                        dash_table.DataTable(
                            columns=[
                                {"name": "목적지 병원", "id": "dst_name"},
                                {"name": "전원건수", "id": "cnt", "type": "numeric"},
                                {"name": "평균 전원시간(분)", "id": "avg_transfer_min", "type": "numeric"},
                                {"name": "중증비율(%)", "id": "severe_ratio", "type": "numeric"},
                            ],
                            data=out_df.to_dict("records"),
                            **table_style,
                        ),

                        html.Div("Top 유입 병원 (Inflow)", style={"fontWeight": "bold", "marginTop": "14px"}),
                        dash_table.DataTable(
                            columns=[
                                {"name": "출발 병원", "id": "src_name"},
                                {"name": "전원건수", "id": "cnt", "type": "numeric"},
                                {"name": "평균 전원시간(분)", "id": "avg_transfer_min", "type": "numeric"},
                                {"name": "중증비율(%)", "id": "severe_ratio", "type": "numeric"},
                            ],
                            data=in_df.to_dict("records"),
                            **table_style,
                        ),

                        html.Hr(),
                        html.Div("환자 분포(참고)", style={"fontWeight": "bold", "marginTop": "6px"}),
                        dcc.Graph(figure=build_sex_pie(sex_s), config=GRAPH_CFG_SHOT, style={"height": "240px"}),
                        dcc.Graph(figure=build_age_hist(ages), config=GRAPH_CFG_SHOT, style={"height": "240px"}),
                    ],
                )
                # Prepare shot title/body for new fixed panel
                shot_title = "전원 네트워크 샷 (Scenario E)"
                try:
                    shot_body = shot_container.children[1:] if hasattr(shot_container, "children") and len(shot_container.children or []) > 1 else (shot_container.children or [])
                except Exception:
                    shot_body = []

                return fig, kpi_panel, rank_children, selected, shot_title, shot_body, open_btn_style, shot_style, entity_options, selected, kpi_cards

            # --- Non-E scenarios keep original outcome-based panel below ---
            outs = outcome_counts_for_hospital(df_f, selected)


            kpi_panel = [

                summary,
                html.Hr(),
                html.Div([html.B("선택 병원: "), f"{r['H_NM']} ({r[COL_REGION_HOSP]})"]),
                html.Div([html.B("총 환자수(필터 적용): "), f"{int(r['total_patients']):,}"]),
                html.Div([html.B("로컬 흡수: "), f"{int(r['local_patients']):,} ({r['local_ratio']*100:.1f}%)"]),
                html.Div([html.B("타지역 유입: "), f"{int(r['inflow_patients']):,} ({r['inflow_ratio']*100:.1f}%)"]),
                html.Div([html.B("전원 발생: "), f"{int(r['transfer_patients']):,} ({r['transfer_ratio']*100:.1f}%)"]),
                html.Div([html.B("Hub score: "), f"{r['hub_score']:.2f}"]),
                html.Hr(),
                html.Div([html.B("ER 결과(1차): "), fmt_dist(outs.get("ER_RESULT_KOR", {}))]),
                html.Div([html.B("입원 결과(1차): "), fmt_dist(outs.get("AD_RESULT_KOR", {}))]),
                html.Div([html.B("ER 결과(2차): "), fmt_dist(outs.get("H2_ER_RESULT_KOR", {}))]),
                html.Div([html.B("입원 결과(2차): "), fmt_dist(outs.get("H2_ADM_RESULT_KOR", {}))]),
            ]

            info, sex_s, ages, er = hospital_shot_data(df_f, selected)

            shot_header = html.Div(
                [
                    html.Div([html.B("병원: "), info["NAME"]]),
                    html.Div([html.B("권역: "), info["REGION"]]),
                    html.Div([html.B("총 환자 수(필터 적용): "), f"{info['total']:,}"]),
                ],
                style={"lineHeight": "1.55"},
            )

            shot_container = html.Div(
                style={
                    "width": "100%",
                    "padding": "12px",
                    "borderLeft": "1px solid #ddd",
                    "overflowY": "auto",
                },
                children=[
                    html.Div(
                        [
                            html.H4("병원 상세 샷", style={"marginTop": "0px", "marginBottom": "0px"}),
                            html.Button(
                                "✕",
                                id="close_shot_btn__dup",
                                n_clicks=0,
                                style={
                                    "marginLeft": "auto",
                                    "border": "none",
                                    "background": "transparent",
                                    "fontSize": "18px",
                                    "cursor": "pointer",
                                    "lineHeight": "18px",
                                },
                            ),
                        ],
                        style={"display": "flex", "alignItems": "center", "marginBottom": "8px", "position": "sticky", "top": "0px", "zIndex": 1000, "background": "white", "paddingTop": "4px", "paddingBottom": "4px"},
                    ),
                    html.Div(shot_header, style={"marginBottom": "10px", "fontSize": "14px"}),
                    dcc.Graph(figure=build_sex_pie(sex_s), style={"height": "260px"}, config=GRAPH_CFG_SHOT),
                    dcc.Graph(figure=build_age_hist(ages), style={"height": "260px"}, config=GRAPH_CFG_SHOT),
                    dcc.Graph(figure=build_er_bar(er), style={"height": "260px"}, config=GRAPH_CFG_SHOT),
                    dcc.Graph(figure=build_patient_state_sankey(df_f[df_f["H_CODE"].astype(str) == str(selected)], title="State transitions (Selected Hospital)"), style={"height": "300px"}),
                ],
            )
        else:
            kpi_panel = [summary]

    
    # -------------------------------------------------
    # Extract shot_title / shot_body from legacy shot_container builds (keeps logic changes minimal)
    # -------------------------------------------------
    if selected and (not shot_body):
        try:
            if "shot_container" in locals() and hasattr(shot_container, "children"):
                ch = shot_container.children or []
                # If first child looks like a header with H4, pull title from it and drop it from body
                if len(ch) >= 1 and hasattr(ch[0], "children"):
                    # attempt title extraction
                    try:
                        hdr_children = ch[0].children or []
                        for _c in hdr_children if isinstance(hdr_children, (list, tuple)) else [hdr_children]:
                            if getattr(_c, "id", None) is None and getattr(_c, "children", None) and str(getattr(_c, "children")):
                                # pick first textual header candidate
                                pass
                    except Exception:
                        pass
                # drop legacy header (often first child is a flex header)
                if len(ch) >= 2:
                    shot_body = ch[1:]
                else:
                    shot_body = ch
                if not shot_title:
                    # best-effort title from scenario
                    if scenario == "D":
                        shot_title = "지역 통합 샷"
                    elif scenario == "E":
                        shot_title = "전원 네트워크 샷 (Scenario E)"
                    else:
                        shot_title = "병원 상세 샷"
        except Exception:
            pass

    return fig, kpi_panel, rank_children, selected, shot_title, shot_body, open_btn_style, shot_style, entity_options, selected, kpi_cards




# =========================================================
# Sankey Modal callbacks
# =========================================================
@app.callback(
    Output("sankey_modal_store", "data"),
    Input("open_sankey_btn", "n_clicks"),
    Input("close_sankey_modal", "n_clicks"),
    Input("scenario", "value"),
    State("sankey_modal_store", "data"),
    prevent_initial_call=True,
)
def toggle_sankey_modal(open_clicks, close_clicks, scenario_value, store):
    store = store or {"open": False}
    trig = ctx.triggered_id
    if trig == "open_sankey_btn":
        store["open"] = True
    elif trig == "close_sankey_modal":
        store["open"] = False
    elif trig == "scenario":
        # close modal when scenario changes
        store["open"] = False
    return store


@app.callback(
    Output("sankey_modal", "style"),
    Output("sankey_modal_graph", "figure"),
    Input("sankey_modal_store", "data"),
    Input("scenario", "value"),
    Input("entity", "value"),
    # Use existing filter component IDs (must exist in layout)
    Input("filter_hosp_region", "value"),
    Input("filter_addr_region", "value"),
    Input("filter_classes", "value"),
    Input("filter_icd_detail", "value"),
    Input("filter_sex", "value"),
    Input("filter_age", "value"),
    Input("filter_iss", "value"),
    Input("filter_iss_na", "value"),
)
def render_sankey_modal(store, scenario, entity,
                       hosp_regions, addr_regions, classes_kor, icd_st_detail,
                       sex, age_range, iss_range, iss_na_value):
    store = store or {"open": False}
    base_style = {
        "display": "none",
        "position": "fixed",
        "left": 0,
        "top": 0,
        "width": "100vw",
        "height": "100vh",
        "background": "rgba(0,0,0,0.45)",
        "zIndex": 5000,
        "padding": "24px",
        "boxSizing": "border-box",
    }
    if not store.get("open", False):
        return base_style, go.Figure()

    include_iss_na = "include" in (iss_na_value or [])

    # Filtered data (same logic as main update)
    df_f = apply_filters(
        df,
        hosp_regions=hosp_regions,
        addr_regions=addr_regions,
        classes_kor=classes_kor,
        icd_st_detail=icd_st_detail,
        sex=sex,
        age_range=age_range,
        iss_range=iss_range,
        include_iss_na=include_iss_na,
    )

    # Focus subset depending on scenario/entity
    title = "Patient State Transitions"
    df_subset = df_f

    if scenario in ["A", "B", "C", "E"] and entity not in [None, "", "ALL"]:
        df_subset = df_f[df_f["H_CODE"].astype(str) == str(entity)]
        hn = (df_subset["H_NM"].dropna().iloc[0] if len(df_subset) and "H_NM" in df_subset.columns else str(entity))
        title = f"{hn} - Patient State Transitions"
    elif scenario == "D" and entity not in [None, "", "ALL"]:
        df_subset = df_f[df_f[COL_REGION_HOSP].astype(str) == str(entity)]
        title = f"{entity} - Patient State Transitions"

    fig = build_patient_state_sankey(df_subset, title=title, top_n=10)
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))

    base_style["display"] = "block"
    return base_style, fig


# =========================================================
# 대화형 AI 채팅 콜백 (PRD 필수기능 4)
#   - 자연어 입력 -> OpenAI Function Calling -> 좌측 필터 강제 갱신
# =========================================================
@app.callback(
    Output("chat_history", "data"),
    Output("chat_window", "children"),
    Output("chat_input", "value"),
    Output("scenario", "value"),
    Output("layers", "value"),
    Output("filter_hosp_region", "value"),
    Output("filter_addr_region", "value"),
    Output("filter_classes", "value"),
    Output("filter_icd_detail", "value"),
    Output("filter_sex", "value"),
    Output("filter_age", "value"),
    Output("filter_iss", "value"),
    Input("chat_send", "n_clicks"),
    Input("chat_input", "n_submit"),
    Input({"type": "chip", "index": ALL}, "n_clicks"),
    State("chat_input", "value"),
    State("chat_history", "data"),
    State("scenario", "value"),
    State("filter_hosp_region", "value"),
    State("filter_addr_region", "value"),
    State("filter_classes", "value"),
    State("filter_sex", "value"),
    State("filter_age", "value"),
    State("filter_iss", "value"),
    State("filter_iss_na", "value"),
    prevent_initial_call=True,
)
def on_chat(send_clicks, submit, chip_clicks, chat_input, history,
            scenario, f_hosp, f_addr, f_classes, f_sex, f_age, f_iss, f_iss_na):
    trig = ctx.triggered_id

    # 메시지 결정 (칩 클릭 / 입력 전송)
    if isinstance(trig, dict) and trig.get("type") == "chip":
        # 실제 클릭으로 트리거됐는지 확인 (초기/레이아웃 렌더 방어)
        if not chip_clicks or not any(chip_clicks):
            raise PreventUpdate
        try:
            msg = _SUGGEST_CHIPS[trig["index"]]
        except Exception:
            raise PreventUpdate
    else:
        msg = (chat_input or "").strip()

    if not msg:
        raise PreventUpdate

    history = list(history or [])

    # 현재 상태 요약(그라운딩용)
    df_cur = df
    try:
        include_iss_na = "include" in (f_iss_na or [])
        df_cur = apply_filters(
            df, hosp_regions=f_hosp, addr_regions=f_addr, classes_kor=f_classes,
            sex=f_sex, age_range=f_age, iss_range=f_iss, include_iss_na=include_iss_na,
        )
        k = compute_kpis(df_cur)
        state_summary = (
            f"현재 시나리오: {scenario}\n"
            f"필터 적용 후 환자수: {k['total']:,}명\n"
            f"전원율: {k['transfer_rate']:.1f}%, 중증손상률(ISS≥16): {k['severe_rate']:.1f}%, "
            f"고령자손상률(65+): {k['elderly_rate']:.1f}%, 전원율 20%+ 취약권역: {k['vuln']}개\n"
            f"적용 권역 필터: {f_hosp or '전체'} / 연령: {f_age} / ISS: {f_iss}"
        )
    except Exception as _e:
        state_summary = f"(상태 요약 계산 실패: {_e})"

    # OpenAI 호출 (현재 필터가 적용된 데이터로 질의 계산)
    answer, updates, viz = ai_chat(history, msg, state_summary, df_cur)

    history.append({"role": "user", "content": msg})
    history.append({"role": "assistant", "content": answer, "viz": viz})

    # 필터 컴포넌트 강제 갱신 (Function Calling 결과)
    def pick(key, default=no_update):
        return updates[key] if key in updates else default

    return (
        history,
        render_chat(history),
        "",  # 입력창 비우기
        pick("scenario"),
        pick("layers"),
        pick("filter_hosp_region"),
        pick("filter_addr_region"),
        pick("filter_classes"),
        pick("filter_icd_detail"),
        pick("filter_sex"),
        pick("filter_age"),
        pick("filter_iss"),
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)