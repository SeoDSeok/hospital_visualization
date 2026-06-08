# 한국형 시군구 손상 정밀예방 플랫폼 (NPIPP)

질병관리청 손상 자료와 소방청 구급 데이터를 통합 분석하여 시군구 단위 중증 손상 위험도를
공간 시각화하고, LLM 기반 대화형 인터페이스로 맞춤형 예방 정책 수립을 지원하는 BI 대시보드.

## 주요 기능
- 좌측 글로벌 다중 조건 필터 패널 (기간/연령/지역/환자유형/ISS 등)
- 중앙 Web-GIS 지도 — 병원/지역 마커, 전원(이송) 네트워크, **시군구 단계구분도(Choropleth)**
- 상단 핵심 지표(KPI) 요약 카드 + 스파크라인
- 하단 지역/병원 상세 프로파일 (손상 피라미드, 분포, Sankey)
- 우측 **대화형 AI 분석 채팅** (OpenAI Function Calling)
  - 자연어로 필터/지도 제어 (예: "전원율 강조해줘", "서울 65세 이상만")
  - 데이터 질의 → 표/그래프 응답 (예: "강원과 제주 전원율 비교해줘")

## 설치
```bash
pip install -r requirements.txt
```

## 환경변수 (.env)
`.env.example` 을 복사해 `.env` 를 만들고 OpenAI 키를 입력하세요.
```bash
cp .env.example .env   # 이후 .env 안의 OPENAI_API_KEY 값을 채움
```
> 키가 없어도 지도·필터·KPI·단계구분도는 정상 동작하며, AI 채팅만 비활성화됩니다.

## 데이터 준비 (별도)
환자 데이터는 민감/대용량이라 저장소에 포함하지 않습니다. 아래 파일을 직접 준비하세요.
- `data/merged_data_v10.csv` — 통합 분석 데이터 (필수)
- `data/sigungu_merged.geojson` — 시군구 경계 (단계구분도용)
  - 포함된 `maps-main.zip` 으로 생성: `python build_geojson.py`

## 실행
```bash
python app9_16.py
# http://localhost:8050
```

## 기술 스택
Python Dash (Flask), Plotly, Pandas/NumPy/NetworkX, Shapely/pyproj, OpenAI SDK
