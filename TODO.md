# Fakenews-detect TODO

## 검증 완료 항목 (2025-02-13, 모델 v2 반영 2025-02-14)

| 항목 | 상태 | 비고 |
|---|---|---|
| 모델 파일 배치 (liar/fever/fnn) | ✅ | 3개 디렉토리 필수 파일 모두 존재 |
| retrieval_index.pkl | ✅ | 425 docs |
| inference.py 추론 | ✅ | 2개 테스트 클레임 정상 출력 |
| API /health | ✅ | 3개 모델 로드 확인 |
| API /verify | ✅ | verdict, score, evidence, model_details 반환 |
| API /verify/batch | ✅ | 입력 2건 → 결과 2건 정상 |

---

## 수정 완료 항목

- [x] `inference.py` — Windows cp949 인코딩 오류 수정 (doc_id + snippet 모두 ascii fallback 처리)
- [x] venv에 `uvicorn`, `pydantic`, `fastapi` 수동 설치 (requirements.txt에는 있으나 venv에 미설치 상태였음)
- [x] `requirements.txt`에 `pydantic>=2.0.0` 명시 추가
- [x] `.gitignore` 추가 (`venv/`, `__pycache__/`, `models/**/*.safetensors` 등 대용량/생성 파일 제외)
- [x] `README.md` 작성 (설치/실행/API 사용법)
- [x] `src/api_server.py` 입력 검증 강화 (공백 claim 차단)
- [x] `src/api_server.py` 에러 핸들링 보강 (`verify`, `verify/batch` 예외 처리)
- [x] `src/api_server.py` CORS 설정 추가 (`CORS_ORIGINS` 환경변수 기반)

---

## 진행해야 할 작업

### 1. 의존성 환경 정리
- [x] `pip install -r requirements.txt`로 전체 의존성 일괄 재설치 검증

### 2. 모델 품질 개선
- [x] Colab에서 3개 모델 v2 재학습 완료 (2025-02-14)
  - `train_liar_colab.ipynb` → LIAR: Acc 29.7%→30.2%, F1 27.4%→30.0% (+2.6%p, metadata 입력 추가)
  - `train_fever_colab.ipynb` → FEVER: Acc 74.4%→73.6%, F1 69.1%→68.6% (소폭 하락, cosine LR)
  - `train_fnn_colab.ipynb` → FNN: Acc 99.2%→99.2%, F1 99.2%→99.2% (유지)
- [x] `calibrate.py`로 Temperature Scaling 적용 (LIAR: 1.5227, FEVER: 1.0873, FNN: 1.6755)
- [x] 각 모델 Test set 기준 Classification Report + Confusion Matrix 생성 (`results/quick_eval/{liar,fever,fnn}_report.json`, confusion_matrix 포함)

### 3. Retrieval 품질 개선
- [x] Retrieval 병목 진단 리포트 생성 (`results/retrieval_quality_report.json`, gold_page_coverage_in_corpus=0.0 확인)
- [x] BM25 인덱스 코퍼스 확장 (425 → ~5000+ docs, Wikipedia API로 FEVER gold pages 다운로드)
- [x] "The earth is flat" 검색 시 관련 없는 문서 → 인덱스 재빌드(5973 docs)로 해결 (→ Flat Earth 문서 반환)
- [x] 증거 문서 품질 개선 (코퍼스 425→5973 docs, seed topics 추가, `src/data/expand_corpus.py`)

### 4. Fusion 앙상블 튜닝
- [x] 가중치 최적화 스크립트 작성 (`src/models/optimize_weights.py`, 결과: `models/optimal_weights.json`)
- [x] 가중치 최적화 (현재 고정: LIAR 0.35 / FEVER 0.40 / FNN 0.25)
- [x] Validation set 기반 최적 가중치 탐색 (grid search 또는 learned weights) (`models/optimal_weights.json`, `models/optimal_weights_quick.json`)
- [x] "The Eiffel Tower is in Paris" → MOSTLY TRUE (0.5980)로 개선 (Temperature Scaling 적용 후)

### 5. 평가 체계 구축
- [x] 통합 평가 스크립트 작성 (`src/evaluate.py`, 결과: `results/*_evaluation.json`, `results/evaluation_summary.json`)
- [x] 통합 평가 스크립트 작성 (3개 Test set 일괄 평가)
- [x] 3모델 앙상블 vs 개별 모델 성능 비교표 생성 (`results/quick_eval/comparison_table.json`)
- [x] 평가 결과를 `results/` 디렉토리에 저장 (`results/quick_eval/*.json`)

### 6. API 서버 보강
- [x] 에러 핸들링 강화 (잘못된 입력, 빈 문자열 등)
- [x] CORS 설정 (프론트엔드 연동 시 필요)
- [x] API 응답 시간 최적화 (INT8 동적 양자화 + LRU 캐시 + asyncio.to_thread 비동기 처리)
- [x] Swagger/OpenAPI 문서 자동 생성 확인 (`/docs` 엔드포인트)

### 7. 프론트엔드 UI (Optional)
- [x] 웹 UI 초안 구현 (`frontend/index.html`, `/verify` 연동)
- [x] 웹 인터페이스 구현 (claim 입력 → verdict 표시)
- [x] 모델별 상세 결과 시각화 (차트/그래프)
- [x] 증거 문서 하이라이트 표시

### 8. 프로젝트 관리
- [x] Git 저장소 초기화 및 `.gitignore` 설정
- [x] README.md 작성 (설치 방법, 사용법, 아키텍처 설명)
- [x] 대용량 모델 파일 관리 (`download_models.py` 체크/매니페스트, `.gitignore`에서 .safetensors 제외 + config 유지)
