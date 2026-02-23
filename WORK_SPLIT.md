# 분업 계획: Claude Code vs Codex

## Codex 담당 (독립적 스크립트 생성, 코드 작성)

### 1. 통합 평가 스크립트 작성
- `src/evaluate.py` 신규 생성
- 3개 Test set (LIAR/FEVER/FNN) 일괄 평가
- Classification Report + Confusion Matrix 출력
- 앙상블 vs 개별 모델 성능 비교표 생성
- 결과를 `results/` 디렉토리에 JSON + PNG 저장

### 2. Fusion 가중치 최적화 스크립트
- `src/models/optimize_weights.py` 신규 생성
- Validation set 기반 grid search로 최적 가중치 탐색
- 현재 고정 가중치: LIAR 0.35 / FEVER 0.40 / FNN 0.25
- 최적 가중치 결과를 `models/optimal_weights.json`에 저장

### 3. 프론트엔드 UI 구현
- 웹 인터페이스 (claim 입력 → verdict 표시)
- 모델별 상세 결과 시각화 (차트/그래프)
- 증거 문서 하이라이트 표시
- FastAPI 백엔드(`localhost:8000`)와 연동

---

## Claude Code 담당 (기존 코드 수정, 실행/검증, 디버깅)

### 1. Calibration (Temperature Scaling)
- 기존 `src/models/calibrate.py` 분석 및 실행
- 3개 모델 각각 Temperature Scaling 적용
- `temperature.json` 생성 → fusion에 반영

### 2. Retrieval 코퍼스 확장
- 기존 `src/models/retrieval.py` 분석
- Wikipedia 페이지 추가 수집 (현재 425 docs → 확장)
- BM25 인덱스 재빌드
- 검색 품질 검증 ("The earth is flat" 등 테스트)

### 3. API 응답 최적화
- 서버 실행 + 응답 시간 측정
- 배치 처리 또는 모델 경량화 적용
- 최적화 전후 벤치마크

### 4. Git / 모델 파일 관리
- Git LFS 설정 또는 모델 다운로드 스크립트 작성
- 대용량 파일(`.safetensors`) 관리 방안 적용

---

## 작업 순서

```
Phase 1 (병렬 진행)
  Codex  → 통합 평가 스크립트
  Claude → Calibration + Retrieval 확장

Phase 2 (병렬 진행)
  Codex  → Fusion 가중치 최적화
  Claude → API 최적화

Phase 3 (병렬 진행)
  Codex  → 프론트엔드 UI
  Claude → Git/모델 관리 + 최종 검증
```

## 현재 모델 성능 (v2 기준)

| 모델 | Accuracy | F1 Macro |
|---|---|---|
| LIAR (6-class) | 30.2% | 30.0% |
| FEVER (3-class) | 73.6% | 68.6% |
| FNN (2-class) | 99.2% | 99.2% |
