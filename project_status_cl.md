# Project Status: Combined Fact-Checking System (FEVER + LIAR + FakeNewsNet)

## 1. Project Overview
This project aims to build a comprehensive Fact-Checking System integrating three datasets: **LIAR** (Political claims), **FEVER** (Fact Extraction and Verification), and **FakeNewsNet** (Social Media News Verification). The system uses a **Pipeline Architecture** with a dedicated Evidence Retrieval module (Ranking/BM25) and multiple Verification Models (DistilBERT).

---

## 2. Current Architecture & Progress

### A. Data Pipeline (`src/data/`)
*   **Unified Schema:** Implemented `UnifiedSample` dataclass to standardize all datasets (id, text, label, evidence). Labels are normalized to a 0.0 ~ 1.0 score.
*   **LIAR (`src/data/load_liar.py`):**
    *   **Status:** ✅ Complete.
    *   **Details:** Downloads manual TSV files, normalizes 6-class labels, and saves as JSONL.
    *   **Data:** `data/liar/train.jsonl` (10,269) / `valid.jsonl` (1,284) / `test.jsonl` (1,283)
*   **FEVER (`src/data/download_fever_final.py`):**
    *   **Status:** ✅ Complete (Augmented).
    *   **Details:** Local `.jsonl` files processed. **Evidence Augmentation** implemented: `build_fever_cache.py` fetches Wikipedia pages for evidence, and `augment_fever_data.py` injects text into training data.
    *   **Data:** `data/fever/train_normalized.jsonl` (145,449) / wiki-cache (425 docs)
*   **FakeNewsNet / WELFake:**
    *   **Status:** ✅ Complete.
    *   **Details:** WELFake CSV (233MB, 72,134건) 수동 다운로드 완료. CSV → JSONL 변환 완료 (80/20 split, seed=42).
    *   **Data:** `data/welfake/train.jsonl` (57,707) / `test.jsonl` (14,427)
    *   **Label 분포:** fake 37,106 / real 35,028 (거의 균형)

### B. Baseline Models (`src/models/`)
*   **LIAR Model (`train_liar.py`):**
    *   **Status:** ⏳ Colab 학습 대기 중.
    *   **Type:** 6-class Text Classification (DistilBERT).
    *   **Colab:** `notebooks/train_liar_colab.ipynb`
*   **FEVER Verifier (`train_fever.py`):**
    *   **Status:** ⏳ Colab 학습 대기 중.
    *   **Type:** 3-class Claim Verification (DistilBERT).
    *   **Colab:** `notebooks/train_fever_colab.ipynb`
*   **FNN Model (`train_fakenewsnet.py`):**
    *   **Status:** ⏳ Colab 재학습 대기 중 (기존 checkpoint-2320은 불완전한 데이터 기반).
    *   **Type:** 2-class Fake/Real Classification (DistilBERT).
    *   **Colab:** `notebooks/train_fnn_colab.ipynb`

### C. Evidence Retrieval (`src/models/retrieval.py`)
*   **Status:** ✅ Complete.
*   **Feature:** BM25 Indexing on cached Wiki pages (pickle-based persistence).
*   **Fix Applied:** `save_index()` / `load_index()` 메서드 추가, NLTK `punkt_tab` 호환성 수정.
*   **Index:** `models/retrieval_index.pkl` (425 docs, 자동 빌드 완료)
*   **Metrics:** `Hit@k` and `Recall@k` evaluation 구현.

### D. Fusion (`src/models/fusion.py`)
*   **Status:** ✅ Implemented.
*   **Feature:** Weighted ensemble combining LIAR (6-class), FEVER (3-class), FNN (2-class) outputs.
*   **Method:** Each model's logits → temperature-scaled softmax → credibility score → weighted average.
*   **Default Weights:** LIAR 0.35 / FEVER 0.40 / FNN 0.25
*   **Verdict Mapping:** Score 0.0~1.0 → FALSE / MOSTLY FALSE / HALF TRUE / MOSTLY TRUE / TRUE.

### E. Calibration (`src/models/calibrate.py`)
*   **Status:** ✅ Implemented.
*   **Feature:** Temperature Scaling (Platt variant) on validation logits. Saves `temperature.json` per model.

### F. Inference Pipeline (`src/models/inference.py`)
*   **Status:** ✅ Implemented & Tested.
*   **Flow:** `Claim → BM25 Retrieval → Parallel Model Inference → Fusion → Verdict`
*   **Features:**
    *   Auto-resolves model checkpoints if final model not saved.
    *   Gracefully handles missing models (runs with whatever is available).
    *   Interactive CLI mode via `python src/models/inference.py`.
*   **Test 결과:** FNN 단독으로 "The earth is flat" → credibility 0.6637, verdict "MOSTLY TRUE" (3모델 앙상블 후 개선 예상)

### G. API Server (`src/api_server.py`)
*   **Status:** ✅ Implemented.
*   **Framework:** FastAPI with Pydantic schemas.
*   **Endpoints:**
    *   `GET /health` — System health & loaded models.
    *   `POST /verify` — Verify single claim (returns verdict, score, evidence, model details).
    *   `POST /verify/batch` — Verify up to 50 claims.
*   **Run:** `uvicorn src.api_server:app --host 0.0.0.0 --port 8000`

### H. Colab Training Notebooks (`notebooks/`)
*   **Status:** ✅ 3개 노트북 생성 완료.
*   **공통 구조:** Setup → Data Upload → Dataset Class → Train → Evaluate (Classification Report + Confusion Matrix) → Save → Download zip
*   **Notebooks:**
    *   `train_liar_colab.ipynb` — 필요 파일: `data/liar/{train,valid,test}.jsonl` (3개)
    *   `train_fever_colab.ipynb` — 필요 파일: `data/fever/train_normalized.jsonl` (1개)
    *   `train_fnn_colab.ipynb` — 필요 파일: `data/welfake/{train,test}.jsonl` (2개)
*   **GPU 예상 시간 (T4):** LIAR ~3분 / FEVER ~15분 / FNN ~10분

---

## 3. Next Steps (To-Do List)

### 즉시 실행 (Colab GPU 학습)
1.  ⏳ Google Drive에 데이터 업로드 (총 6개 파일)
2.  ⏳ `train_liar_colab.ipynb` 실행 → `models/liar_baseline/` 저장
3.  ⏳ `train_fever_colab.ipynb` 실행 → `models/fever_baseline/` 저장
4.  ⏳ `train_fnn_colab.ipynb` 실행 → `models/fakenewsnet_baseline/` 저장

### 학습 완료 후
1.  학습된 모델 zip을 로컬 `models/` 폴더에 압축 해제
2.  `python src/models/inference.py`로 3모델 앙상블 테스트
3.  (Optional) `calibrate.py`로 각 모델 Temperature Scaling
4.  `uvicorn src.api_server:app`으로 API 서버 기동

### 평가
1.  각 데이터셋 Test set에 대해 Confusion Matrix + Classification Report 생성
2.  3모델 앙상블 vs 개별 모델 성능 비교

### Optional Enhancements
1.  **Stacking Ensemble:** Train a meta-learner on model outputs instead of fixed weights.
2.  **Frontend UI:** Web interface for claim verification.
3.  **Async Batch Processing:** Background job queue for large batch requests.

---

## 4. How to Resume Work
*   **Environment:** `d:\develop\Fakenews-detect\venv\Scripts\activate`
*   **Key Scripts:**
    *   Train LIAR: `python src/models/train_liar.py` (or Colab notebook)
    *   Train FEVER: `python src/models/train_fever.py` (or Colab notebook)
    *   Train FNN: `python src/models/train_fakenewsnet.py` (or Colab notebook)
    *   Build Retrieval Index: `python src/models/retrieval.py`
    *   Interactive Inference: `python src/models/inference.py`
    *   Start API Server: `uvicorn src.api_server:app --host 0.0.0.0 --port 8000`

---

## 5. File Structure
```
Fakenews-detect/
├── data/
│   ├── liar/          train.jsonl (10,269) / valid.jsonl (1,284) / test.jsonl (1,283)
│   ├── fever/         train_normalized.jsonl (145,449) / wiki-cache/ (425 docs)
│   └── welfake/       WELFake_Dataset.csv (72,134) / train.jsonl (57,707) / test.jsonl (14,427)
├── models/
│   ├── liar_baseline/           ⏳ 학습 대기
│   ├── fever_baseline/          ⏳ 학습 대기
│   ├── fakenewsnet_baseline/    checkpoint-2320 (이전 학습), ⏳ 재학습 대기
│   └── retrieval_index.pkl      ✅ BM25 인덱스 (425 docs)
├── notebooks/
│   ├── train_liar_colab.ipynb   ✅
│   ├── train_fever_colab.ipynb  ✅
│   └── train_fnn_colab.ipynb    ✅
├── src/
│   ├── data/          normalize.py, load_liar.py, download_fever_final.py, ...
│   ├── models/        train_liar.py, train_fever.py, train_fakenewsnet.py,
│   │                  retrieval.py, fusion.py, inference.py, calibrate.py, common.py
│   └── api_server.py  FastAPI wrapper
└── project_status_cl.md
```
