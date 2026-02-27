# Veritas-AI

LIAR + FEVER + FakeNewsNet(WELFake) 데이터셋 기반 통합 팩트체킹 시스템.

**Pipeline:** Claim → BM25 Retrieval → DistilBERT 3-model Inference → Weighted Fusion → Verdict

## 주요 기능

| 기능 | 설명 |
|------|------|
| BM25 Evidence Retrieval | Wikipedia 5,993개 문서에서 관련 근거 추출 |
| 3-model 병렬 추론 | LIAR(6-class) / FEVER(3-class) / FNN(2-class) DistilBERT |
| Weighted Fusion | 그리드서치 최적 가중치: liar=0.0, fever=0.8, fnn=0.2 |
| SSE 스트리밍 | `/verify/stream` — 단계별 실시간 진행 상황 전송 |
| LOO 증거 기여도 | `/verify/explain` — Leave-One-Out 분석으로 증거별 영향도 계산 |
| OpenAI 보조 분석 | `/verify/assist` — GPT 모델로 추가 팩트체크 의견 제공 (선택) |
| 기사 분석 | `/analyze/article` — URL/본문 → claim 추출 → 일괄 검증 |
| 배치 검증 | `/verify/batch` — 최대 50개 claim 동시 처리 |
| Rate Limiting | 엔드포인트별 sliding-window 제한 (verify 30/min, batch 10/min) |
| INT8 양자화 | CPU 추론 자동 양자화로 메모리/속도 최적화 |
| LRU 캐시 | 동일 claim 재요청 시 즉시 반환 (256건) |
| 구조화 로깅 | 요청별 method/path/status/duration, 검증 결과 verdict/score 기록 |

## 모델 성능

| 모델 | 클래스 | 정확도 | Fusion 가중치 |
|------|--------|--------|--------------|
| fever_baseline | 3-class (SUPPORTS/REFUTES/NEI) | ~74% | 0.8 |
| fakenewsnet_baseline | 2-class (REAL/FAKE) | ~99% | 0.2 |
| liar_baseline | 6-class | ~30% (SOTA 수준) | 0.0 |

## API 엔드포인트

| Method | Path | 설명 |
|--------|------|------|
| GET | `/health` | 시스템 상태, 로드된 모델 목록 |
| POST | `/verify` | 단일 claim 검증 |
| GET | `/verify/stream` | SSE 스트리밍 검증 (실시간) |
| POST | `/verify/explain` | 검증 + 증거별 기여도(LOO) |
| POST | `/verify/assist` | 검증 + OpenAI 보조 분석 |
| POST | `/verify/batch` | 최대 50개 claim 일괄 검증 |
| POST | `/analyze/article` | 기사 URL/본문 팩트체크 |

## 설치

```bash
git clone https://github.com/teriyakki-jin/Veritas-AI.git
cd Veritas-AI
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

## 실행

### API 서버

```bash
uvicorn src.api_server:app --host 127.0.0.1 --port 8000
```

브라우저에서 `http://127.0.0.1:8000` 접속 (프론트엔드 포함).
API 문서: `http://127.0.0.1:8000/docs`

### Docker

```bash
docker-compose up --build
```

## 환경 변수

`.env` 파일을 프로젝트 루트에 생성해 설정합니다 (api_server 자동 로드).

```env
# CORS (기본값: *)
CORS_ORIGINS=http://127.0.0.1:8000,http://localhost:8000

# OpenAI 보조 분석 (선택)
OPENAI_ENABLED=false
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
OPENAI_TIMEOUT_SECONDS=30
OPENAI_MAX_OUTPUT_TOKENS=300

# 서버
HOST=127.0.0.1
PORT=8000
```

## API 예시

### 단건 검증

```bash
curl -X POST http://127.0.0.1:8000/verify \
  -H "Content-Type: application/json" \
  -d '{"claim": "The Eiffel Tower is in Paris"}'
```

### SSE 스트리밍

```bash
curl -N "http://127.0.0.1:8000/verify/stream?claim=The+earth+is+flat"
```

이벤트 순서: `retrieving` → `evidence` → `verifying` → `model_done` → `fusing` → `result` → `done`

### 증거 기여도

```bash
curl -X POST http://127.0.0.1:8000/verify/explain \
  -H "Content-Type: application/json" \
  -d '{"claim": "Vaccines are safe"}'
```

### 배치 검증

```bash
curl -X POST http://127.0.0.1:8000/verify/batch \
  -H "Content-Type: application/json" \
  -d '{"claims": ["The earth is flat", "The Eiffel Tower is in Paris"]}'
```

### 기사 분석

```bash
# URL 입력
curl -X POST http://127.0.0.1:8000/analyze/article \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/news"}'

# 본문 직접 입력
curl -X POST http://127.0.0.1:8000/analyze/article \
  -H "Content-Type: application/json" \
  -d '{"article_text": "NASA announced 3 new missions in 2026."}'
```

## 프로젝트 구조

```
src/
├── api_server.py          # FastAPI 서버 (엔드포인트, 미들웨어)
├── models/
│   ├── inference.py       # 메인 파이프라인 (FactCheckPipeline)
│   ├── fusion.py          # 가중 앙상블 퓨전
│   ├── retrieval.py       # BM25 evidence retrieval
│   ├── calibrate.py       # Temperature scaling
│   ├── common.py          # 공통 유틸 (레이블맵, compute_metrics)
│   ├── train_liar.py      # LIAR 학습
│   ├── train_fever.py     # FEVER 학습
│   └── train_fakenewsnet.py  # FakeNewsNet 학습
└── utils/
    ├── article_scraper.py # 기사 스크래핑 (SSRF 방어)
    ├── claim_extractor.py # claim 추출
    └── openai_client.py   # OpenAI 래퍼

frontend/
├── index.html
├── css/style.css
└── js/
    ├── app.js             # UI 핸들러, SSE 스트리밍 UI
    └── api.js             # API 클라이언트

tests/                     # 102개 테스트 (pytest)
models/                    # 학습된 모델 (.safetensors, Git 미포함)
data/                      # 정규화된 데이터셋
```

## 테스트

```bash
source venv/Scripts/activate  # Windows
python -m pytest tests/ -v
```

102개 테스트, CI: GitHub Actions (Python 3.10 / 3.11)

## 모델 파일

모델 파일(`.safetensors`, 각 256MB)은 Git에 포함되지 않습니다.
Colab 노트북(`notebooks/`)으로 재학습하거나, 팀원에게서 `models/` 디렉토리를 복사하세요.

```bash
python download_models.py --check     # 필수 파일 존재 여부 검사
python download_models.py --manifest  # SHA256 해시 목록 생성
```

## 보안

- pickle → JSON (retrieval 인덱스)
- XSS: `escapeHtml()` + `safeUrl()` (프론트엔드)
- SSRF: `_check_private_ip()` + 5MB 제한 (기사 스크래핑)
- CORS: wildcard 시 `allow_credentials=False`
- Rate limiting: 엔드포인트별 sliding-window
- 에러 메시지 내부 정보 노출 차단
