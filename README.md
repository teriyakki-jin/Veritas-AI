# Fakenews-detect

LIAR + FEVER + FakeNewsNet(WELFake) 기반의 통합 팩트체킹 시스템입니다.

## 주요 기능
- BM25 기반 evidence retrieval
- 3개 분류 모델(LIAR/FEVER/FNN) 병렬 추론
- 가중치 기반 fusion으로 최종 credibility score/verdict 산출
- FastAPI 기반 `/verify`, `/verify/batch`, `/health` 제공

## 디렉토리
- `src/models/`: 학습/추론/퓨전/캘리브레이션
- `src/api_server.py`: API 서버
- `data/`: 정규화된 데이터셋
- `models/`: 학습 결과물 및 retrieval 인덱스
- `notebooks/`: Colab 학습 노트북

## 설치
Windows PowerShell 기준:

```powershell
cd D:\develop\Fakenews-detect
.\venv\Scripts\activate
pip install -r requirements.txt
```

## 실행
### 1) CLI 추론
```powershell
python src\models\inference.py
```

### 2) API 서버
```powershell
uvicorn src.api_server:app --host 0.0.0.0 --port 8000
```

## API 예시
### Health
```powershell
curl http://127.0.0.1:8000/health
```

### 단건 검증
```powershell
curl -X POST http://127.0.0.1:8000/verify ^
  -H "Content-Type: application/json" ^
  -d "{\"claim\":\"The Eiffel Tower is in Paris\"}"
```

### 배치 검증
```powershell
curl -X POST http://127.0.0.1:8000/verify/batch ^
  -H "Content-Type: application/json" ^
  -d "{\"claims\":[\"The earth is flat\",\"The Eiffel Tower is in Paris\"]}"
```

## 모델 파일 관리

모델 파일(`.safetensors`, 각 256MB)은 Git에 포함되지 않습니다.

```powershell
# 모델 파일 상태 확인 (필수 파일 존재 여부 검사)
python download_models.py --check

# 모델 매니페스트 생성 (SHA256 해시 목록 생성)
python download_models.py --manifest
```

모델이 없는 경우 Colab 노트북으로 재학습하거나, 팀원에게서 `models/` 디렉토리를 복사하세요.
자세한 안내는 `python download_models.py`를 실행하면 표시됩니다.

## 성능 최적화

- **INT8 Dynamic Quantization**: CPU 추론 시 모델 자동 양자화
- **LRU Cache**: 동일 claim 반복 요청 시 캐시 반환 (256건)
- **Async Processing**: API 배치 처리 시 `asyncio.to_thread` 사용
- **BM25 Corpus**: 5,973개 Wikipedia 문서 기반 evidence retrieval

## 검증 문서
- `verification_guide.md`: 모델 배치 및 추론/API 검증 절차
- `TODO.md`: 작업 항목 체크리스트
