# 모델/시스템 검증 가이드

## 1. 목적
Colab에서 만든 모델 zip을 `models/*_baseline`에 반영한 뒤,
실제로 추론과 API가 정상 동작하는지 검증합니다.

## 2. 사전 확인
아래 경로에 모델 파일이 있어야 합니다.

- `models/liar_baseline/`
- `models/fever_baseline/`
- `models/fakenewsnet_baseline/`

각 폴더 최소 파일:
- `config.json`
- `model.safetensors` (또는 `pytorch_model.bin`)
- `tokenizer.json`
- `tokenizer_config.json`
- `label_map.json`

## 3. 실행 환경 주의
현재 프로젝트 가상환경은 Windows 기준(`venv\\Scripts`)입니다.
따라서 검증은 **Windows PowerShell/CMD**에서 진행하세요.

WSL에서 `python3 src/models/inference.py`를 실행하면
`torch` 공유 라이브러리 로딩 오류가 날 수 있습니다.

## 4. 모델 파일 검증 (PowerShell)
프로젝트 루트에서 실행:

```powershell
cd D:\develop\Fakenews-detect

Test-Path .\models\liar_baseline\model.safetensors
Test-Path .\models\fever_baseline\model.safetensors
Test-Path .\models\fakenewsnet_baseline\model.safetensors
```

모두 `True`가 나오면 모델 파일은 정상 배치된 상태입니다.

## 5. 추론 파이프라인 검증

```powershell
cd D:\develop\Fakenews-detect
.\venv\Scripts\activate
python src\models\inference.py
```

실행 후 CLI에 테스트 클레임 입력:
- `The earth is flat`
- `The Eiffel Tower is in Paris`

확인 포인트:
- retrieval 결과가 출력되는지
- 3개 모델 점수/결합 점수가 출력되는지
- 최종 verdict가 출력되는지
- 예외 없이 반복 입력이 가능한지

## 6. API 서버 검증

### 6.1 서버 실행
```powershell
cd D:\develop\Fakenews-detect
.\venv\Scripts\activate
uvicorn src.api_server:app --host 0.0.0.0 --port 8000
```

### 6.2 Health 체크
새 터미널에서:

```powershell
curl http://127.0.0.1:8000/health
```

확인 포인트:
- 서버 상태가 정상(`ok`/유사 상태)
- 로드된 모델 목록에 LIAR/FEVER/FNN이 표시되는지

### 6.3 단건 검증
```powershell
curl -X POST http://127.0.0.1:8000/verify ^
  -H "Content-Type: application/json" ^
  -d "{\"claim\":\"The earth is flat\"}"
```

확인 포인트:
- `verdict`
- `credibility_score`
- `evidence` 배열
- 모델별 상세 결과 필드

### 6.4 배치 검증
```powershell
curl -X POST http://127.0.0.1:8000/verify/batch ^
  -H "Content-Type: application/json" ^
  -d "{\"claims\":[\"The earth is flat\",\"The Eiffel Tower is in Paris\"]}"
```

확인 포인트:
- 입력 개수와 결과 개수가 일치하는지
- 각 항목에 verdict/score가 정상 포함되는지

## 7. 실패 시 점검 순서
1. 모델 파일 존재 여부 재확인 (`models/*_baseline/*`)
2. `retrieval_index.pkl` 존재 확인 (`models/retrieval_index.pkl`)
3. 가상환경 활성화 확인 (`.\\venv\\Scripts\\activate`)
4. `pip list`로 `torch`, `transformers`, `fastapi`, `uvicorn` 설치 확인
5. 포트 충돌 시 `--port 8001`로 변경

## 8. 완료 기준
아래 3가지를 모두 만족하면 검증 완료:

1. `inference.py`에서 claim 입력 시 최종 verdict가 안정적으로 출력됨
2. `/health`가 정상 응답하고 모델 로드 상태가 확인됨
3. `/verify`, `/verify/batch`가 200 응답과 함께 결과 JSON을 반환함
