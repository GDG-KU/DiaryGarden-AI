## 빠른 시작

1. 가상 환경 생성(선택 사항이지만 권장):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. 의존성 설치:
   ```powershell
   pip install -r requirements.txt
   ```
3. 개발 서버 실행:
   ```powershell
   uvicorn app.main:app --reload
   ```
4. 상호작용형 문서(Interactive Docs)를 열어 API를 테스트합니다: http://127.0.0.1:8000/docs

## API 개요

- `GET /health` — 서비스의 기본 상태(헬스체크)를 확인합니다.
- `POST /api/v1/inference` — `text` 필드(선택적 `metadata` 포함)를 가진 JSON 페이로드를 수신하며, 실제 모델 출력으로 교체할 수 있는 플레이스홀더 응답을 반환합니다.

## 구성

기본값을 환경 변수로 재정의할 수 있습니다:

- `PROJECT_NAME` — OpenAPI 메타데이터에 노출되는 서비스 명칭입니다.
- `API_VERSION` — `/health` 응답에 반환되는 버전 문자열입니다.
- `DEBUG` — `true`로 설정하면 FastAPI의 디버그 모드를 활성화합니다.


## 향후 작업

- `app/services/inference.py` 내부에 모델 로딩 기능을 구현합니다.
- 배포 요구사항에 맞는 인증 및 요청 검증 규칙을 추가합니다.
- 운영 환경 수준의 관찰성(로깅, 트레이싱 등)을 확장합니다.
