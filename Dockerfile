# Vertex AI Custom Container for DiaryGarden-AI
# GPU 지원 베이스 이미지 (CUDA 11.8 + Python 3.10)
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# pip 업그레이드
RUN pip install --no-cache-dir --upgrade pip

# requirements 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 앱 코드 복사
COPY app/ ./app/

# 학습된 감정 분류 모델 복사 (로컬에서 학습 후)
COPY emotion-model/ ./emotion-model/

# Vertex AI는 기본적으로 8080 포트 사용
ENV PORT=8080
ENV AIP_HTTP_PORT=8080
ENV AIP_HEALTH_ROUTE=/health
ENV AIP_PREDICT_ROUTE=/api/v1/inference

# HuggingFace 캐시 디렉토리 (선택)
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

# 서버 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
