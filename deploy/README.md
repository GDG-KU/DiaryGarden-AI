# Vertex AI 배포 가이드

## 📋 사전 준비

### 1. GCP 프로젝트 설정
```bash
# GCP CLI 로그인
gcloud auth login

# 프로젝트 설정
gcloud config set project YOUR_PROJECT_ID
```

### 2. 필수 API 활성화
```bash
gcloud services enable \
    aiplatform.googleapis.com \
    artifactregistry.googleapis.com \
    cloudbuild.googleapis.com
```

### 3. 감정 분류 모델 학습 (로컬)
```bash
# 프로젝트 루트에서 실행
cd train
python train.py
# → emotion-model/ 폴더가 생성됨
```

---

## 🚀 배포 방법

### Option A: 스크립트 사용 (Linux/Mac)
```bash
# 스크립트 수정 (PROJECT_ID 등)
vim deploy/deploy_vertex_ai.sh

# 실행
chmod +x deploy/deploy_vertex_ai.sh
./deploy/deploy_vertex_ai.sh
```

### Option B: 수동 배포 (Windows)

#### Step 1: Docker 이미지 빌드
```powershell
docker build -t asia-northeast3-docker.pkg.dev/YOUR_PROJECT/diary-garden-ai/inference:v1 .
```

#### Step 2: Artifact Registry에 푸시
```powershell
gcloud auth configure-docker asia-northeast3-docker.pkg.dev
docker push asia-northeast3-docker.pkg.dev/YOUR_PROJECT/diary-garden-ai/inference:v1
```

#### Step 3: Vertex AI 모델 업로드
```powershell
gcloud ai models upload `
    --region=asia-northeast3 `
    --display-name=diary-garden-model `
    --container-image-uri=asia-northeast3-docker.pkg.dev/YOUR_PROJECT/diary-garden-ai/inference:v1 `
    --container-ports=8080 `
    --container-health-route=/health `
    --container-predict-route=/api/v1/inference
```

#### Step 4: Endpoint 생성 및 배포
```powershell
# Endpoint 생성
gcloud ai endpoints create --region=asia-northeast3 --display-name=diary-garden-endpoint

# 모델 배포 (GPU 포함)
gcloud ai endpoints deploy-model ENDPOINT_ID `
    --region=asia-northeast3 `
    --model=MODEL_ID `
    --machine-type=n1-standard-4 `
    --accelerator=type=NVIDIA_TESLA_T4,count=1 `
    --min-replica-count=1
```

---

## 📡 API 호출 방법

### Vertex AI Endpoint 직접 호출
```bash
curl -X POST \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  https://asia-northeast3-aiplatform.googleapis.com/v1/projects/PROJECT_ID/locations/asia-northeast3/endpoints/ENDPOINT_ID:predict \
  -d '{
    "instances": [{
      "title": "오늘의 일기",
      "text": "오늘은 정말 기분이 좋았다!"
    }]
  }'
```

### 응답 예시
```json
{
  "predictions": [{
    "comment": "정말 기분 좋은 하루였네요, 그 행복이 오래 이어지길 바라요.",
    "dominantEmotion": "joy",
    "emotionScores": {
      "joy": 0.92,
      "sadness": 0.02,
      "anger": 0.01,
      "neutral": 0.05
    }
  }]
}
```

---

## 💰 비용 예상 (월 기준)

| 항목 | 사양 | 예상 비용 |
|------|------|----------|
| Vertex AI Endpoint | n1-standard-4 + T4 GPU | ~$200-400/월 (사용량에 따라) |
| Artifact Registry | 이미지 저장 | ~$1-5/월 |

> **팁**: `min-replica-count=0`으로 설정하면 사용하지 않을 때 비용 절감 가능 (Cold Start 있음)

---

## 🔧 Backend 연동

BE에서는 다음과 같이 호출:

### Node.js/NestJS 예시
```typescript
import { VertexAI } from '@google-cloud/vertexai';

const vertexAI = new VertexAI({
  project: 'YOUR_PROJECT_ID',
  location: 'asia-northeast3',
});

async function getEmotionAndComment(title: string, text: string) {
  const endpoint = vertexAI.preview.getEndpoint('ENDPOINT_ID');
  
  const response = await endpoint.predict({
    instances: [{ title, text }],
  });
  
  return response.predictions[0];
}
```

### REST API 직접 호출 (권장)
```typescript
// 서비스 계정 인증 사용
const response = await fetch(
  `https://asia-northeast3-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/asia-northeast3/endpoints/${ENDPOINT_ID}:predict`,
  {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${accessToken}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      instances: [{ title, text }],
    }),
  }
);
```
