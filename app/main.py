# app/main.py

from fastapi import FastAPI
from pydantic import BaseModel
from .services.inference import generate_comment_for_diary

app = FastAPI(
    title="DiaryGarden AI Service",
    version="0.1.0",
)

class InferenceRequest(BaseModel):
    text: str        # 사용자의 일기
    metadata: dict | None = None  # 선택. 나중에 mood, userId 등 넣을 수 있음

class InferenceResponse(BaseModel):
    comment: str     # AI가 생성한 코멘트

@app.get("/health")
def health():
    return {"status": "ok", "model": "HyperCLOVAX-SEED-Text-Instruct-0.5B"}

@app.post("/api/v1/inference", response_model=InferenceResponse)
def run_inference(req: InferenceRequest):
    diary_text = req.text

    # 일기 코멘트 생성
    ai_comment = generate_comment_for_diary(diary_text)

    return InferenceResponse(comment=ai_comment)
