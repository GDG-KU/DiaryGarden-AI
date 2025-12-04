# app/main.py
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict

app = FastAPI(title="DiaryGarden-AI", version="0.2.0")

# 한글 깨짐 방지 미들웨어
@app.middleware("http")
async def ensure_utf8_content_type(request, call_next):
    response = await call_next(request)
    if response.headers.get("content-type", "").startswith("application/json"):
        response.headers["content-type"] = "application/json; charset=utf-8"
    return response

# [변경 1] 요청 스키마: title 추가
class InferenceRequest(BaseModel):
    title: str = Field(..., description="일기 제목")
    text: str = Field(..., description="일기 내용")
    # metadata는 필요 없다면 제거해도 되지만, 확장을 위해 유지
    metadata: Optional[Dict] = Field(None)

# [변경 2] 응답 스키마: 구조 변경
class InferenceResponse(BaseModel):
    dominantEmotion: str = Field(..., description="지배적인 감정 (happy, sad, angry, calm)")
    emotionScores: Dict[str, float] = Field(..., description="감정별 점수")

@app.get("/health", summary="헬스체크")
def health():
    return {"ok": True}

@app.post("/api/v1/inference", response_model=InferenceResponse)
async def inference(req: InferenceRequest):
    from app.services.inference import InferenceService
    
    result = await InferenceService.generate_response(req.title, req.text)

    # 여기! JSONResponse → Pydantic 모델로 변환해서 반환
    return InferenceResponse(**result)
