
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

app = FastAPI(title="DiaryGarden-AI", version="0.1.0")

# 모든 JSON 응답에 charset=utf-8 명시 (PowerShell 출력 깨짐 완화)
@app.middleware("http")
async def ensure_utf8_content_type(request, call_next):
    response = await call_next(request)
    if response.headers.get("content-type", "").startswith("application/json"):
        response.headers["content-type"] = "application/json; charset=utf-8"
    return response

class InferenceRequest(BaseModel):
    text: str = Field(..., description="코멘트를 생성할 입력 텍스트(일기 등)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="선택 메타데이터")

class InferenceResponse(BaseModel):
    comment: str = Field(..., description="생성된 코멘트 한 단락")

@app.get("/health", summary="헬스체크")
def health():
    return {"ok": True}

@app.post(
    "/api/v1/inference",
    response_model=InferenceResponse,
    summary="코멘트 생성",
    description="입력 텍스트를 바탕으로 한국어 공감 코멘트를 한 단락 생성합니다.",
)
async def inference(req: InferenceRequest):
    from app.services.inference import CommentGenerator
    comment = await CommentGenerator.generate_comment(req.text, req.metadata)
    return JSONResponse(
        content=InferenceResponse(comment=comment).model_dump(),
        media_type="application/json; charset=utf-8",
    )
