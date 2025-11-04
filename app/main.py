# main.py 골격 필요 시 생성

from fastapi import FastAPI
from pydantic import BaseModel
from app.services.inference import CommentGenerator

app = FastAPI()

class InferenceRequest(BaseModel):
    text: str
    metadata: dict | None = None

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/api/v1/inference")
async def inference(req: InferenceRequest):
    comment = await CommentGenerator.generate_comment(req.text, req.metadata)
    return {"comment": comment}
