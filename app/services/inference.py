# app/services/inference.py
from __future__ import annotations
import os
from typing import Optional, Dict, Any, List

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ----------------------------------------------------------------
# 1. 설정 및 상수
# ----------------------------------------------------------------

# 감정 분석 모델 경로 (학습된 모델 경로)
EMOTION_MODEL_PATH = "./emotion-model"

# 감정 라벨 (train.py와 동일)
ID2LABEL = {0: "happy", 1: "sad", 2: "angry", 3: "calm"}

# ----------------------------------------------------------------
# 2. 모델 홀더 (Emotion Model만 로드)
# ----------------------------------------------------------------
class _Holder:
    emo_tok = None
    emo_model = None

    device = (
        "cuda" if torch.cuda.is_available()
        else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
              else "cpu")
    )

    @classmethod
    def load(cls):
        if cls.emo_model is not None:
            return

        print(f"Loading emotion model on {cls.device}...")

        # 감정 분석 모델 로드
        path = EMOTION_MODEL_PATH if os.path.exists(EMOTION_MODEL_PATH) else "monologg/koelectra-base-v3-discriminator"
        cls.emo_tok = AutoTokenizer.from_pretrained(path)
        cls.emo_model = AutoModelForSequenceClassification.from_pretrained(path)
        cls.emo_model.to(cls.device)
        cls.emo_model.eval()

# ----------------------------------------------------------------
# 3. 메인 서비스 (감정 분석만)
# ----------------------------------------------------------------
class InferenceService:
    @classmethod
    async def generate_response(cls, title: str, content: str) -> dict:
        _Holder.load()

        # 감정 분석 실행
        emo_scores, dominant_emo = cls._predict_emotion(content)

        return {
            "dominantEmotion": dominant_emo,
            "emotionScores": emo_scores
        }

    @classmethod
    def _predict_emotion(cls, text: str):
        tok, model = _Holder.emo_tok, _Holder.emo_model
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)[0]

        probs = probs.squeeze()

        num_labels = min(len(probs), len(ID2LABEL))

        # 차원 mismatch 경고
        if len(probs) != len(ID2LABEL):
            print(f"[WARN] logits/probs size({len(probs)}) != ID2LABEL size({len(ID2LABEL)})")

        # 감정 점수 매핑
        scores = {ID2LABEL[i]: float(probs[i]) for i in range(num_labels)}

        # 주요 감정
        dominant_idx = int(torch.argmax(probs[:num_labels]))
        dominant_emo = ID2LABEL[dominant_idx]

        print("DEBUG logits shape:", logits.shape)

        return scores, dominant_emo
