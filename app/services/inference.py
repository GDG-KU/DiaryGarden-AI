# app/services/inference.py
from __future__ import annotations
import os
from typing import Optional, Dict, Any, List

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification

# ----------------------------------------------------------------
# 1. 설정 및 상수
# ----------------------------------------------------------------
# LLM 모델 (댓글 생성용)
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B").strip()

# 감정 분석 모델 경로 (학습된 모델이 있는 경로)
EMOTION_MODEL_PATH = "./emotion-model"

# 감정 라벨 (train.py와 동일해야 함)
ID2LABEL = {0: "joy", 1: "sadness", 2: "anger", 3: "neutral"}

GEN_KW = dict(
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.2,
    no_repeat_ngram_size=4,
)

STOP_STRINGS = ["<|endofturn|>", "<|stop|>", "<|im_end|>"]

# ----------------------------------------------------------------
# 2. 모델 홀더 (LLM + Emotion Model 로드)
# ----------------------------------------------------------------
class _Holder:
    # LLM
    llm_tok = None
    llm_model = None
    
    # Emotion Model
    emo_tok = None
    emo_model = None

    device = (
        "cuda" if torch.cuda.is_available()
        else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
              else "cpu")
    )

    @classmethod
    def load(cls):
        # 이미 로드되었으면 패스
        if cls.llm_model is not None and cls.emo_model is not None:
            return

        print(f"Loading models on {cls.device}...")

        # (A) LLM 로드
        if cls.llm_model is None:
            cls.llm_tok = AutoTokenizer.from_pretrained(HF_MODEL_ID, use_fast=True)
            if cls.device == "cpu":
                cls.llm_model = AutoModelForCausalLM.from_pretrained(
                    HF_MODEL_ID, device_map={"": "cpu"}, torch_dtype=torch.float32, low_cpu_mem_usage=True
                )
            else:
                cls.llm_model = AutoModelForCausalLM.from_pretrained(
                    HF_MODEL_ID, device_map="auto", torch_dtype=torch.float16, low_cpu_mem_usage=True
                )

        # (B) 감정 분석 모델 로드
        if cls.emo_model is None:
            # 학습된 모델이 없으면 기본 모델 로드 (에러 방지용)
            path = EMOTION_MODEL_PATH if os.path.exists(EMOTION_MODEL_PATH) else "monologg/koelectra-base-v3-discriminator"
            cls.emo_tok = AutoTokenizer.from_pretrained(path)
            cls.emo_model = AutoModelForSequenceClassification.from_pretrained(path)
            cls.emo_model.to(cls.device)
            cls.emo_model.eval()

# ----------------------------------------------------------------
# 3. 유틸리티 함수
# ----------------------------------------------------------------
def _build_chat(title: str, content: str) -> list[dict[str, str]]:
    user_input = f"제목: {title}\n내용: {content}"
    return [
        {"role": "system", "content": (
            "너는 사용자의 일기를 읽고 따뜻하게 공감해주는 친구야. "
            "반드시 한 문장으로 간결하게, 해요체로 부드럽게 위로하거나 칭찬해줘."
        )},
        {"role": "user", "content": user_input},
    ]

def _clean_text(text: str) -> str:
    # 특수 토큰 및 불필요한 기호 제거
    for s in STOP_STRINGS:
        if s in text:
            text = text.split(s)[0]
    return text.strip().strip('"').strip("'")

# ----------------------------------------------------------------
# 4. 메인 서비스 클래스
# ----------------------------------------------------------------
class InferenceService:
    @classmethod
    async def generate_response(cls, title: str, content: str) -> dict:
        _Holder.load()
        
        # 1. 감정 분석 실행
        emo_scores, dominant_emo = cls._predict_emotion(content)

        # 2. LLM 코멘트 생성
        comment = cls._generate_comment(title, content)

        return {
            "comment": comment,
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
            # Softmax로 확률 변환
            probs = F.softmax(outputs.logits, dim=-1)[0]
            
        # 결과 딕셔너리 생성
        scores = {ID2LABEL[i]: float(probs[i]) for i in range(len(ID2LABEL))}
        # 가장 높은 점수의 감정 찾기
        dominant = max(scores, key=scores.get)
        
        return scores, dominant

    @classmethod
    def _generate_comment(cls, title: str, text: str) -> str:
        tok, model = _Holder.llm_tok, _Holder.llm_model
        
        chat = _build_chat(title, text)
        inputs = tok.apply_chat_template(chat, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(**inputs, **GEN_KW)
            decoded = tok.batch_decode(out, skip_special_tokens=False)[0]
            
            # 답변 부분만 추출 (모델마다 다를 수 있음, 일반적인 파싱)
            if "<|im_start|>assistant" in decoded:
                decoded = decoded.split("<|im_start|>assistant")[-1]
            
            return _clean_text(decoded)
