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
ID2LABEL = {0: "happy", 1: "sad", 2: "angry", 3: "calm"}

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
def _build_chat(title: str, content: str, emotion: str) -> list[dict[str, str]]:
    user_input = f"제목: {title}\n내용: {content}\n지배 감정: {emotion}"
    return [
        {"role": "system", "content": (
            "역할: 따뜻하게 위로하는 친구처럼 한국어 코멘트를 쓰는 작가.\n"
            "규칙:\n"
            "1) 반드시 개행 없이 한 문장으로만 작성한다.\n"
            "2) 한글 기준 45~60자로 자연스럽게 맞춘다.\n"
            "3) 문장 끝은 마침표 하나로 끝내며, ?, !, 따옴표, 이모지, 해시태그 금지.\n"
            "4) AI, 언어모델, 시스템 등 자기 언급 금지.\n"
            "5) 입력 내용의 핵심 감정을 반드시 반영해 공감한다.\n"
            "6) 조언은 강요가 아닌 부드러운 제안 형태로 표현한다.\n"
            "6) 자기 자신이 아닌 일기의 본인인 타인에 대한 코멘트를 말해야 한다.\n"
            "\n"
            f"분석된 지배 감정은 '{emotion}'입니다. 이 감정을 중심으로 공감해주세요."

            "출력 예시:\n"
            "- 너 즐거운 하루였구나, 그 기분 오래 유지되도록 스스로를 칭찬해줘.\n"
            "- 너 오늘 많이 힘들었겠다, 잠시 쉬며 마음을 가볍게 만들어보자.\n"
            "- 너 정말 억울했을 것 같아, 네 감정을 인정하고 천천히 정리해보자.\n"  
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
        comment = cls._generate_comment(title, content, dominant_emo)

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
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)[0]

        # --- 여기부터 결과 처리 ---
        probs = probs.squeeze()

        num_labels = min(len(probs), len(ID2LABEL))

        # 차원 mismatch 경고
        if len(probs) != len(ID2LABEL):
            print(f"[WARN] logits/probs size({len(probs)}) != ID2LABEL size({len(ID2LABEL)})")

        # 가능한 범위까지만 매핑
        scores = {ID2LABEL[i]: float(probs[i]) for i in range(num_labels)}

        # 주요 감정
        dominant_idx = int(torch.argmax(probs[:num_labels]))
        dominant_emo = ID2LABEL[dominant_idx]

        # 디버그
        print("DEBUG logits shape:", logits.shape)

        return scores, dominant_emo


    @classmethod
    def _generate_comment(cls, title: str, text: str, emotion: str) -> str:
        tok, model = _Holder.llm_tok, _Holder.llm_model
        
        chat = _build_chat(title, text, emotion)
        inputs = tok.apply_chat_template(chat, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(**inputs, **GEN_KW)
            decoded = tok.batch_decode(out, skip_special_tokens=False)[0]
            
            if "<|im_start|>assistant" in decoded:
                decoded = decoded.split("<|im_start|>assistant")[-1]
            
            return _clean_text(decoded)
