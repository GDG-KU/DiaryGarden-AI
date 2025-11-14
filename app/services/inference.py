# app/services/inference.py
from __future__ import annotations
import os
from typing import Optional, Dict, Any, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

STOP_STRINGS = ["<|endofturn|>", "<|stop|>", "<|im_end|>"]

def _strip_after_stop(text: str) -> str:
    for s in STOP_STRINGS:
        if s and s in text:
            text = text.split(s)[0]
    return text

def _clean_text(t: str) -> str:
    t = t.replace("\u200b", "")
    t = re.sub(r"저는\s*AI[^.?!\n]*[.?!]?", "", t, flags=re.IGNORECASE)
    t = re.sub(r"I am an AI[^.?!\n]*[.?!]?", "", t, flags=re.IGNORECASE)
    t = t.replace("*", "")
    t = re.sub(r"[\"“”‘’]+", "", t)
    # 공백 정리
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _to_one_sentence(t: str) -> str:
    t = t.replace("\r", " ").replace("\n", " ")
    sents = re.split(r"(?:(?<=[\.!?])\s+|(?<=다\.)\s+|(?<=요\.)\s+)", t)
    for s in sents:
        s = s.strip(" '\"`")
        if s:
            if not re.search(r"[\.!?]$", s):
                s += "."
            return s
    return t if t.endswith(".") else (t + ".")

# .env 에서 바꿀 수 있음
HF_MODEL_ID = os.getenv(
    "HF_MODEL_ID",
    "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B",
).strip()
# (선택) LoRA 어댑터 쓰면 여기 모델 repo/id 넣어두기
USE_LORA_ADAPTER_ID = os.getenv("USE_LORA_ADAPTER_ID", "").strip()

# CPU면 토큰 수를 너무 크게 잡지 말자 (속도 ↑)


GEN_KW = dict(
    max_new_tokens=70,
    do_sample=True,
    temperature=1.5,
    top_p=0.85,
    repetition_penalty=1.2,
    no_repeat_ngram_size=4,
)


STOP_STRINGS = ["<|endofturn|>", "<|stop|>"]   # 모델 템플릿 기준 종료 토큰 후보

class _Holder:
    tok = None
    model = None
    device = (
        "cuda" if torch.cuda.is_available()
        else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
              else "cpu")
    )

    @classmethod
    def load(cls):
     if cls.model is not None:
        return

    # CPU 최적화(스레드 수 조정)
     if cls.device == "cpu":
        try:
            torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
        except Exception:
            pass

    # 토크나이저
     cls.tok = AutoTokenizer.from_pretrained(HF_MODEL_ID, use_fast=True)

     if cls.device == "cpu":
        # ✅ CPU에서는 device_map을 'cpu'로 고정하고 dtype은 float32로
        cls.model = AutoModelForCausalLM.from_pretrained(
            HF_MODEL_ID,
            device_map={"": "cpu"},
            dtype=torch.float32,

            low_cpu_mem_usage=True,
        )
     else:
        # GPU/MPS일 때만 bfloat16 등 사용
        dtype = torch.bfloat16 if cls.device == "cuda" else None
        cls.model = AutoModelForCausalLM.from_pretrained(
            HF_MODEL_ID,
            device_map="auto",
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )

    if USE_LORA_ADAPTER_ID:
        try:
            from peft import PeftModel
            cls.model = PeftModel.from_pretrained(cls.model, USE_LORA_ADAPTER_ID)
            print(f"[LoRA] Loaded adapter: {USE_LORA_ADAPTER_ID}")
        except Exception as e:
            print(f"[WARN] Failed to load LoRA adapter: {e}")
# app/services/inference.py
def _build_chat(user_text: str) -> list[dict[str, str]]:
    return [
        {"role": "tool_list", "content": ""},
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

            "출력 예시:\n"
            "- 너 즐거운 하루였구나, 그 기분 오래 유지되도록 스스로를 칭찬해줘.\n"
            "- 너 오늘 많이 힘들었겠다, 잠시 쉬며 마음을 가볍게 만들어보자.\n"
            "- 너 정말 억울했을 것 같아, 네 감정을 인정하고 천천히 정리해보자.\n"
        )},
        {"role": "user", "content": user_text},
    ]


def _strip_after_stop(text: str) -> str:
    for s in STOP_STRINGS:
        if s and s in text:
            text = text.split(s)[0]
    return text.strip()

class CommentGenerator:
    @classmethod
    async def generate_comment(cls, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        _Holder.load()
        tok, model = _Holder.tok, _Holder.model

        # 템플릿 적용 (HyperCLOVAX 계열은 apply_chat_template 권장)
        chat = _build_chat(text)
        inputs = tok.apply_chat_template(
            chat,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(**inputs, **GEN_KW)
            decoded = tok.batch_decode(out, skip_special_tokens=False)[0]
            if "<|im_start|>assistant" in decoded:
                decoded = decoded.split("<|im_start|>assistant")[-1]
            decoded = _clean_text(_strip_after_stop(decoded))
            decoded = _to_one_sentence(decoded)   # ← 최종 한 문장 강제
            return decoded
