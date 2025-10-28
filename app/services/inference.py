# app/services/inference.py

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B"

# 1) 모델/토크나이저 로딩 (앱 시작 시 1번만 실행되도록 모듈 전역에 둔다)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",  # GPU 있으면 GPU에, 없으면 CPU에
)

SYSTEM_PROMPT = (
    "You are a supportive Korean reflection assistant. "
    "Read the user's diary entry, understand their feelings, "
    "and reply in natural Korean with empathy. "
    "Do not diagnose like a doctor. Be kind and short."
)

def build_diary_instruction(diary_text: str) -> str:
    """
    일기 코멘트 전용 프롬프트. (서비스 value: 감정 피드백/코멘트)
    """
    return (
        "다음은 사용자의 오늘 일기야.\n"
        "1) 먼저 공감해줘.\n"
        "2) 사용자의 현재 감정을 한 줄로 요약해줘.\n"
        "3) 내일을 위한 부드러운 제안을 한 줄로 해줘.\n\n"
        f"[일기]\n{diary_text}\n"
        "---\n"
        "위 가이드만 지키고 한국어로 답해."
    )

def _generate(prompt: str, max_new_tokens: int = 256) -> str:
    """
    공통 생성 함수
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,  # 반복 줄이기
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 우리가 넣은 프롬프트가 앞부분에 그대로 섞여서 나올 수 있으니까 간단하게 잘라줌
    cleaned = decoded.replace(prompt, "").strip()
    return cleaned

def generate_comment_for_diary(diary_text: str) -> str:
    """
    외부에서 실제로 쓰는 함수.
    백엔드 엔드포인트에서 이 함수만 호출하면 됨.
    """
    user_instruction = build_diary_instruction(diary_text)

    full_prompt = f"<s>[INST] {SYSTEM_PROMPT}\n{user_instruction} [/INST]"

    return _generate(full_prompt)
