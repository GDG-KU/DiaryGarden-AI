import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset

# -------------------------------------------------------------
# 1) 기본 설정
# -------------------------------------------------------------
MODEL_NAME = "monologg/koelectra-base-v3-discriminator"

# 4개 감정 라벨
label2id = {
    "joy": 0,
    "sadness": 1,
    "anger": 2,
    "neutral": 3,
}
id2label = {v: k for k, v in label2id.items()}

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# -------------------------------------------------------------
# 2) 데이터셋 로드
# -------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "emotion_data.json")

print(f"Loading dataset from: {DATA_FILE}")
dataset = load_dataset("json", data_files=DATA_FILE)

# 자동 train/test split (8:2)
dataset = dataset["train"].train_test_split(test_size=0.2)

# -------------------------------------------------------------
# 3) 전처리 함수 (수정됨: remove_columns 추가)
# -------------------------------------------------------------
def preprocess(batch):
    enc = tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )
    # 'label' 컬럼(문자열)을 'labels' 컬럼(숫자)으로 변환
    enc["labels"] = [label2id[x] for x in batch["label"]]
    return enc

# [중요] 기존의 문자열 컬럼('text', 'label')을 삭제해야 에러가 안 납니다.
dataset = dataset.map(
    preprocess, 
    batched=True, 
    remove_columns=dataset["train"].column_names
)

# -------------------------------------------------------------
# 4) 모델 준비
# -------------------------------------------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=4,
    label2id=label2id,
    id2label=id2label,
)

# -------------------------------------------------------------
# 5) 학습 설정
# -------------------------------------------------------------
OUTPUT_DIR = os.path.join(os.getcwd(), "emotion-model")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",  
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True,
    fp16=torch.cuda.is_available(),
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)

# -------------------------------------------------------------
# 6) 학습 시작
# -------------------------------------------------------------
print("🚀 학습을 시작합니다...")
trainer.train()

# -------------------------------------------------------------
# 7) 결과 저장
# -------------------------------------------------------------
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\n🎉 Training Finished! 모델이 '{OUTPUT_DIR}'에 저장되었습니다.")