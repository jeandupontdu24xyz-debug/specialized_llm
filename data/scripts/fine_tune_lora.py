from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

MODEL_NAME = "mistralai/Mistral-7B-v0.3"
DATA_PATH = "../data/processed/train.jsonl"
OUTPUT_DIR = "../models/fine_tuned"

dataset = load_dataset("json", data_files=DATA_PATH, split="train")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_fn(example):
    text = f"### Instruction:\n{example['prompt']}\n### Réponse:\n{example['response']}"
    return tokenizer(text, truncation=True, padding="max_length", max_length=1024)
dataset = dataset.map(tokenize_fn)

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, load_in_8bit=True, device_map="auto")

lora_config = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj","v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    num_train_epochs=1,
    learning_rate=2e-4,
    save_strategy="epoch",
    logging_steps=10,
    fp16=True,
)

trainer = Trainer(model=model, args=args, train_dataset=dataset)
trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Fine-tuning terminé et modèle sauvegardé.")
