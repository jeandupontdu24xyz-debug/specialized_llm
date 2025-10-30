"""
train_rlhf.py
-------------
Phase RLHF (PPO) qui utilise :
- un modèle causal (avec adaptateurs LoRA / QLoRA chargés)
- un reward model entraîné (classification scalar)
- le dataset pairwise (prompt/chosen/rejected) généré précédemment

Dépendances :
  pip install torch transformers datasets accelerate peft bitsandbytes trl wandb
"""

import os
import json
from pathlib import Path
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig, PeftType, get_peft_model, LoraConfig
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from transformers import AutoModelForSequenceClassification

# ====== CONFIG ======
BASE_MODEL = "meta-llama/Llama-3-8B"    # modèle de base (ou identifiant HF)
ADAPTER_PATH = "models/llama-lora-adapted"  # chemin vers adaptateurs LoRA (si existants)
REWARD_MODEL_PATH = "models/reward_model"
RLHF_DATA_PATH = "data/dataset_rlhf.jsonl"  # format: {"prompt","chosen","rejected"}
OUTPUT_DIR = "models/llama-rlhf-ppo"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# PPO Hyperparams (ajuster selon resources)
ppo_config = PPOConfig(
    model_name=BASE_MODEL,
    learning_rate=1.41e-5,
    ppo_epochs=4,
    # kl_ctl (beta, target) will be handled via scheduler in TRL; you may tune beta
    batch_size=1,               # PPO batch size (per device)
    mini_batch_size=1,          # inner minibatch
    forward_batch_size=1,       # batch size for forward pass (memory)
)

# ====== 1) Charger tokenizer et modèle causal (avec valeur head) ======
print(">> Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token  # required sometimes

print(">> Loading base causal model (value head will be added by TRL)...")
# Use TRL helper to add a value head
causal_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_4bit=False,    # si tu veux QLoRA, adapter la logique de chargement (bitsandbytes)
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto" if DEVICE == "cuda" else None,
)

# Si tu as un adapter LoRA déjà entraîné, applique-le (PeftModel)
if Path(ADAPTER_PATH).exists():
    print(f">> Applying PEFT adapter from {ADAPTER_PATH} ...")
    causal_model = PeftModel.from_pretrained(causal_model, ADAPTER_PATH)
    causal_model.eval()

# Wrap with value head for PPO (TRL)
print(">> Wrapping model with value head (AutoModelForCausalLMWithValueHead)...")
# Note: trl.AutoModelForCausalLMWithValueHead expects a transformers model
model_with_value = AutoModelForCausalLMWithValueHead.from_pretrained(causal_model.config.name_or_path, 
                                                                      revision=None, 
                                                                      torch_dtype=causal_model.dtype)

# If we loaded PeftModel above, model_with_value loading may require re-attaching adapters.
# Simpler workflow: instead, load base model into AutoModelForCausalLMWithValueHead then apply adapters:
# (If the above approach gives issues, use the alternative commented logic below.)
#
# Alternative:
# base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=...)
# model_with_value = AutoModelForCausalLMWithValueHead.from_pretrained(BASE_MODEL, torch_dtype=...)
# if Path(ADAPTER_PATH).exists():
#     model_with_value = PeftModel.from_pretrained(model_with_value, ADAPTER_PATH)

model = model_with_value.to(DEVICE)
model.config.use_cache = False  # required for training

# ====== 2) Charger reward model ======
print(">> Loading reward model for scoring...")
reward_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_PATH, use_fast=False)
reward_model = AutoModelForSequenceClassification.from_pretrained(REWARD_MODEL_PATH)
reward_model = reward_model.to(DEVICE)
reward_model.eval()

# Reward scoring function
def score_response(prompt_text, response_text):
    """
    Retourne un score scalaire (float) plus élevé = meilleur (aligné avec humain).
    Ici on utilise le reward_model qui a été entraîné comme un classif. binaire / regression.
    """
    text = prompt_text + "\n" + response_text
    enc = reward_tokenizer(text, truncation=True, padding="longest", return_tensors="pt", max_length=512).to(DEVICE)
    with torch.no_grad():
        out = reward_model(**enc)
        # out.logits shape (batch, 1) or (batch,2) depending on training; adapt:
        logits = out.logits.squeeze(-1)
        # if binary class logits, transform to scalar prob:
        if logits.dim() == 0:
            score = logits.item()
        else:
            # if shape (batch,2): assume logits for classes [bad, good], take softmax prob of good
            if out.logits.shape[-1] == 2:
                probs = torch.softmax(out.logits, dim=-1)
                score = probs[:, 1].cpu().item()
            else:
                score = logits.cpu().item()
    return float(score)

# ====== 3) Charger dataset RLHF (pairwise) ======
print(">> Loading RLHF dataset ...")
# Expected format per line: {"prompt": "...", "chosen": "...", "rejected": "..."}
with open(RLHF_DATA_PATH, "r", encoding="utf-8") as f:
    examples = [json.loads(l) for l in f if l.strip()]

# Convert to a simple list of prompts for which we will generate responses and compute rewards
# We'll create a training list where for each example we use the prompt and let the policy generate
train_prompts = [ex["prompt"] for ex in examples]

# ====== 4) PPO Trainer setup (TRL) ======
print(">> Configuring PPO Trainer ...")
ppo_trainer = PPOTrainer(
    model=model,
    ref_model=None,     # optional reference model (for KL constraint); TRL can create internally
    tokenizer=tokenizer,
    dataset=None,       # we'll use custom loop below feeding queries
    config=ppo_config,
    # log_with="wandb",   # optional: enable Weights & Biases
)

# ====== 5) Training loop (generation -> reward -> ppo step) ======
print(">> Starting RLHF PPO loop ...")
max_steps = 1000               # nombre total d'itérations PPO (ajuster)
gen_kwargs = {"max_length": 128, "do_sample": True, "top_k": 50, "top_p": 0.95, "temperature": 0.95}

for step_idx, prompt in enumerate(train_prompts):
    if step_idx >= max_steps:
        break

    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    # 1) Générer une réponse avec le current policy (model)
    # Use model.generate via tokenizer (ensure model.generate uses the policy head)
    generation_output = model.generate(**inputs, **gen_kwargs)
    response = tokenizer.decode(generation_output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # 2) Compute reward using reward_model
    reward_score = score_response(prompt, response)
    # Normalize reward if necessary (e.g. to [-1, 1] or [0,1]) depending on reward_model
    reward_tensor = torch.tensor([reward_score], dtype=torch.float32).to(DEVICE)

    # 3) Convert inputs and outputs to PPO-friendly tensors and call ppo_trainer.step
    # ppo_trainer.step expects (query_tensors, response_tensors, rewards)
    # TRL helper takes raw prompts and responses optionally; we'll use step API that accepts strings
    try:
        stats = ppo_trainer.step(prompt, response, reward_tensor)
    except Exception as e:
        print(f"Warning: PPO step failed at idx {step_idx} : {e}")
        continue

    if step_idx % 10 == 0:
        print(f"Step {step_idx} | Reward: {reward_score:.4f} | Stats: {stats}")

    # Save checkpoints periodically
    if step_idx % 200 == 0 and step_idx > 0:
        ckpt_dir = Path(OUTPUT_DIR) / f"ckpt_{step_idx}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving checkpoint to {ckpt_dir}")
        # Save peft adapters if model uses peft
        try:
            # if model is wrapped by PeftModel
            if isinstance(model, PeftModel):
                model.save_pretrained(ckpt_dir)
            else:
                # Use model.save_pretrained for full model (not recommended large)
                model.save_pretrained(ckpt_dir)
        except Exception as e:
            print(f"Warning: save failed: {e}")

print(">> PPO training finished. Saving final model ...")
# Save final adapter / model
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
try:
    if isinstance(model, PeftModel):
        model.save_pretrained(OUTPUT_DIR)
    else:
        model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
except Exception as e:
    print(f"Saving final model failed: {e}")

print(f"RLHF PPO terminée. Artefacts sauvegardés dans {OUTPUT_DIR}")
