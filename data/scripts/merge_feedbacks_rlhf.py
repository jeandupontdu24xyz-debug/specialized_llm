import json, glob

DATA_PATH = "../data/processed/train.jsonl"
FEEDBACK_DIR = "../data/feedbacks/"
OUTPUT_PATH = "../data/processed/rlhf_dataset.jsonl"

merged = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        merged.append(json.loads(line))

for file in glob.glob(FEEDBACK_DIR + "*.jsonl"):
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            merged.append(json.loads(line))

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for item in merged:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Fusion des feedbacks terminée ({len(merged)} exemples).")
