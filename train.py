import json, random
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

SEED, MODEL_ID = 42, "openai-community/gpt2"
DATA = Path("data/study_tips.json")
OUT = Path("outputs/study_tip_gpt2")
random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

items = json.loads(DATA.read_text(encoding="utf-8"))
train_items, held_out = items[:-10], items[-10:]
print(f"Training: {len(train_items)} | Held-out: {len(held_out)}")

tok = AutoTokenizer.from_pretrained(MODEL_ID)
tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)
model.requires_grad_(False)
model.transformer.h[-1].requires_grad_(True)
optimizer = torch.optim.AdamW(model.transformer.h[-1].parameters(), lr=2e-5)

logs = []
for epoch in range(3):
    model.train()
    random.shuffle(train_items)
    losses = []
    for item in train_items:
        prefix = tok.encode(f"Topic: {item['topic']}\nStudy tip:", add_special_tokens=False)
        target = tok.encode(" " + item["tip"], add_special_tokens=False) + [tok.eos_token_id]
        ids = torch.tensor([prefix + target], device=device)
        labels = torch.tensor([[-100] * len(prefix) + target], device=device)
        optimizer.zero_grad()
        loss = model(input_ids=ids, labels=labels).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.transformer.h[-1].parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
    mean_loss = sum(losses) / len(losses)
    logs.append(mean_loss)
    print(f"Epoch {epoch+1}/3 - mean loss: {mean_loss:.4f}")

OUT.mkdir(parents=True, exist_ok=True)
model.save_pretrained(OUT)
tok.save_pretrained(OUT)
Path("outputs/training_log.txt").write_text(
    f"Model: {MODEL_ID}\nDevice: {device}\nSeed: {SEED}\nTraining examples: {len(train_items)}\nHeld-out examples: {len(held_out)}\nEpochs: 3\nLearning rate: 2e-5\n" +
    "".join(f"Epoch {i+1} mean loss: {v:.6f}\n" for i,v in enumerate(logs)),
    encoding="utf-8"
)
print("Saved fine-tuned model:", OUT)
