import csv, json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "openai-community/gpt2"
OUT = Path("outputs/study_tip_gpt2")
if not (OUT / "config.json").exists():
    raise FileNotFoundError("Run python train.py first.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
items = json.loads(Path("data/study_tips.json").read_text(encoding="utf-8"))
held_out = items[-10:]

def load(folder):
    tok = AutoTokenizer.from_pretrained(folder)
    tok.pad_token = tok.eos_token
    net = AutoModelForCausalLM.from_pretrained(folder).to(device).eval()
    return tok, net

base_tok, base = load(MODEL_ID)
tuned_tok, tuned = load(str(OUT))

def generate(net, tok, topic):
    x = tok(f"Topic: {topic}\nStudy tip:", return_tensors="pt").to(device)
    with torch.inference_mode():
        y = net.generate(**x, do_sample=False, max_new_tokens=55,
                         pad_token_id=tok.eos_token_id, no_repeat_ngram_size=3)
    return tok.decode(y[0, x["input_ids"].shape[1]:], skip_special_tokens=True).strip()

fields = ["topic","reference_tip","base_output","adapted_output",
          "base_relevance","base_readability","base_usefulness","base_total",
          "adapted_relevance","adapted_readability","adapted_usefulness","adapted_total",
          "failure_analysis"]
Path("outputs").mkdir(exist_ok=True)
with open("outputs/evaluation_results.csv","w",newline="",encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
    for item in held_out:
        b = generate(base, base_tok, item["topic"])
        a = generate(tuned, tuned_tok, item["topic"])
        w.writerow({"topic":item["topic"],"reference_tip":item["tip"],
                    "base_output":b,"adapted_output":a})
        print(f"\n{item['topic']}\nBASE: {b}\nADAPTED: {a}")
print("\nSaved outputs/evaluation_results.csv")
print("Fill score columns after reviewing actual outputs. Rubric: 0 poor, 1 partial, 2 strong.")
