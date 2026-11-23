from pathlib import Path
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

st.set_page_config(page_title="Study Tip Generator", page_icon="📚")
st.title("📚 Study Tip Generator")
st.caption("AI-generated study suggestions based on your topic.")
st.info("Disclosure: This app uses GPT-2. Review generated tips; they may be vague or incorrect.")
MODEL_DIR = Path("outputs/study_tip_gpt2")

@st.cache_resource
def load_model(folder):
    tok = AutoTokenizer.from_pretrained(folder)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(folder).eval()
    return tok, model

topic = st.text_input("Enter a study topic", max_chars=120, placeholder="e.g. Binary Search")
if st.button("Generate Study Tip", type="primary"):
    if not topic.strip():
        st.warning("Please enter a topic first.")
    elif not (MODEL_DIR / "config.json").exists():
        st.error("Fine-tuned model missing. Run `python train.py` first.")
    else:
        with st.spinner("Generating..."):
            tok, model = load_model(str(MODEL_DIR))
            x = tok(f"Topic: {topic.strip()}\nStudy tip:", return_tensors="pt")
            with torch.inference_mode():
                y = model.generate(**x, do_sample=False, max_new_tokens=55,
                                   pad_token_id=tok.eos_token_id, no_repeat_ngram_size=3)
            answer = tok.decode(y[0, x["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        st.subheader("Generated Study Tip")
        st.write(answer or "No text generated. Try a more specific topic.")
st.divider()
st.caption("For learning support only. Verify technical facts with course materials.")
