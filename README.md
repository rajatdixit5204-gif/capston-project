# Study Tip Generator — Generative AI Capstone

This project follows the ETMMGA304 manual's workflow: **Define → Prepare data → Adapt model → Evaluate → Demonstrate**. It fine-tunes GPT-2 on topic/tip examples and provides a Streamlit interface.

## Files
- `data/study_tips.json`: 30 topic-tip pairs; first 20 are training data, last 10 are held out.
- `train.py`: fine-tunes only the last GPT-2 Transformer block and saves `outputs/study_tip_gpt2/`.
- `evaluate.py`: compares base and adapted GPT-2 on the same 10 held-out topics and writes raw outputs to CSV.
- `app.py`: Streamlit demonstration interface.

## Run in Google Colab
1. Upload/extract this folder in Colab.
2. Select **Runtime → Change runtime type → T4 GPU** (recommended; CPU can be very slow).
3. Run:
```bash
!pip install -r requirements.txt
!python train.py
!python evaluate.py
```
Download the `outputs/` folder after execution.

## Run the app locally
After training, from this folder:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Evaluation — do not invent scores
Open `outputs/evaluation_results.csv`. For every base/adapted output, assign 0 (poor), 1 (partial), or 2 (strong) for relevance, readability, and usefulness. Calculate mean total score out of 6 for each model. Include at least one weak output and explain the failure. The score columns are blank intentionally; they must be based on actual outputs.

## Dataset note
The included small dataset is a demonstration dataset written for this project. Replace or extend it only with instructor-approved examples. Document its source, permissions, and train/held-out split in your report.

## Limitations and responsible use
GPT-2 is not a verified tutor and may produce vague, repetitive, or inaccurate advice. The small dataset demonstrates the workflow but does not establish reliable improvement. Review outputs before use; do not enter personal or sensitive information. The app discloses that its output is AI-generated.
