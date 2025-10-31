## Decoder-only (GPT-2) — Pseudocode → Python

This repository also includes a simple decoder-only pipeline under `decoder_only/` built with Hugging Face Transformers.

Quickstart:

```
pip install -r requirements.txt

# 1) Prepare data (downloads SPOC automatically)
python decoder_only/prepare_data.py --out_dir decoder_only/processed

# 2) Fine-tune GPT-2
python decoder_only/train_gpt2.py --data_dir decoder_only/processed --model_name_or_path gpt2 --output_dir decoder_only/models/gpt2-spoc

# 3) Evaluate (BLEU, CodeBLEU if package available)
python decoder_only/evaluate.py --data_dir decoder_only/processed --model_dir decoder_only/models/gpt2-spoc --out_path decoder_only/predictions_test.jsonl

# 4) Launch Gradio app
python decoder_only/app.py --model_dir decoder_only/models/gpt2-spoc --server_port 7860
```

Notes:

- The training objective masks prompt tokens and learns to generate only the Python code.
- CodeBLEU is optional; if installing the `codebleu` package fails, evaluation will still report BLEU.
