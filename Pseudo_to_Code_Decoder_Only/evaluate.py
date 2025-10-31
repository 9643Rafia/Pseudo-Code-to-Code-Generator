import argparse
import json
from pathlib import Path
from typing import List, Tuple

import sacrebleu
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_gpt2 import PROMPT_TEMPLATE


def generate_codes(model, tokenizer, pseudos: List[str], max_new_tokens: int = 256, temperature: float = 0.2, top_p: float = 0.95) -> List[str]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    results: List[str] = []
    for pseudo in pseudos:
        prompt = PROMPT_TEMPLATE.format(pseudo=pseudo)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        results.append(text)
    return results


def try_codebleu(cands: List[str], refs: List[str]) -> float:
    try:
        # Prefer the public pip package API if available
        from codebleu import calc_codebleu
        score_dict = calc_codebleu.calc_codebleu(refs, cands, lang="python")
        return float(score_dict.get("codebleu", 0.0))
    except Exception:
        try:
            # Alternate API (some forks expose a function directly)
            import codebleu
            score = codebleu.corpus_code_bleu(cands, refs, lang="python")
            return float(score)
        except Exception:
            return -1.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model on SPOC test set with BLEU and CodeBLEU")
    parser.add_argument("--data_dir", type=str, default=str(Path(__file__).resolve().parent / "processed"))
    parser.add_argument("--model_dir", type=str, required=True, help="Path to fine-tuned model directory")
    parser.add_argument("--out_path", type=str, default=str(Path(__file__).resolve().parent / "predictions_test.jsonl"))
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    dataset = load_dataset("json", data_files={"test": str(data_dir / "test.jsonl")})
    test = dataset["test"]

    pseudos = test["pseudo"]
    refs = test["code"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir)

    cands = generate_codes(model, tokenizer, pseudos, max_new_tokens=args.max_new_tokens)

    # BLEU (sacrebleu expects list of references lists)
    bleu = sacrebleu.corpus_bleu(cands, [refs]).score

    codebleu = try_codebleu(cands, refs)

    # Save predictions
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for pseudo, ref, pred in zip(pseudos, refs, cands):
            f.write(json.dumps({"pseudo": pseudo, "reference": ref, "prediction": pred}, ensure_ascii=False) + "\n")

    print(json.dumps({"BLEU": bleu, "CodeBLEU": codebleu, "predictions_path": str(out_path)}, indent=2))


if __name__ == "__main__":
    main()


