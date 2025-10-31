import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


PROMPT_TEMPLATE = (
    "You are an expert Python programmer. Generate Python code that matches the pseudocode.\n\n"
    "Pseudocode:\n{pseudo}\n\nPython code:\n"
)


def build_example(example: Dict[str, str], tokenizer, max_length: int) -> Dict[str, List[int]]:
    pseudo = example["pseudo"]
    code = example["code"]
    prompt = PROMPT_TEMPLATE.format(pseudo=pseudo)

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    code_ids = tokenizer(code, add_special_tokens=False)["input_ids"]
    input_ids = prompt_ids + code_ids + [tokenizer.eos_token_id]
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]

    # Labels: mask prompt tokens
    labels = [-100] * min(len(prompt_ids), len(input_ids))
    if len(input_ids) > len(labels):
        labels += input_ids[len(labels):]
    labels = labels[: len(input_ids)]

    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


@dataclass
class SimpleDataCollator:
    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attention_mask = [], [], []
        for f in features:
            pad_len = max_len - len(f["input_ids"]) 
            input_ids.append(f["input_ids"] + [tokenizer.pad_token_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)
        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        return batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 for pseudo-code → Python code")
    parser.add_argument("--data_dir", type=str, default=str(Path(__file__).resolve().parent / "processed"), help="Directory containing train.jsonl, eval.jsonl, test.jsonl")
    parser.add_argument("--model_name_or_path", type=str, default="gpt2", help="Base model")
    parser.add_argument("--output_dir", type=str, default=str(Path(__file__).resolve().parent / "models" / "gpt2-spoc"))
    parser.add_argument("--max_length", type=int, default=768)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not (data_dir / "train.jsonl").exists():
        raise FileNotFoundError(f"Processed data not found in {data_dir}. Run prepare_data.py first.")

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def preprocess_batch(batch: Dict[str, List[str]]):
        examples = [build_example({"pseudo": p, "code": c}, tokenizer, args.max_length) for p, c in zip(batch["pseudo"], batch["code"])]
        return {
            "input_ids": [e["input_ids"] for e in examples],
            "labels": [e["labels"] for e in examples],
            "attention_mask": [e["attention_mask"] for e in examples],
        }

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(data_dir / "train.jsonl"),
            "validation": str(data_dir / "eval.jsonl"),
        },
    )
    tokenized = dataset.map(preprocess_batch, batched=True, remove_columns=dataset["train"].column_names)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=50,
        seed=args.seed,
        fp16=args.fp16,
        bf16=args.bf16,
        report_to=["none"],
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=SimpleDataCollator(),
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()


