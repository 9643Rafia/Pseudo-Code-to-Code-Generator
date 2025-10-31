import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a CSV for human evaluation from predictions JSONL")
    parser.add_argument("--predictions", type=str, required=True, help="Path to predictions JSONL produced by evaluate.py")
    parser.add_argument("--out_csv", type=str, default="decoder_only/human_eval.csv")
    parser.add_argument("--max_rows", type=int, default=200)
    args = parser.parse_args()

    preds_path = Path(args.predictions)
    rows = []
    with open(preds_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.max_rows and i >= args.max_rows:
                break
            ex = json.loads(line)
            rows.append(
                {
                    "pseudo": ex.get("pseudo", ""),
                    "reference": ex.get("reference", ""),
                    "prediction": ex.get("prediction", ""),
                    "correct_syntax (y/n)": "",
                    "passes_tests (y/n)": "",
                    "comments": "",
                }
            )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pseudo",
                "reference",
                "prediction",
                "correct_syntax (y/n)",
                "passes_tests (y/n)",
                "comments",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()


