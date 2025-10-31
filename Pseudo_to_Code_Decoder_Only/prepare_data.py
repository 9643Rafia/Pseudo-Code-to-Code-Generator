import argparse
import json
import os
import random
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Tuple

import requests


SPOC_ZIP_URL = "https://sumith1896.github.io/spoc/data/spoc.zip"


def download_file(url: str, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)


def ensure_spoc_dataset(root_dir: Path) -> Path:
    data_root = root_dir / "data"
    split_dir = data_root / "train" / "split"
    if split_dir.exists():
        return split_dir

    zip_path = data_root / "spoc.zip"
    if not zip_path.exists():
        print(f"Downloading SPOC dataset to {zip_path} ...")
        download_file(SPOC_ZIP_URL, zip_path)

    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        tmp_extract_dir = data_root / "_spoc_tmp"
        if tmp_extract_dir.exists():
            shutil.rmtree(tmp_extract_dir)
        zf.extractall(tmp_extract_dir)

        # The zip contains a top-level 'spoc/' folder
        inner = tmp_extract_dir / "spoc" / "train" / "split"
        if not inner.exists():
            raise FileNotFoundError(
                f"Unexpected archive structure; expected {inner} inside zip"
            )

        # Move into data/train/split
        (data_root / "train").mkdir(parents=True, exist_ok=True)
        final_split_dir = data_root / "train" / "split"
        if final_split_dir.exists():
            shutil.rmtree(final_split_dir)
        shutil.move(str(inner), str(final_split_dir))
        # Clean temp
        shutil.rmtree(tmp_extract_dir)

    return split_dir


def iter_spoc_pairs(split_dir: Path, split: str) -> Iterable[Tuple[str, str]]:
    tsv_path = split_dir / f"spoc-train-{split}.tsv"
    if not tsv_path.exists():
        raise FileNotFoundError(f"Missing split file: {tsv_path}")

    header = None
    with open(tsv_path, "r", encoding="utf-8") as fin:
        first = fin.readline()
        if not first:
            return
        cols = first.rstrip("\n").split("\t")
        if "text" in cols and "code" in cols:
            header = cols
        else:
            # No header; rewind file
            fin.seek(0)

        for line in fin:
            parts = line.rstrip("\n").split("\t")
            if header:
                row: Dict[str, str] = dict(zip(header, parts))
                text, code = row.get("text", ""), row.get("code", "")
            else:
                if len(parts) < 2:
                    continue
                text, code = parts[0], parts[1]
            if not text or not code:
                continue
            yield text, code


def write_jsonl(examples: Iterable[Dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fout:
        for ex in examples:
            fout.write(json.dumps(ex, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SPOC pairs for decoder-only training")
    parser.add_argument("--root", type=str, default=str(Path(__file__).resolve().parents[1]), help="Project root (where 'data/' will be located)")
    parser.add_argument("--out_dir", type=str, default=str(Path(__file__).resolve().parent / "processed"), help="Directory to write processed JSONL files")
    parser.add_argument("--max_examples", type=int, default=0, help="Optional cap for examples per split (0 = no cap)")
    args = parser.parse_args()

    root_dir = Path(args.root)
    out_dir = Path(args.out_dir)

    split_dir = ensure_spoc_dataset(root_dir)

    rng = random.Random(13)

    for split_name in ("train", "eval", "test"):
        pairs = iter_spoc_pairs(split_dir, split_name)
        count = 0

        def gen() -> Iterable[Dict[str, str]]:
            nonlocal count
            for text, code in pairs:
                # Normalize line endings
                normalized_text = text.strip()
                normalized_code = code.rstrip() + "\n"
                ex = {
                    "pseudo": normalized_text,
                    "code": normalized_code,
                }
                yield ex
                count += 1
                if args.max_examples and count >= args.max_examples:
                    break

        out_path = out_dir / f"{split_name}.jsonl"
        write_jsonl(gen(), out_path)
        print(f"Wrote {count} examples to {out_path}")


if __name__ == "__main__":
    main()


