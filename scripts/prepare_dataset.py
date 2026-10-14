"""Convert a public fake-news dataset into the CSV this project expects.

Output schema (data/raw/news_dataset.csv): title, text, label   (1 = Fake, 0 = Real)

Usage
-----
# ISOT / Kaggle "Fake and Real News Dataset" (folder containing True.csv and Fake.csv)
python scripts/prepare_dataset.py --isot-dir path/to/folder

# Any other CSV: tell the script which columns to use and which label value means "fake"
python scripts/prepare_dataset.py --csv other.csv --text-col content --title-col headline \
    --label-col verdict --fake-value FAKE
"""

import argparse
import re
from pathlib import Path

import pandas as pd

# Source-specific giveaways that let a model "cheat" instead of learning real signal.
# ISOT real articles are almost all Reuters, e.g. "WASHINGTON (Reuters) - ...".
_LEAKAGE_PATTERNS = [
    re.compile(r"^\s*[^()\n]{0,80}\(Reuters\)\s*-?\s*", re.IGNORECASE),
    re.compile(r"\(Reuters\)", re.IGNORECASE),
    re.compile(r"\bReuters\b", re.IGNORECASE),
    re.compile(r"(featured|image|photo)\s+(image\s+)?(via|by|courtesy of)[^.\n]{0,60}", re.IGNORECASE),
    re.compile(r"pic\.twitter\.com/\S+", re.IGNORECASE),
    re.compile(r"21st century wire", re.IGNORECASE),
]


def strip_leakage(text: str) -> str:
    for pattern in _LEAKAGE_PATTERNS:
        text = pattern.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_isot(folder: Path) -> pd.DataFrame:
    fake = pd.read_csv(folder / "Fake.csv")
    real = pd.read_csv(folder / "True.csv")
    fake["label"] = 1
    real["label"] = 0
    return pd.concat([fake, real], ignore_index=True)[["title", "text", "label"]]


def load_generic(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.csv)
    out = pd.DataFrame(
        {
            "title": df[args.title_col] if args.title_col else "",
            "text": df[args.text_col],
            "label": (df[args.label_col].astype(str) == str(args.fake_value)).astype(int),
        }
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--isot-dir", type=Path, help="Folder containing True.csv and Fake.csv")
    p.add_argument("--csv", type=Path, help="Generic CSV (use with the options below)")
    p.add_argument("--text-col", default="text")
    p.add_argument("--title-col", default=None)
    p.add_argument("--label-col", default="label")
    p.add_argument("--fake-value", default="1", help="Value in label column that means FAKE")
    p.add_argument("--out", type=Path, default=Path("data/raw/news_dataset.csv"))
    p.add_argument("--min-words", type=int, default=30, help="Drop very short articles")
    args = p.parse_args()

    if args.isot_dir:
        df = load_isot(args.isot_dir)
    elif args.csv:
        df = load_generic(args)
    else:
        p.error("Provide --isot-dir or --csv")

    df = df.dropna(subset=["text"]).copy()
    df["title"] = df["title"].fillna("").astype(str)
    df["text"] = df["text"].astype(str).map(strip_leakage)
    df = df[df["text"].str.split().str.len() >= args.min_words]
    df = df.drop_duplicates(subset="text").sample(frac=1, random_state=42).reset_index(drop=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df[["title", "text", "label"]].to_csv(args.out, index=False)

    n_fake = int(df["label"].sum())
    print(f"Wrote {len(df)} articles to {args.out}  (fake={n_fake}, real={len(df) - n_fake})")


if __name__ == "__main__":
    main()
