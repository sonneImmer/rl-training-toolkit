#!/usr/bin/env python3
"""Convert UCI Taiwanese Bankruptcy data into fixed VERL train/dev/test splits."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import zipfile
from pathlib import Path

import pandas as pd


PREFERRED_FEATURES = (
    "ROA(C) before interest and depreciation before interest",
    "ROA(A) before interest and % after tax",
    "ROA(B) before interest and depreciation after tax",
    "Operating Gross Margin",
    "Realized Sales Gross Margin",
    "Operating Profit Rate",
    "Pre-tax net Interest Rate",
    "After-tax net Interest Rate",
    "Non-industry income and expenditure/revenue",
    "Continuous interest rate (after tax)",
    "Operating Expense Rate",
    "Research and development expense rate",
    "Cash flow rate",
    "Interest-bearing debt interest rate",
    "Tax rate (A)",
    "Net Value Per Share (B)",
    "Net Value Per Share (A)",
    "Net Value Per Share (C)",
    "Persistent EPS in the Last Four Seasons",
    "Cash Flow Per Share",
    "Revenue Per Share (Yuan ¥)",
    "Operating Profit Per Share (Yuan ¥)",
    "Per Share Net profit before tax (Yuan ¥)",
    "Realized Sales Gross Profit Growth Rate",
    "Operating Profit Growth Rate",
    "After-tax Net Profit Growth Rate",
    "Regular Net Profit Growth Rate",
    "Continuous Net Profit Growth Rate",
    "Total Asset Growth Rate",
    "Net Value Growth Rate",
    "Total Asset Return Growth Rate Ratio",
    "Cash Reinvestment %",
    "Current Ratio",
    "Quick Ratio",
    "Interest Expense Ratio",
    "Total debt/Total net worth",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_csv(source_zip: Path) -> tuple[pd.DataFrame, str]:
    with zipfile.ZipFile(source_zip) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"expected one CSV in archive, found {names}")
        with archive.open(names[0]) as handle:
            return pd.read_csv(handle), names[0]


def choose_features(columns: list[str], count: int) -> list[str]:
    normalized = {column.strip(): column for column in columns}
    selected = [normalized[name] for name in PREFERRED_FEATURES if name in normalized]
    for column in columns:
        if column not in selected and column.strip() != "Bankrupt?":
            selected.append(column)
        if len(selected) >= count:
            break
    if len(selected) < count:
        raise ValueError(f"dataset only provides {len(selected)} usable features")
    return selected[:count]


def stratified_three_way_split(
    frame: pd.DataFrame,
    label: str,
    seed: int,
    dev_fraction: float,
    test_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    train_parts = []
    dev_parts = []
    test_parts = []
    for value in sorted(frame[label].unique()):
        indices = list(frame.index[frame[label] == value])
        rng.shuffle(indices)
        test_size = max(1, round(len(indices) * test_fraction))
        dev_size = max(1, round(len(indices) * dev_fraction))
        test_parts.append(frame.loc[indices[:test_size]])
        dev_parts.append(frame.loc[indices[test_size:test_size + dev_size]])
        train_parts.append(frame.loc[indices[test_size + dev_size:]])
    train = pd.concat(train_parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    dev = pd.concat(dev_parts).sample(frac=1, random_state=seed + 1).reset_index(drop=True)
    test = pd.concat(test_parts).sample(frac=1, random_state=seed + 2).reset_index(drop=True)
    return train, dev, test


def cap_training_ratio(frame: pd.DataFrame, label: str, negative_ratio: int, seed: int) -> pd.DataFrame:
    positive = frame[frame[label] == 1]
    negative = frame[frame[label] == 0]
    keep_negative = min(len(negative), len(positive) * negative_ratio)
    sampled_negative = negative.sample(n=keep_negative, random_state=seed)
    return pd.concat([positive, sampled_negative]).sample(frac=1, random_state=seed + 7).reset_index(drop=True)


def format_number(value) -> str:
    return f"{float(value):.6g}"


def build_prompt(row, features: list[str]) -> str:
    facts = "\n".join(f"- {name.strip()}: {format_number(row[name])}" for name in features)
    return (
        "You are reviewing a company's normalized financial indicators for bankruptcy risk.\n"
        "Classify the company using only the supplied values. Do not invent facts.\n\n"
        f"Financial indicators ({len(features)} dimensions):\n{facts}\n\n"
        "Return exactly two lines:\n"
        "Risk: BANKRUPTCY or HEALTHY\n"
        "Evidence: cite at least two indicator names from the input with their values.\n"
        "/no_think"
    )


def convert_split(frame: pd.DataFrame, features: list[str], split: str, source_hash: str) -> pd.DataFrame:
    records = []
    for index, row in frame.iterrows():
        label = int(row["Bankrupt?"])
        feature_values = {name.strip(): float(row[name]) for name in features}
        sample_id = f"uci-taiwanese-bankruptcy/{split}/{index}"
        records.append({
            "prompt": [{"role": "user", "content": build_prompt(row, features)}],
            "reward_model": {"ground_truth": "BANKRUPTCY" if label else "HEALTHY"},
            "data_source": "finance_risk_taiwanese_bankruptcy",
            "extra_info": {
                "index": index,
                "sample_id": sample_id,
                "label": label,
                "feature_names": [name.strip() for name in features],
                "feature_values": feature_values,
                "source_sha256": source_hash,
            },
        })
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-count", type=int, default=36)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--train-negative-ratio", type=int, default=3)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    if args.dev_fraction + args.test_fraction >= 1:
        raise ValueError("dev_fraction + test_fraction must be below 1")
    frame, archive_member = find_csv(args.source_zip)
    frame.columns = [column.strip() for column in frame.columns]
    if "Bankrupt?" not in frame.columns:
        raise ValueError("Bankrupt? label column not found")
    if frame.isna().any().any():
        raise ValueError("dataset contains missing values")
    if frame.duplicated().any():
        raise ValueError("dataset contains duplicate rows")

    features = choose_features(frame.columns.tolist(), args.feature_count)
    natural_train, dev, test = stratified_three_way_split(
        frame, "Bankrupt?", args.seed, args.dev_fraction, args.test_fraction
    )
    train = cap_training_ratio(natural_train, "Bankrupt?", args.train_negative_ratio, args.seed)

    args.output.mkdir(parents=True)
    source_hash = sha256(args.source_zip)
    split_stats = {}
    for name, split in (("train", train), ("dev", dev), ("test", test)):
        converted = convert_split(split, features, name, source_hash)
        path = args.output / f"{name}.parquet"
        converted.to_parquet(path, index=False)
        positives = int(split["Bankrupt?"].sum())
        split_stats[name] = {
            "rows": len(split),
            "positive": positives,
            "negative": len(split) - positives,
            "positive_rate": positives / len(split),
            "sha256": sha256(path),
        }

    manifest = {
        "schema_version": "2.0.0",
        "source": "UCI Taiwanese Bankruptcy Prediction",
        "source_archive": str(args.source_zip),
        "source_archive_member": archive_member,
        "source_sha256": source_hash,
        "license": "CC BY 4.0",
        "original_rows": len(frame),
        "original_features": len(frame.columns) - 1,
        "original_positive_rate": float(frame["Bankrupt?"].mean()),
        "selected_feature_count": len(features),
        "selected_features": [name.strip() for name in features],
        "seed": args.seed,
        "split_policy": {
            "dev_fraction": args.dev_fraction,
            "test_fraction": args.test_fraction,
            "train_negative_ratio": args.train_negative_ratio,
            "dev_test_keep_natural_distribution": True,
        },
        "splits": split_stats,
        "contains_cot": False,
        "task": "binary_bankruptcy_risk_classification",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
