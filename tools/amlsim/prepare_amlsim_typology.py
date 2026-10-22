#!/usr/bin/env python3
"""Generate AMLSim-compatible AML typology prompts with deterministic evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


SOURCE_REPOSITORY = "https://github.com/IBM/AMLSim.git"
SOURCE_REVISION = "7338a4bcb1af9bcfea2201ad7daccfe2a4d569ca"
TYPOLOGIES = ("FAN_IN", "FAN_OUT", "CYCLE", "NORMAL")
RULES = {
    "FAN_IN": "At least four distinct source accounts transfer to one hub account within 48 hours.",
    "FAN_OUT": "One hub account transfers to at least four distinct destination accounts within 48 hours.",
    "CYCLE": "At least four accounts form a closed directed transfer cycle within 72 hours.",
    "NORMAL": "No complete FAN_IN, FAN_OUT, or CYCLE pattern is present; near-patterns with a broken condition remain NORMAL.",
}

RULE_PROVENANCE = {
    "upstream_semantics": {
        "FAN_IN": "AMLSim models multiple originator accounts sending to one main beneficiary account.",
        "FAN_OUT": "AMLSim models one main originator account distributing to multiple beneficiary accounts.",
        "CYCLE": "AMLSim models directed transfers that return from the final member to the first member.",
    },
    "experiment_defined_parameters": {
        "minimum_distinct_counterparties": 4,
        "fan_in_window_hours": 48,
        "fan_out_window_hours": 48,
        "cycle_window_hours": 72,
        "normal_controls": ["broken_fan_in", "broken_fan_out", "broken_cycle", "ordinary"],
        "statement": (
            "Thresholds, time windows, negative-control construction, class balance, prompt format, "
            "and reward weights are experiment-specific choices. They are not claimed as AMLSim defaults "
            "or validated bank production rules."
        ),
    },
}


@dataclass
class Transaction:
    tx_id: str
    timestamp: datetime
    source: str
    destination: str
    amount: float
    channel: str

    def line(self) -> str:
        return (
            f"{self.tx_id} | {self.timestamp:%Y-%m-%d %H:%M} | "
            f"{self.source} -> {self.destination} | {self.amount:.2f} USD | {self.channel}"
        )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def account(case_id: int, index: int) -> str:
    return f"A{case_id:06d}_{index:02d}"


def make_tx(case_id: int, index: int, when: datetime, source: str, destination: str, amount: float, rng: random.Random) -> Transaction:
    return Transaction(
        tx_id=f"TX{case_id:06d}_{index:02d}",
        timestamp=when,
        source=source,
        destination=destination,
        amount=round(amount, 2),
        channel=rng.choice(("WIRE", "ACH", "TRANSFER", "CHECK")),
    )


def noise_transactions(case_id: int, start_index: int, base: datetime, accounts: list[str], count: int, rng: random.Random) -> list[Transaction]:
    result = []
    for offset in range(count):
        source = accounts[(2 * offset) % len(accounts)]
        destination = accounts[(2 * offset + 1) % len(accounts)]
        result.append(make_tx(
            case_id,
            start_index + offset,
            base + timedelta(hours=rng.randint(-120, 120)),
            source,
            destination,
            rng.uniform(25, 900),
            rng,
        ))
    return result


def suspicious_case(case_id: int, typology: str, rng: random.Random) -> tuple[list[Transaction], list[str], dict]:
    base = datetime(2025, 1, 1) + timedelta(days=rng.randint(0, 300), hours=rng.randint(0, 23))
    member_count = rng.randint(5, 7)
    members = [account(case_id, index) for index in range(24)]
    pattern = []
    if typology == "FAN_IN":
        hub = members[0]
        for index, source in enumerate(members[1:member_count]):
            pattern.append(make_tx(case_id, index, base + timedelta(hours=rng.randint(0, 40)), source, hub, rng.uniform(800, 4800), rng))
    elif typology == "FAN_OUT":
        hub = members[0]
        for index, destination in enumerate(members[1:member_count]):
            pattern.append(make_tx(case_id, index, base + timedelta(hours=rng.randint(0, 40)), hub, destination, rng.uniform(800, 4800), rng))
    elif typology == "CYCLE":
        cycle_members = members[:member_count]
        amount = rng.uniform(2400, 6800)
        for index, source in enumerate(cycle_members):
            destination = cycle_members[(index + 1) % len(cycle_members)]
            pattern.append(make_tx(case_id, index, base + timedelta(hours=index * rng.randint(2, 8)), source, destination, amount * (0.96 ** index), rng))
    else:
        raise ValueError(typology)
    noise = noise_transactions(case_id, len(pattern), base, members[member_count + 1:], rng.randint(3, 6), rng)
    transactions = pattern + noise
    rng.shuffle(transactions)
    metadata = {"pattern_transaction_count": len(pattern), "noise_transaction_count": len(noise)}
    return transactions, [item.tx_id for item in pattern], metadata


def normal_case(case_id: int, rng: random.Random) -> tuple[list[Transaction], list[str], dict]:
    base = datetime(2025, 1, 1) + timedelta(days=rng.randint(0, 300), hours=rng.randint(0, 23))
    variant = rng.choice(("broken_fan_in", "broken_fan_out", "broken_cycle", "ordinary"))
    members = [account(case_id, index) for index in range(24)]
    transactions = []
    if variant == "broken_fan_in":
        hub = members[0]
        for index, source in enumerate(members[1:5]):
            transactions.append(make_tx(case_id, index, base + timedelta(hours=index * 30), source, hub, rng.uniform(100, 900), rng))
    elif variant == "broken_fan_out":
        hub = members[0]
        for index, destination in enumerate(members[1:5]):
            transactions.append(make_tx(case_id, index, base + timedelta(hours=index * 30), hub, destination, rng.uniform(100, 900), rng))
    elif variant == "broken_cycle":
        chain = members[:5]
        for index in range(len(chain) - 1):
            transactions.append(make_tx(case_id, index, base + timedelta(hours=index * 8), chain[index], chain[index + 1], rng.uniform(150, 1200), rng))
    transactions.extend(noise_transactions(case_id, len(transactions), base, members[5:], rng.randint(5, 8), rng))
    rng.shuffle(transactions)
    return transactions, [], {"normal_variant": variant, "pattern_transaction_count": 0, "noise_transaction_count": len(transactions)}


def build_prompt(transactions: list[Transaction]) -> str:
    rules = "\n".join(f"- {name}: {description}" for name, description in RULES.items())
    tx_lines = "\n".join(item.line() for item in sorted(transactions, key=lambda item: (item.timestamp, item.tx_id)))
    return (
        "You are an anti-money-laundering analyst. Apply only the supplied deterministic rules.\n\n"
        f"Rules:\n{rules}\n\n"
        f"Transactions:\n{tx_lines}\n\n"
        "Return exactly three lines:\n"
        "Risk: SUSPICIOUS or NORMAL\n"
        "Typology: FAN_IN, FAN_OUT, CYCLE, or NORMAL\n"
        "Evidence: comma-separated transaction IDs that form the complete suspicious pattern, or NONE for NORMAL.\n"
        "/no_think"
    )


def response_for(typology: str, evidence: list[str]) -> str:
    risk = "NORMAL" if typology == "NORMAL" else "SUSPICIOUS"
    evidence_text = ", ".join(evidence) if evidence else "NONE"
    return f"Risk: {risk}\nTypology: {typology}\nEvidence: {evidence_text}"


def generate_split(split: str, count: int, seed: int, start_case_id: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rng = random.Random(seed)
    labels = [TYPOLOGIES[index % len(TYPOLOGIES)] for index in range(count)]
    rng.shuffle(labels)
    rl_records = []
    sft_records = []
    counts = {name: 0 for name in TYPOLOGIES}
    for offset, typology in enumerate(labels):
        case_id = start_case_id + offset
        case_rng = random.Random(seed * 1_000_003 + case_id)
        if typology == "NORMAL":
            transactions, evidence, metadata = normal_case(case_id, case_rng)
        else:
            transactions, evidence, metadata = suspicious_case(case_id, typology, case_rng)
        prompt_text = build_prompt(transactions)
        risk = "NORMAL" if typology == "NORMAL" else "SUSPICIOUS"
        ground_truth = json.dumps({"risk": risk, "typology": typology, "evidence": evidence}, sort_keys=True)
        sample_id = f"amlsim-typology/{split}/{case_id}"
        extra = {
            "sample_id": sample_id,
            "case_id": case_id,
            "risk": risk,
            "typology": typology,
            "evidence_tx_ids": evidence,
            "all_tx_ids": [item.tx_id for item in transactions],
            "source_revision": SOURCE_REVISION,
            **metadata,
        }
        rl_records.append({
            "prompt": [{"role": "user", "content": prompt_text}],
            "reward_model": {"ground_truth": ground_truth},
            "data_source": "amlsim_typology_rules_v1",
            "extra_info": extra,
        })
        sft_records.append({"prompt": prompt_text, "response": response_for(typology, evidence), "sample_id": sample_id})
        counts[typology] += 1
    return pd.DataFrame(rl_records), pd.DataFrame(sft_records), counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=2400)
    parser.add_argument("--dev-size", type=int, default=400)
    parser.add_argument("--test-size", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    (args.output / "rl").mkdir()
    (args.output / "sft").mkdir()

    split_specs = (("train", args.train_size, args.seed, 0), ("dev", args.dev_size, args.seed + 1, 100_000), ("test", args.test_size, args.seed + 2, 200_000))
    split_manifest = {}
    for split, count, seed, start in split_specs:
        rl_frame, sft_frame, counts = generate_split(split, count, seed, start)
        rl_path = args.output / "rl" / f"{split}.parquet"
        sft_path = args.output / "sft" / f"{split}.parquet"
        rl_frame.to_parquet(rl_path, index=False)
        sft_frame.to_parquet(sft_path, index=False)
        split_manifest[split] = {
            "rows": count,
            "class_counts": counts,
            "rl_sha256": sha256(rl_path),
            "sft_sha256": sha256(sft_path),
        }

    rule_cards = {
        "schema_version": "1.0.0",
        "source": "AMLSim typology semantics",
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": SOURCE_REVISION,
        "rules": RULES,
        "rule_provenance": RULE_PROVENANCE,
        "reward_contract": {
            "risk": "binary SUSPICIOUS/NORMAL",
            "typology": list(TYPOLOGIES),
            "evidence": "exact transaction IDs forming the injected suspicious pattern",
        },
    }
    (args.output / "rule_cards.json").write_text(json.dumps(rule_cards, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "1.0.0",
        "dataset_id": "amlsim_typology_rules_v1",
        "source_repository": "IBM/AMLSim",
        "source_repository_url": SOURCE_REPOSITORY,
        "source_revision": rule_cards["source_revision"],
        "license": "Apache-2.0",
        "generation": {
            "method": "deterministic controlled typology generator",
            "uses_original_amlsim_simulator": False,
            "rule_provenance": RULE_PROVENANCE,
        },
        "seed": args.seed,
        "typologies": list(TYPOLOGIES),
        "contains_free_text_cot": False,
        "splits": split_manifest,
        "limitations": [
            "Synthetic typology cases do not represent real-bank prevalence.",
            "Rules are deterministic and may be easier than unseen real AML behavior.",
            "Experiment thresholds and reward weights have not been validated as bank production policy.",
            "Account identities do not overlap across splits.",
        ],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
