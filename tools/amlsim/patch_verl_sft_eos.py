#!/usr/bin/env python3
"""Enable EOS-token supervision in verl's SFTDataset."""
from __future__ import annotations

import argparse
from pathlib import Path


TARGET = "        # mask out the last token in response\n        loss_mask[min(prompt_length + response_length, loss_mask.size(0)) - 1] = 0\n"
REPLACEMENT = "        # Keep the response EOS token supervised so generation learns to stop.\n"


def apply(root: Path) -> Path:
    path = root / "verl" / "utils" / "dataset" / "sft_dataset.py"
    text = path.read_text(encoding="utf-8")
    if REPLACEMENT in text:
        return path
    count = text.count(TARGET)
    if count != 1:
        raise RuntimeError(f"{path}: expected one EOS-mask anchor, found {count}")
    backup = path.with_suffix(path.suffix + ".amlsim_eos.bak")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")
    path.write_text(text.replace(TARGET, REPLACEMENT), encoding="utf-8")
    return path


def check(root: Path) -> bool:
    path = root / "verl" / "utils" / "dataset" / "sft_dataset.py"
    text = path.read_text(encoding="utf-8")
    return REPLACEMENT in text and TARGET not in text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("apply", "check"))
    parser.add_argument("--verl-root", type=Path, default=Path("/workspace/verl"))
    args = parser.parse_args()
    if args.command == "apply":
        print(apply(args.verl_root))
        return 0
    status = check(args.verl_root)
    print({"sft_eos_supervised": status})
    return 0 if status else 1


if __name__ == "__main__":
    raise SystemExit(main())
