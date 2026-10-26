# FinQA tools

Exploratory executable-reward pipeline for the official FinQA dataset.

The output protocol is exactly four lines: `Answer`, `Program`, `Evidence`, and `End: FINQA_END`. The deterministic scorer executes the program and checks its result against the gold answer; evidence contributes only a bounded bonus.

See `docs/finqa/README.md` for scope and reproduction commands. FinQA is a financial numerical reasoning benchmark, not a credit-risk dataset.
