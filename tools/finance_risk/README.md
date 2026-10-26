# Finance risk tools

Scripts for the UCI Taiwanese Bankruptcy controlled experiment.

- `prepare_taiwanese_bankruptcy.py`: deterministic 36-feature train/dev/test conversion.
- `prepare_finance_risk_sft.py`: Risk/Evidence SFT records.
- `finance_risk_outcome_reward.py`: pure `[0, 1]` outcome reward used for controlled comparisons.
- `finance_risk_class_weighted_reward.py`: optional class-sensitive experimental reward.
- `launch_finance_sft.sh`, `launch_finance_grpo.sh`, `launch_finance_eval.sh`: container launchers.
- `summarize_finance_rollouts.py`: group-level diagnostic evidence.
- `build_finance_case_report.py`: compact case-study export.

See `docs/finance_risk/` for provenance, results, limitations, and reproduction steps.
