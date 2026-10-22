import json
import subprocess
import sys
from pathlib import Path


def test_generator_records_upstream_and_experiment_rule_boundaries(tmp_path: Path) -> None:
    output = tmp_path / "dataset"
    subprocess.run(
        [
            sys.executable,
            "tools/amlsim/prepare_amlsim_typology.py",
            "--output",
            str(output),
            "--train-size",
            "4",
            "--dev-size",
            "4",
            "--test-size",
            "4",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    provenance = manifest["generation"]["rule_provenance"]
    assert manifest["source_repository_url"] == "https://github.com/IBM/AMLSim.git"
    assert manifest["source_revision"] == "7338a4bcb1af9bcfea2201ad7daccfe2a4d569ca"
    assert manifest["generation"]["uses_original_amlsim_simulator"] is False
    assert provenance["experiment_defined_parameters"]["minimum_distinct_counterparties"] == 4
    assert "not claimed as AMLSim defaults" in provenance["experiment_defined_parameters"]["statement"]
