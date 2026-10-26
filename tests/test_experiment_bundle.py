import json
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from tools.experiment_bundle import export_bundle, import_bundle


def test_export_excludes_checkpoints_and_imports_dashboard_files(tmp_path):
    runs = tmp_path / "runs"
    run = runs / "exp" / "attempt"
    (run / "rollouts").mkdir(parents=True)
    (run / "checkpoints").mkdir(parents=True)
    (run / "train.log").write_text("Training completed\n", encoding="utf-8")
    (run / "run_manifest.json").write_text("{}\n", encoding="utf-8")
    (run / "rollouts" / "step_1.jsonl").write_text('{"reward": 4}\n', encoding="utf-8")
    (run / "checkpoints" / "model.safetensors").write_bytes(b"weights")
    archive = tmp_path / "bundle.tar.gz"

    manifest = export_bundle(runs, archive, None, True)
    assert len(manifest["runs"]) == 1
    with tarfile.open(archive, "r:gz") as handle:
        names = handle.getnames()
    assert not any("checkpoints" in name for name in names)

    restored = tmp_path / "restored"
    imported = import_bundle(archive, restored)
    assert len(imported["runs"]) == 1
    restored_run = restored / "exp" / "attempt"
    assert (restored_run / "train.log").read_text(encoding="utf-8") == "Training completed\n"
    assert not (restored_run / "checkpoints").exists()


def test_manifest_is_json_and_contains_checksums(tmp_path):
    run = tmp_path / "runs" / "exp"
    run.mkdir(parents=True)
    (run / "train.log").write_text("log", encoding="utf-8")
    archive = tmp_path / "bundle.tar.gz"
    export_bundle(tmp_path / "runs", archive, None, True)
    with tarfile.open(archive, "r:gz") as handle:
        manifest = json.load(handle.extractfile("rl_experiment_bundle/manifest.json"))
    assert manifest["schema_version"] == "rl-experiment-bundle-v1"
    assert manifest["runs"][0]["files"][0]["sha256"]
