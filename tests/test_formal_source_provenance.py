from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from industrial_anomaly_lab.formal_source_provenance import (
    capture_formal_source_provenance,
    validate_formal_source_provenance,
)
from industrial_anomaly_lab.run_artifacts import write_json


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        check=True,
        text=True,
    )
    return result.stdout.strip()


def _repository(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "project"
    root.mkdir(parents=True)
    _git(root, "init")
    _git(root, "config", "user.name", "Synthetic Test")
    _git(root, "config", "user.email", "synthetic@example.invalid")
    configs = root / "configs"
    configs.mkdir()
    authority = configs / "authority.json"
    authority.write_text('{"frozen": true}\n', encoding="utf-8")
    scope = configs / "formal_source_provenance.json"
    write_json(
        scope,
        {
            "schema_version": 1,
            "kind": "formal_source_scope",
            "status": "required_clean_at_formal_launch",
            "repositories": [
                {"name": "project", "path": ".", "expected_commit": None}
            ],
            "authority_files": [
                "configs/formal_source_provenance.json",
                "configs/authority.json",
            ],
        },
    )
    _git(root, "add", ".")
    _git(root, "commit", "-m", "freeze synthetic source")
    return root, scope


def test_capture_and_revalidate_clean_repository(tmp_path: Path) -> None:
    root, scope = _repository(tmp_path)
    evidence = capture_formal_source_provenance(project_root=root, scope_path=scope)

    assert evidence["all_registered_repositories_clean"] is True
    assert evidence["project_head_binds_all_tracked_source_files"] is True
    assert evidence["repositories"][0]["head_commit"] == _git(root, "rev-parse", "HEAD")
    assert validate_formal_source_provenance(evidence, project_root=root) == evidence


@pytest.mark.parametrize("kind", ["tracked", "untracked", "staged"])
def test_dirty_project_is_rejected(tmp_path: Path, kind: str) -> None:
    root, scope = _repository(tmp_path)
    if kind == "tracked":
        (root / "configs" / "authority.json").write_text("{}\n", encoding="utf-8")
    elif kind == "untracked":
        (root / "new_source.py").write_text("print('new')\n", encoding="utf-8")
    else:
        (root / "configs" / "authority.json").write_text("{}\n", encoding="utf-8")
        _git(root, "add", "configs/authority.json")
    with pytest.raises(ValueError, match="repository is dirty"):
        capture_formal_source_provenance(project_root=root, scope_path=scope)


def test_evidence_rejected_after_clean_new_commit(tmp_path: Path) -> None:
    root, scope = _repository(tmp_path)
    evidence = capture_formal_source_provenance(project_root=root, scope_path=scope)
    (root / "configs" / "authority.json").write_text('{"frozen": "v2"}\n', encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "change authority")

    with pytest.raises(ValueError, match="differs from the current clean source state"):
        validate_formal_source_provenance(evidence, project_root=root)


def test_scope_rejects_path_escape_and_unpinned_external_repo(tmp_path: Path) -> None:
    root, scope = _repository(tmp_path)
    payload = json.loads(scope.read_text(encoding="utf-8"))
    payload["authority_files"].append("../outside.json")
    write_json(scope, payload)
    _git(root, "add", ".")
    _git(root, "commit", "-m", "bad scope")
    with pytest.raises(ValueError, match="safe relative path"):
        capture_formal_source_provenance(project_root=root, scope_path=scope)

    root, scope = _repository(tmp_path / "other")
    payload = json.loads(scope.read_text(encoding="utf-8"))
    payload["repositories"].append(
        {"name": "vendor", "path": "third_party/vendor", "expected_commit": None}
    )
    write_json(scope, payload)
    _git(root, "add", ".")
    _git(root, "commit", "-m", "bad vendor pin")
    with pytest.raises(ValueError, match="expected_commit is invalid"):
        capture_formal_source_provenance(project_root=root, scope_path=scope)
