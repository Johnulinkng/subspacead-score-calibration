"""Deterministic Git provenance required at every formal launch.

A formal run is valid only when the project repository and every registered
vendor repository are clean.  A clean HEAD plus its tree object binds all
tracked implementation files without maintaining a fragile hand-written list.
Untracked source files also make the project dirty and therefore fail closed.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .run_artifacts import sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCOPE = PROJECT_ROOT / "configs" / "formal_source_provenance.json"


def _safe_relative_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{label} must be a non-empty POSIX relative path")
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise ValueError(f"{label} is not a safe relative path: {value!r}")
    return candidate.as_posix()


def _resolve_under(root: Path, relative: object, label: str, *, file: bool) -> Path:
    safe = _safe_relative_path(relative, label)
    candidate = root / Path(*PurePosixPath(safe).parts)
    if candidate.is_symlink():
        raise ValueError(f"{label} cannot be a symlink: {candidate}")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes project root") from error
    if file and not resolved.is_file():
        raise FileNotFoundError(resolved)
    if not file and not resolved.is_dir():
        raise FileNotFoundError(resolved)
    return resolved


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{label} is missing or linked: {path}")
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _git(repository: Path, *arguments: str) -> str:
    process = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        capture_output=True,
        check=False,
        encoding="utf-8",
        errors="replace",
    )
    if process.returncode != 0:
        detail = process.stderr.strip() or process.stdout.strip()
        raise ValueError(
            f"git {' '.join(arguments)} failed for {repository}: {detail}"
        )
    return process.stdout.strip()


def _repository_evidence(
    *, root: Path, name: str, relative: str, expected_commit: str | None
) -> dict[str, Any]:
    repository = root if relative == "." else _resolve_under(
        root, relative, f"{name} repository", file=False
    )
    top_level = Path(_git(repository, "rev-parse", "--show-toplevel")).resolve()
    if top_level != repository:
        raise ValueError(
            f"registered repository is not its own Git top level: {relative} -> {top_level}"
        )
    head = _git(repository, "rev-parse", "HEAD")
    tree = _git(repository, "rev-parse", "HEAD^{tree}")
    if len(head) != 40 or len(tree) != 40:
        raise ValueError(f"unexpected Git object identity for repository: {name}")
    if expected_commit is not None and head != expected_commit:
        raise ValueError(
            f"{name} commit mismatch: expected={expected_commit}, observed={head}"
        )
    status = _git(
        repository,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignore-submodules=none",
    )
    if status:
        first_rows = status.splitlines()[:20]
        raise ValueError(f"{name} repository is dirty: {first_rows}")
    return {
        "name": name,
        "path": relative,
        "head_commit": head,
        "head_tree": tree,
        "expected_commit": expected_commit,
        "worktree_clean": True,
        "untracked_files_checked": True,
        "submodule_changes_checked": True,
    }


def capture_formal_source_provenance(
    *,
    project_root: str | Path = PROJECT_ROOT,
    scope_path: str | Path = DEFAULT_SCOPE,
) -> dict[str, Any]:
    """Capture source identity, failing if any registered repository is dirty."""

    root = Path(project_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    scope_candidate = Path(scope_path)
    if not scope_candidate.is_absolute():
        scope_candidate = root / scope_candidate
    scope_resolved = scope_candidate.resolve()
    try:
        scope_relative = scope_resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("formal source scope must stay below project root") from error
    scope = _load_json(scope_resolved, "formal source scope")
    if scope.get("schema_version") != 1 or scope.get("kind") != "formal_source_scope":
        raise ValueError("unsupported formal source scope schema")
    if scope.get("status") != "required_clean_at_formal_launch":
        raise ValueError("formal source provenance scope is not active")

    raw_repositories = scope.get("repositories")
    if not isinstance(raw_repositories, list) or not raw_repositories:
        raise ValueError("formal source scope has no repositories")
    repositories: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    seen_paths: set[str] = set()
    for index, raw in enumerate(raw_repositories):
        if not isinstance(raw, Mapping):
            raise ValueError(f"repository entry {index} must be an object")
        if set(raw) != {"name", "path", "expected_commit"}:
            raise ValueError(f"repository entry {index} schema mismatch")
        name = raw.get("name")
        relative = raw.get("path")
        expected = raw.get("expected_commit")
        if not isinstance(name, str) or not name:
            raise ValueError(f"repository entry {index} name is invalid")
        if relative == ".":
            if name != "project" or expected is not None:
                raise ValueError("project repository must use path='.' and expected_commit=null")
            safe_relative = "."
        else:
            safe_relative = _safe_relative_path(relative, f"repository entry {index} path")
            if (
                not isinstance(expected, str)
                or len(expected) != 40
                or any(character not in "0123456789abcdef" for character in expected)
            ):
                raise ValueError(f"repository entry {index} expected_commit is invalid")
        if name in seen_names or safe_relative.casefold() in seen_paths:
            raise ValueError("formal source scope contains duplicate repository identity")
        seen_names.add(name)
        seen_paths.add(safe_relative.casefold())
        repositories.append(
            _repository_evidence(
                root=root,
                name=name,
                relative=safe_relative,
                expected_commit=expected,
            )
        )
    if repositories[0]["name"] != "project" or repositories[0]["path"] != ".":
        raise ValueError("project repository must be the first scope entry")

    raw_authority_files = scope.get("authority_files")
    if not isinstance(raw_authority_files, list) or not raw_authority_files:
        raise ValueError("formal source scope has no authority files")
    authority_files: list[dict[str, Any]] = []
    seen_files: set[str] = set()
    for index, raw in enumerate(raw_authority_files):
        relative = _safe_relative_path(raw, f"authority file {index}")
        if relative.casefold() in seen_files:
            raise ValueError("formal source scope contains duplicate authority files")
        seen_files.add(relative.casefold())
        path = _resolve_under(root, relative, f"authority file {index}", file=True)
        authority_files.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    scope_posix = PurePosixPath(*scope_relative.parts).as_posix()
    if scope_posix not in seen_files:
        raise ValueError("formal source scope must hash itself as an authority file")
    return {
        "schema_version": 1,
        "contract": "clean_git_source_provenance_v1",
        "scope": scope_posix,
        "scope_sha256": sha256_file(scope_resolved),
        "repositories": repositories,
        "authority_files": authority_files,
        "all_registered_repositories_clean": True,
        "project_head_binds_all_tracked_source_files": True,
        "untracked_source_files_forbidden": True,
    }


def validate_formal_source_provenance(
    evidence: Mapping[str, Any],
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Re-capture and require byte-for-byte-equivalent formal source identity."""

    if evidence.get("schema_version") != 1:
        raise ValueError("formal source provenance must use schema_version 1")
    if evidence.get("contract") != "clean_git_source_provenance_v1":
        raise ValueError("unexpected formal source provenance contract")
    scope = evidence.get("scope")
    current = capture_formal_source_provenance(
        project_root=project_root,
        scope_path=str(scope),
    )
    if dict(evidence) != current:
        raise ValueError("formal source provenance differs from the current clean source state")
    return current
