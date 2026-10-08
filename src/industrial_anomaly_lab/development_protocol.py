"""Deterministic train/development manifests without touching formal test data.

The protocol inventories only explicitly allowed normal-training and DTD roots.
Exact duplicate files are grouped by SHA-256 before assignment, so identical
source images cannot appear on both sides of a split.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence

import imgaug.augmenters as iaa
import numpy as np


IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})
AUGMENTER_IDS = (
    "gamma_contrast",
    "brightness_multiply_add",
    "sharpness",
    "hue_saturation",
    "solarize",
    "posterize",
    "invert",
    "autocontrast",
    "equalize",
    "affine_rotate",
)
PERLIN_SCALES = (1, 2, 4, 8, 16, 32)
SYNTHETIC_RENDERER_ID = "draem_synthetic_dev_renderer_v2"
SYNTHETIC_RECIPE_SCHEMA_VERSION = 3
SYNTHETIC_CONFIG_KEYS = frozenset(
    {
        "renderer_id",
        "recipe_schema_version",
        "normal_examples_per_dev_image",
        "positive_variants_per_dev_image",
        "target_rotation_probability",
        "recipe_salt",
        "image_size",
        "perlin_scales",
        "perlin_threshold",
        "blend_beta_range",
        "normal_false_positive_threshold",
        "augmenters_per_positive",
        "render_policy",
    }
)
RENDERER_PROVENANCE_FILES = (
    "pyproject.toml",
    "uv.lock",
    "src/industrial_anomaly_lab/development_protocol.py",
    "src/industrial_anomaly_lab/synthetic_development.py",
    "src/industrial_anomaly_lab/draem_synthetic_dev.py",
    "src/industrial_anomaly_lab/training_inputs.py",
    "src/industrial_anomaly_lab/draem_runtime.py",
    "src/industrial_anomaly_lab/draem_checkpointing.py",
    "src/industrial_anomaly_lab/draem_infer.py",
    "src/industrial_anomaly_lab/draem_train.py",
    "src/industrial_anomaly_lab/metrics.py",
    "src/industrial_anomaly_lab/run_artifacts.py",
    "scripts/train_draem.py",
    "scripts/evaluate_draem_synthetic_dev.py",
    "third_party/DRAEM/data_loader.py",
    "third_party/DRAEM/perlin.py",
    "third_party/DRAEM/model_unet.py",
)
RENDERER_DISTRIBUTIONS = (
    "numpy",
    "imgaug",
    "opencv-python",
    "Pillow",
    "torch",
    "scipy",
    "scikit-learn",
)
SOURCE_SPLIT_KEYS = frozenset(
    {"dataset", "category", "relative_path", "size_bytes", "sha256", "split", "split_key"}
)
SYNTHETIC_RECIPE_KEYS = frozenset(
    {
        "recipe_schema_version",
        "protocol_id",
        "renderer_id",
        "sample_id",
        "category",
        "target_relative_path",
        "target_sha256",
        "target_rotation_applied",
        "target_rotation_degrees",
        "dtd_relative_path",
        "dtd_sha256",
        "variant",
        "recipe_seed",
        "image_size",
        "augmenter_ids",
        "augmenter_seeds",
        "perlin_seed",
        "perlin_seed_attempt",
        "perlin_scale_x",
        "perlin_scale_y",
        "perlin_rotation_degrees",
        "perlin_threshold",
        "blend_beta",
        "label",
        "rendered",
    }
)


@dataclass(frozen=True)
class SourceRecord:
    dataset: str
    category: str
    relative_path: str
    size_bytes: int
    sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "category": self.category,
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_file(path)


def collect_renderer_provenance() -> dict[str, object]:
    """Hash the renderer/evaluator implementation and its locked runtime."""

    package_project_root = Path(__file__).resolve().parents[2]
    files: list[dict[str, object]] = []
    for relative in RENDERER_PROVENANCE_FILES:
        path = package_project_root / Path(relative)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"renderer provenance file is missing or linked: {path}")
        files.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for distribution in RENDERER_DISTRIBUTIONS:
        versions[distribution] = importlib.metadata.version(distribution)
    return {
        "schema_version": 1,
        "renderer_id": SYNTHETIC_RENDERER_ID,
        "recipe_schema_version": SYNTHETIC_RECIPE_SCHEMA_VERSION,
        "files": files,
        "versions": versions,
    }


def _stable_digest(*parts: object) -> str:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def derive_renderer_seed(recipe_seed: int, component: str, *parts: object) -> int:
    """Derive one explicit unsigned 32-bit renderer sub-seed."""

    if not 0 <= int(recipe_seed) <= 0xFFFFFFFF:
        raise ValueError("recipe_seed must fit in an unsigned 32-bit integer")
    digest = _stable_digest(SYNTHETIC_RENDERER_ID, int(recipe_seed), component, *parts)
    return int(digest[:8], 16)


def _perlin_noise(
    shape: tuple[int, int], res: tuple[int, int], *, seed: int
) -> np.ndarray:
    """DRAEM's NumPy Perlin formula with a recipe-local legacy RNG."""

    if any(value <= 0 for value in (*shape, *res)):
        raise ValueError("Perlin shape and resolution must be positive")
    if shape[0] % res[0] or shape[1] % res[1]:
        raise ValueError("Perlin resolution must divide the rendered image shape")
    delta = (res[0] / shape[0], res[1] / shape[1])
    repeat = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0 : res[0] : delta[0], 0 : res[1] : delta[1]].transpose(1, 2, 0) % 1
    rng = np.random.RandomState(seed)
    angles = 2 * math.pi * rng.rand(res[0] + 1, res[1] + 1)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)

    def tile(slice_y: tuple[int, int | None], slice_x: tuple[int, int | None]) -> np.ndarray:
        return np.repeat(
            np.repeat(
                gradients[slice_y[0] : slice_y[1], slice_x[0] : slice_x[1]],
                repeat[0],
                axis=0,
            ),
            repeat[1],
            axis=1,
        )

    def dot(gradient: np.ndarray, shift: tuple[int, int]) -> np.ndarray:
        ramp = np.stack(
            (
                grid[: shape[0], : shape[1], 0] + shift[0],
                grid[: shape[0], : shape[1], 1] + shift[1],
            ),
            axis=-1,
        )
        return (ramp * gradient[: shape[0], : shape[1]]).sum(axis=-1)

    n00 = dot(tile((0, -1), (0, -1)), (0, 0))
    n10 = dot(tile((1, None), (0, -1)), (-1, 0))
    n01 = dot(tile((0, -1), (1, None)), (0, -1))
    n11 = dot(tile((1, None), (1, None)), (-1, -1))
    t = 6 * grid[: shape[0], : shape[1]] ** 5 - 15 * grid[: shape[0], : shape[1]] ** 4 + 10 * grid[: shape[0], : shape[1]] ** 3
    n0 = (n10 - n00) * t[..., 0] + n00
    n1 = (n11 - n01) * t[..., 0] + n01
    return math.sqrt(2) * ((n1 - n0) * t[..., 1] + n0)


def render_perlin_mask(
    *,
    image_size: int,
    scale_x: int,
    scale_y: int,
    rotation_degrees: float,
    threshold: float,
    seed: int,
) -> np.ndarray:
    """Render the exact binary Perlin mask used by renderer v1."""

    perlin = _perlin_noise(
        (image_size, image_size), (scale_x, scale_y), seed=seed
    )
    perlin = iaa.Affine(
        rotate=rotation_degrees,
        order=1,
        cval=0,
        mode="constant",
    )(image=perlin)
    return perlin > threshold


def validate_registered_model_seeds(config: dict[str, object]) -> list[int]:
    """Verify the pre-registered seeds are derived, rather than hand selected."""

    registration = config.get("model_seed_registration")
    if not isinstance(registration, dict):
        raise ValueError("model_seed_registration must be an object")
    derivation_string = str(registration["derivation_string"])
    digest = hashlib.sha256(derivation_string.encode("utf-8")).hexdigest()
    if digest != str(registration["sha256"]):
        raise ValueError("registered model seed SHA-256 does not match its source string")
    offsets = [int(value) for value in registration["hex_chunk_offsets"]]
    if offsets != [0, 8, 16]:
        raise ValueError("model seed chunk offsets must remain [0, 8, 16]")
    derived = [int(digest[offset : offset + 8], 16) & 0x7FFFFFFF for offset in offsets]
    configured = [int(value) for value in config["model_seeds"]]
    registered = [int(value) for value in registration["derived_seeds"]]
    if derived != configured or derived != registered:
        raise ValueError("configured model seeds do not match the registered derivation")
    if len(set(derived)) != 3:
        raise ValueError("exactly three distinct model seeds are required")
    return derived


def validate_source_split_config(section: dict[str, object], *, label: str) -> None:
    root_value = section.get("root")
    if not isinstance(root_value, str) or not root_value or "\\" in root_value:
        raise ValueError(f"{label} root must be a non-empty POSIX relative path")
    root_path = PurePosixPath(root_value)
    if (
        root_path.is_absolute()
        or ":" in root_value
        or any(part in {"", ".", ".."} for part in root_path.parts)
    ):
        raise ValueError(f"{label} root must remain inside the project")
    fraction = section.get("dev_fraction")
    if type(fraction) is not float or not 0.0 < fraction < 1.0:
        raise ValueError(f"{label} dev_fraction must be a float in (0, 1)")
    if not isinstance(section.get("split_salt"), str) or not section["split_salt"]:
        raise ValueError(f"{label} split_salt must be a non-empty string")


def validate_synthetic_development_config(section: dict[str, object]) -> None:
    """Validate every renderer-v2 config field without numeric coercion."""

    if set(section) != SYNTHETIC_CONFIG_KEYS:
        missing = sorted(SYNTHETIC_CONFIG_KEYS - set(section))
        extra = sorted(set(section) - SYNTHETIC_CONFIG_KEYS)
        raise ValueError(f"synthetic development config schema mismatch; missing={missing}, extra={extra}")
    if section["renderer_id"] != SYNTHETIC_RENDERER_ID:
        raise ValueError("synthetic renderer_id is unsupported")
    if type(section["recipe_schema_version"]) is not int or section[
        "recipe_schema_version"
    ] != SYNTHETIC_RECIPE_SCHEMA_VERSION:
        raise ValueError("synthetic recipe_schema_version is unsupported")
    if type(section["normal_examples_per_dev_image"]) is not int or section[
        "normal_examples_per_dev_image"
    ] != 1:
        raise ValueError("renderer-v2 requires exactly one normal example per dev image")
    for field in ("positive_variants_per_dev_image", "image_size", "augmenters_per_positive"):
        if type(section[field]) is not int or int(section[field]) <= 0:
            raise ValueError(f"synthetic {field} must be a positive integer")
    if int(section["augmenters_per_positive"]) > len(AUGMENTER_IDS):
        raise ValueError("augmenters_per_positive exceeds registered augmenters")
    for field in (
        "target_rotation_probability",
        "perlin_threshold",
        "normal_false_positive_threshold",
    ):
        if type(section[field]) is not float or not 0.0 <= float(section[field]) <= 1.0:
            raise ValueError(f"synthetic {field} must be a float in [0, 1]")
    if not isinstance(section["recipe_salt"], str) or not section["recipe_salt"]:
        raise ValueError("synthetic recipe_salt must be a non-empty string")
    if not isinstance(section["render_policy"], str) or not section["render_policy"]:
        raise ValueError("synthetic render_policy must be a non-empty string")
    scales = section["perlin_scales"]
    image_size = int(section["image_size"])
    if (
        not isinstance(scales, list)
        or not scales
        or any(type(scale) is not int for scale in scales)
        or len(scales) != len(set(scales))
        or any(scale <= 0 or scale >= image_size or image_size % scale for scale in scales)
    ):
        raise ValueError("perlin_scales must be unique positive proper integer divisors")
    beta_range = section["blend_beta_range"]
    if (
        not isinstance(beta_range, list)
        or len(beta_range) != 2
        or any(type(value) is not float for value in beta_range)
        or not 0.0 <= beta_range[0] <= beta_range[1] <= 1.0
    ):
        raise ValueError("blend_beta_range must contain two ordered floats in [0, 1]")


def _assert_within(path: Path, allowed_root: Path) -> None:
    try:
        path.resolve().relative_to(allowed_root.resolve())
    except ValueError as error:
        raise ValueError(f"path escapes allowed root: {path}") from error


def inventory_directory(
    *,
    dataset: str,
    category: str,
    scan_root: Path,
    relative_to: Path,
) -> list[SourceRecord]:
    """Hash supported files below one explicitly allowed source directory."""

    scan_root = scan_root.resolve()
    relative_to = relative_to.resolve()
    if not scan_root.is_dir():
        raise FileNotFoundError(f"source directory is missing: {scan_root}")
    _assert_within(scan_root, relative_to)

    records: list[SourceRecord] = []
    candidates = sorted(
        (
            path
            for path in scan_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=lambda path: path.as_posix().casefold(),
    )
    for path in candidates:
        if path.is_symlink():
            raise ValueError(f"symbolic links are not accepted in manifests: {path}")
        _assert_within(path, scan_root)
        records.append(
            SourceRecord(
                dataset=dataset,
                category=category,
                relative_path=path.resolve().relative_to(relative_to).as_posix(),
                size_bytes=path.stat().st_size,
                sha256=sha256_file(path),
            )
        )
    if len(records) < 2:
        raise ValueError(f"at least two source images are required: {scan_root}")
    return records


def split_by_content_group(
    records: Sequence[SourceRecord],
    *,
    dev_fraction: float,
    salt: str,
    namespace: str,
) -> list[dict[str, object]]:
    """Assign exact-content groups to train/dev in a stable hash order."""

    if not 0.0 < dev_fraction < 1.0:
        raise ValueError("dev_fraction must be between zero and one")
    if len(records) < 2:
        raise ValueError("at least two records are required")

    groups: dict[str, list[SourceRecord]] = defaultdict(list)
    for record in records:
        groups[record.sha256].append(record)
    ranked_groups = sorted(
        groups.items(),
        key=lambda item: (_stable_digest(salt, namespace, item[0]), item[0]),
    )
    target_dev = min(
        len(records) - 1,
        max(1, int(math.ceil(len(records) * dev_fraction))),
    )
    dev_hashes: set[str] = set()
    dev_count = 0
    for content_hash, group in ranked_groups:
        if dev_count >= target_dev:
            break
        dev_hashes.add(content_hash)
        dev_count += len(group)
    if len(dev_hashes) == len(ranked_groups):
        # This is possible only with a dominant duplicate group. Fail closed:
        # no valid content-disjoint split exists.
        raise ValueError(f"content groups cannot form non-empty train/dev split: {namespace}")

    output: list[dict[str, object]] = []
    for record in sorted(records, key=lambda item: item.relative_path.casefold()):
        payload = record.to_dict()
        payload["split"] = "dev" if record.sha256 in dev_hashes else "train"
        payload["split_key"] = _stable_digest(
            salt, namespace, record.sha256, record.relative_path
        )
        output.append(payload)
    return output


def derive_expected_source_split_rows(
    rows: Sequence[dict[str, object]],
    *,
    dataset: str,
    dev_fraction: float,
    split_salt: str,
) -> list[dict[str, object]]:
    """Re-derive canonical source rows from their immutable inventory fields."""

    categories = sorted({str(row["category"]) for row in rows})
    expected: list[dict[str, object]] = []
    for category in categories:
        records = [
            SourceRecord(
                dataset=dataset,
                category=category,
                relative_path=str(row["relative_path"]),
                size_bytes=int(row["size_bytes"]),
                sha256=str(row["sha256"]),
            )
            for row in rows
            if row["category"] == category
        ]
        expected.extend(
            split_by_content_group(
                records,
                dev_fraction=dev_fraction,
                salt=split_salt,
                namespace=f"{dataset}/{category}",
            )
        )
    expected.sort(key=lambda row: str(row["relative_path"]).casefold())
    return expected


def require_exact_canonical_rows(
    actual: Sequence[dict[str, object]],
    expected: Sequence[dict[str, object]],
    *,
    label: str,
) -> None:
    """Compare JSON rows without Python's bool/int/float equality coercions."""

    if len(actual) != len(expected):
        raise ValueError(
            f"{label} row-count mismatch: actual={len(actual)}, expected={len(expected)}"
        )
    for index, (actual_row, expected_row) in enumerate(
        zip(actual, expected, strict=True), start=1
    ):
        actual_line = json.dumps(
            actual_row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        expected_line = json.dumps(
            expected_row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        if actual_line != expected_line:
            raise ValueError(f"{label} does not exactly match deterministic derivation at row {index}")


def build_synthetic_dev_recipes(
    visa_dev_records: Sequence[dict[str, object]],
    dtd_dev_records: Sequence[dict[str, object]],
    *,
    protocol_id: str,
    recipe_salt: str,
    variants_per_image: int,
    image_size: int,
    augmenters_per_positive: int,
    perlin_scales: Sequence[int],
    perlin_threshold: float,
    blend_beta_range: tuple[float, float],
    target_rotation_probability: float,
    recipe_schema_version: int,
) -> list[dict[str, object]]:
    """Create deterministic recipes while reading no dataset image content."""

    if variants_per_image <= 0 or augmenters_per_positive <= 0:
        raise ValueError("recipe counts must be positive")
    if augmenters_per_positive > len(AUGMENTER_IDS):
        raise ValueError("augmenters_per_positive exceeds registered augmenters")
    scales = tuple(int(scale) for scale in perlin_scales)
    if not scales or any(
        scale <= 0 or scale >= image_size or image_size % scale for scale in scales
    ):
        raise ValueError("Perlin scales must be positive proper divisors of image_size")
    beta_low, beta_high = (float(blend_beta_range[0]), float(blend_beta_range[1]))
    if image_size <= 0 or not 0.0 <= beta_low <= beta_high <= 1.0:
        raise ValueError("image size or blend beta range is invalid")
    if not 0.0 <= target_rotation_probability <= 1.0:
        raise ValueError("target_rotation_probability must be in [0, 1]")
    if recipe_schema_version != SYNTHETIC_RECIPE_SCHEMA_VERSION:
        raise ValueError("synthetic recipe schema version is unsupported")
    targets = sorted(
        (record for record in visa_dev_records if record["split"] == "dev"),
        key=lambda item: (str(item["category"]), str(item["relative_path"])),
    )
    sources = sorted(
        (record for record in dtd_dev_records if record["split"] == "dev"),
        key=lambda item: str(item["relative_path"]),
    )
    if not targets or not sources:
        raise ValueError("development targets and DTD development sources are required")

    recipes: list[dict[str, object]] = []
    for target in targets:
        for variant in range(variants_per_image):
            recipes.append(
                derive_expected_recipe(
                    target=target,
                    variant=variant,
                    ordered_dtd_dev_records=sources,
                    protocol_id=protocol_id,
                    recipe_salt=recipe_salt,
                    image_size=image_size,
                    augmenters_per_positive=augmenters_per_positive,
                    perlin_scales=scales,
                    perlin_threshold=perlin_threshold,
                    blend_beta_range=(beta_low, beta_high),
                    target_rotation_probability=target_rotation_probability,
                    recipe_schema_version=recipe_schema_version,
                )
            )
    return recipes


def derive_expected_recipe(
    *,
    target: dict[str, object],
    variant: int,
    ordered_dtd_dev_records: Sequence[dict[str, object]],
    protocol_id: str,
    recipe_salt: str,
    image_size: int,
    augmenters_per_positive: int,
    perlin_scales: Sequence[int],
    perlin_threshold: float,
    blend_beta_range: tuple[float, float],
    target_rotation_probability: float,
    recipe_schema_version: int,
) -> dict[str, object]:
    """Derive one complete recipe; builder and consumer share this authority."""

    if not ordered_dtd_dev_records:
        raise ValueError("ordered DTD-dev sources must not be empty")
    seed_hex = _stable_digest(
        recipe_salt,
        target["category"],
        target["sha256"],
        target["relative_path"],
        variant,
    )
    recipe_seed = int(seed_hex[:8], 16)
    rng = random.Random(recipe_seed)
    source = ordered_dtd_dev_records[int(seed_hex[8:16], 16) % len(ordered_dtd_dev_records)]
    target_rotation_applied = rng.random() < target_rotation_probability
    target_rotation_degrees = (
        round(rng.uniform(-90.0, 90.0), 8) if target_rotation_applied else None
    )
    augmenter_ids = rng.sample(list(AUGMENTER_IDS), augmenters_per_positive)
    augmenter_seeds = [
        derive_renderer_seed(recipe_seed, "augmenter", position, augmenter_id)
        for position, augmenter_id in enumerate(augmenter_ids)
    ]
    scale_x = rng.choice(list(perlin_scales))
    scale_y = rng.choice(list(perlin_scales))
    perlin_rotation = round(rng.uniform(-90.0, 90.0), 8)
    beta_low, beta_high = blend_beta_range
    blend_beta = round(beta_low + rng.random() * (beta_high - beta_low), 8)
    perlin_attempt = 0
    while True:
        perlin_seed = derive_renderer_seed(recipe_seed, "perlin", perlin_attempt)
        mask = render_perlin_mask(
            image_size=image_size,
            scale_x=scale_x,
            scale_y=scale_y,
            rotation_degrees=perlin_rotation,
            threshold=perlin_threshold,
            seed=perlin_seed,
        )
        if bool(mask.any()):
            break
        perlin_attempt += 1
        if perlin_attempt > 1024:
            raise RuntimeError("could not derive a non-empty synthetic mask")
    recipe: dict[str, object] = {
        "recipe_schema_version": recipe_schema_version,
        "protocol_id": protocol_id,
        "renderer_id": SYNTHETIC_RENDERER_ID,
        "sample_id": (
            f"{target['category']}/{Path(str(target['relative_path'])).stem}"
            f"/synthetic_v{variant}"
        ),
        "category": target["category"],
        "target_relative_path": target["relative_path"],
        "target_sha256": target["sha256"],
        "target_rotation_applied": target_rotation_applied,
        "target_rotation_degrees": target_rotation_degrees,
        "dtd_relative_path": source["relative_path"],
        "dtd_sha256": source["sha256"],
        "variant": variant,
        "recipe_seed": recipe_seed,
        "image_size": image_size,
        "augmenter_ids": augmenter_ids,
        "augmenter_seeds": augmenter_seeds,
        "perlin_seed": perlin_seed,
        "perlin_seed_attempt": perlin_attempt,
        "perlin_scale_x": scale_x,
        "perlin_scale_y": scale_y,
        "perlin_rotation_degrees": perlin_rotation,
        "perlin_threshold": perlin_threshold,
        "blend_beta": blend_beta,
        "label": 1,
        "rendered": False,
    }
    if set(recipe) != SYNTHETIC_RECIPE_KEYS:
        raise AssertionError("derived synthetic recipe schema drift")
    return recipe


def _canonical_json_line(value: dict[str, object]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(rows)
    payload = "".join(f"{_canonical_json_line(row)}\n" for row in materialized)
    path.write_text(payload, encoding="utf-8", newline="\n")
    return {
        "path": path.as_posix(),
        "rows": len(materialized),
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }


def load_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def build_protocol_artifacts(
    project_root: Path,
    config_path: Path,
    output_root: Path,
) -> dict[str, object]:
    """Build source splits and recipe manifests using only allowed train roots."""

    project_root = project_root.resolve()
    config = load_json(config_path.resolve())
    protocol_id = str(config["protocol_id"])
    model_seeds = validate_registered_model_seeds(config)
    visa = config["visa"]
    dtd = config["dtd"]
    synthetic = config["synthetic_development"]
    if not isinstance(visa, dict) or not isinstance(dtd, dict) or not isinstance(synthetic, dict):
        raise ValueError("visa, dtd and synthetic_development must be objects")
    validate_source_split_config(visa, label="VisA")
    validate_source_split_config(dtd, label="DTD")
    validate_synthetic_development_config(synthetic)

    visa_root = (project_root / str(visa["root"])).resolve()
    dtd_root = (project_root / str(dtd["root"])).resolve()
    _assert_within(visa_root, project_root)
    _assert_within(dtd_root, project_root)
    output_root = output_root.resolve()
    _assert_within(output_root, project_root)

    categories = [str(item) for item in visa["categories"]]
    if not categories or len(categories) != len(set(categories)):
        raise ValueError("VisA categories must be a non-empty unique list")

    visa_rows: list[dict[str, object]] = []
    artifact_files: list[dict[str, object]] = []
    category_counts: dict[str, dict[str, int]] = {}
    for category in categories:
        allowed_train_root = visa_root / category / "train" / "good"
        records = inventory_directory(
            dataset="visa",
            category=category,
            scan_root=allowed_train_root,
            relative_to=visa_root,
        )
        rows = split_by_content_group(
            records,
            dev_fraction=float(visa["dev_fraction"]),
            salt=str(visa["split_salt"]),
            namespace=f"visa/{category}",
        )
        visa_rows.extend(rows)
        category_counts[category] = {
            "total": len(rows),
            "train": sum(row["split"] == "train" for row in rows),
            "dev": sum(row["split"] == "dev" for row in rows),
        }
        artifact_files.append(
            write_jsonl(output_root / f"visa_{category}_split.jsonl", rows)
        )

    dtd_records: list[SourceRecord] = []
    dtd_categories = sorted(
        (path for path in dtd_root.iterdir() if path.is_dir()),
        key=lambda path: path.name.casefold(),
    )
    if not dtd_categories:
        raise ValueError(f"no DTD category directories found: {dtd_root}")
    for category_root in dtd_categories:
        dtd_records.extend(
            inventory_directory(
                dataset="dtd",
                category=category_root.name,
                scan_root=category_root,
                relative_to=dtd_root,
            )
        )
    dtd_rows: list[dict[str, object]] = []
    for category in sorted({record.category for record in dtd_records}):
        category_rows = split_by_content_group(
            [record for record in dtd_records if record.category == category],
            dev_fraction=float(dtd["dev_fraction"]),
            salt=str(dtd["split_salt"]),
            namespace=f"dtd/{category}",
        )
        dtd_rows.extend(category_rows)
    dtd_rows.sort(key=lambda row: str(row["relative_path"]).casefold())
    artifact_files.append(write_jsonl(output_root / "dtd_source_split.jsonl", dtd_rows))

    recipes = build_synthetic_dev_recipes(
        visa_rows,
        dtd_rows,
        protocol_id=protocol_id,
        recipe_salt=str(synthetic["recipe_salt"]),
        variants_per_image=int(synthetic["positive_variants_per_dev_image"]),
        image_size=int(synthetic["image_size"]),
        augmenters_per_positive=int(synthetic["augmenters_per_positive"]),
        perlin_scales=[int(value) for value in synthetic["perlin_scales"]],
        perlin_threshold=float(synthetic["perlin_threshold"]),
        blend_beta_range=(
            float(synthetic["blend_beta_range"][0]),
            float(synthetic["blend_beta_range"][1]),
        ),
        target_rotation_probability=float(synthetic["target_rotation_probability"]),
        recipe_schema_version=int(synthetic["recipe_schema_version"]),
    )
    artifact_files.append(
        write_jsonl(output_root / "synthetic_dev_recipes.jsonl", recipes)
    )

    # Keep the frozen bundle portable.  Paths in the root manifest are anchored
    # to the project root rather than to the workstation where it was built.
    for artifact in artifact_files:
        artifact_path = Path(str(artifact["path"])).resolve()
        artifact["path"] = artifact_path.relative_to(project_root).as_posix()

    visa_train_hashes = {str(row["sha256"]) for row in visa_rows if row["split"] == "train"}
    visa_dev_hashes = {str(row["sha256"]) for row in visa_rows if row["split"] == "dev"}
    dtd_train_hashes = {str(row["sha256"]) for row in dtd_rows if row["split"] == "train"}
    dtd_dev_hashes = {str(row["sha256"]) for row in dtd_rows if row["split"] == "dev"}
    if visa_train_hashes & visa_dev_hashes:
        raise AssertionError("VisA content hash leakage detected")
    if dtd_train_hashes & dtd_dev_hashes:
        raise AssertionError("DTD content hash leakage detected")

    summary = {
        "schema_version": 1,
        "protocol_id": protocol_id,
        "path_base": "project_root",
        "project_root": ".",
        "config_path": config_path.resolve().relative_to(project_root).as_posix(),
        "formal_test_content_accessed": False,
        "allowed_roots_read": [
            f"{visa['root']}/<category>/train/good",
            str(dtd["root"]),
        ],
        "model_seeds": model_seeds,
        "model_seed_registration": config["model_seed_registration"],
        "visa": {
            "categories": category_counts,
            "total": len(visa_rows),
            "train": sum(row["split"] == "train" for row in visa_rows),
            "dev": sum(row["split"] == "dev" for row in visa_rows),
            "content_hash_overlap": 0,
        },
        "dtd": {
            "categories": len(dtd_categories),
            "total": len(dtd_rows),
            "train": sum(row["split"] == "train" for row in dtd_rows),
            "dev": sum(row["split"] == "dev" for row in dtd_rows),
            "content_hash_overlap": 0,
        },
        "synthetic_development": {
            "renderer_id": SYNTHETIC_RENDERER_ID,
            "recipe_schema_version": SYNTHETIC_RECIPE_SCHEMA_VERSION,
            "normal_samples": sum(row["split"] == "dev" for row in visa_rows),
            "synthetic_positive_recipes": len(recipes),
            "empty_mask_recipes": 0,
            "images_rendered": 0,
        },
        "renderer_provenance": collect_renderer_provenance(),
        "artifact_files": artifact_files,
        "training_started": False,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "manifest.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return summary
