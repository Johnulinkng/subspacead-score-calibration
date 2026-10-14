"""核对原文件，留好原稿，再替换本次整理的报告正文。"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

BASE_SHA = "bbf37c926d0354b576d39f0b4d972592233e84844a4b57ffb9cc22092f17ad1f"
EXT = "优化实验/同骨干Subspace对照-20261014"
RECORDS = "文稿整理记录/20261014"
CHANGED = {
    "项目总览.html", "先读我.txt", "实验说明.txt", "report/report.html", "report/复现说明.md",
    "优化实验/DINO小型SubspaceAD验证/report.html",
    "优化实验/DINO小型SubspaceAD验证/DINO_SubspaceAD验收报告.md",
    "优化实验/OAS验证/report.html", "优化实验/OAS验证/OAS验收报告.md",
    EXT + "/report.html", EXT + "/同骨干对照报告.txt", EXT + "/给同事的说明.txt", EXT + "/复现说明.txt",
}
ADDED_DOCS = {"code/代码说明.txt", EXT + "/技术说明.txt"}
ADDED_CODE = {
    "scripts/verify_readable_delivery.py", "scripts/prepare_readable_reports.py",
    "scripts/apply_readable_revision.py", "tests/test_readable_delivery_metadata.py",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(ok: bool, why: str) -> None:
    if not ok:
        raise ValueError(why)


def copy_new(source: Path, target: Path) -> None:
    require(not target.exists(), "已有存档或新增文件，不能覆盖：" + str(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as f:
        f.write(source.read_bytes())
    require(sha(source) == sha(target), "复制校验失败")


def write_new(path: Path, obj: dict) -> None:
    with path.open("x", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    a = parser.parse_args()
    project, package, stage = (x.resolve(strict=True) for x in (a.project_root, a.package_root, a.stage_root))
    require(package.is_relative_to(project / "artifacts/deliveries") and not stage.is_relative_to(package), "目录范围不正确")
    master_path = package / "DELIVERY-MANIFEST.json"
    require(sha(master_path) == BASE_SHA, "交付底稿清单不是已核对的版本")
    base = json.loads(master_path.read_text(encoding="utf-8-sig"))
    index = {x["path"]: x for x in base["files"]}
    require(len(index) == len(base["files"]) == 42854, "底稿文件数不符")
    actual = {f.relative_to(package).as_posix() for f in package.rglob("*") if f.is_file()}
    require(actual == set(index) | {"DELIVERY-MANIFEST.json"}, "原交付出现未登记文件或缺失文件")
    for relative, row in index.items():
        f = package / relative
        require(not f.is_symlink() and f.resolve().is_relative_to(package) and f.stat().st_size == row["size_bytes"] and sha(f) == row["sha256"], "原交付文件有变化：" + relative)
    drafts = {f.relative_to(stage).as_posix() for f in stage.rglob("*") if f.is_file()}
    require(drafts == CHANGED | ADDED_DOCS, "草稿必须严格匹配本次正文范围")
    records = package / RECORDS
    require(not records.exists(), "整理记录已存在，不能重复执行")
    plan = []
    for relative in sorted(CHANGED):
        old, new = package / relative, stage / relative
        require(relative in index and sha(old) != sha(new), "正文没有变化或不在原清单：" + relative)
        original, revised = old.read_text(encoding="utf-8-sig"), new.read_text(encoding="utf-8-sig")
        require(bool(revised.strip()) and "\0" not in revised, "草稿为空或编码不正确")
        if relative.endswith(".html"):
            old_tables = Counter(re.findall(r"<table\b.*?</table>", original, re.S))
            new_tables = Counter(re.findall(r"<table\b.*?</table>", revised, re.S))
            require(all(new_tables[t] >= n for t, n in old_tables.items()), "原表格内容变化：" + relative)
            require(Counter(re.findall(r"<figure\b.*?</figure>", original, re.S)) == Counter(re.findall(r"<figure\b.*?</figure>", revised, re.S)), "固定对比图内容变化：" + relative)
        if relative.endswith(".md"):
            require(Counter(re.findall(r"```.*?```", original, re.S)) == Counter(re.findall(r"```.*?```", revised, re.S)), "原复现命令变化：" + relative)
        plan.append({"path": relative, "before_sha256": sha(old), "after_sha256": sha(new),
                     "original_copy": RECORDS + "/原文/" + relative,
                     "before_characters": len(original), "after_characters": len(revised)})
    for relative in ADDED_DOCS:
        require(not (package / relative).exists(), "新增说明已经存在")
    for relative in ADDED_CODE:
        require((project / relative).is_file() and not (package / "code" / relative).exists(), "新增整理工具路径冲突")
    records.mkdir(parents=True)
    copy_new(master_path, records / "整理前DELIVERY-MANIFEST.json")
    copy_new(package / EXT / "extension_manifest.json", records / "整理前extension_manifest.json")
    for row in plan:
        copy_new(package / row["path"], package / row["original_copy"])
    for row in plan:
        target = package / row["path"]
        require(sha(target) == row["before_sha256"] and sha(stage / row["path"]) == row["after_sha256"], "文稿复制期间出现外部变化")
        tmp = target.with_name(target.name + ".editorial-tmp")
        copy_new(stage / row["path"], tmp)
        tmp.replace(target)
    added = []
    for relative in sorted(ADDED_DOCS):
        copy_new(stage / relative, package / relative)
        added.append({"path": relative, "sha256": sha(package / relative)})
    for relative in sorted(ADDED_CODE):
        target = package / "code" / relative
        copy_new(project / relative, target)
        added.append({"path": target.relative_to(package).as_posix(), "sha256": sha(target)})
    receipt = {"schema_version": 1, "status": "completed", "edited_at_utc": datetime.now(timezone.utc).isoformat(),
               "base_master_sha256": BASE_SHA, "base_extension_manifest_sha256": sha(records / "整理前extension_manifest.json"),
               "scope": "作者报告正文整理；新增阅读说明及文稿版本核验入口", "files": plan, "added_files": added,
               "algorithm_sources_modified": False, "models_weights_maps_masks_metrics_modified": False,
               "third_party_licenses_and_provenance_preserved": True, "historical_receipts_and_snapshots_preserved": True,
               "new_method_or_originality_claim_added": False, "new_computation_performed": False,
               "numeric_html_tables_and_fixed_figures_preserved": True,
               "checked_base_members": len(index), "utility_sha256": sha(Path(__file__).resolve()),
               "next_step": "重新封存扩展与总清单，然后用新增入口核对结果"}
    write_new(records / "文稿整理回执.json", receipt)
    print(json.dumps({"status": "completed", "reports_revised": len(plan), "added_files": len(added), "base_members_verified": len(index)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
