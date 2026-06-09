from __future__ import annotations

import argparse
import importlib
import time
from typing import Any, Dict, List

from kg_config import KGConfig
from kg_utils import read_result_json, setup_logging, write_json


CONFIG = KGConfig()
LOGGER = setup_logging("run_build_kg")


STEPS = [
    ("extract_catproduct_kg", "extract_catproduct_kg.py"),
    ("extract_pdf_kg", "extract_pdf_kg.py"),
    ("build_process_kg", "build_process_kg.py"),
    ("inspect_front_cabin_kg", "inspect_front_cabin_kg.py"),
]


def run_step(module_name: str, clear_graph: bool) -> Dict[str, Any]:
    started = time.time()
    LOGGER.info("Starting step: %s", module_name)
    module = importlib.import_module(module_name)
    if module_name == "extract_catproduct_kg":
        module.extract(clear_graph=clear_graph)
    else:
        module.main()
    elapsed = round(time.time() - started, 3)
    LOGGER.info("Finished step: %s in %.3fs", module_name, elapsed)
    return {"module": module_name, "elapsed_seconds": elapsed, "status": "ok"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear-graph", action="store_true", default=True, help="Clear graph before rebuilding. Default: true.")
    parser.add_argument("--no-clear-graph", action="store_false", dest="clear_graph")
    args = parser.parse_args()

    total: Dict[str, Any] = {"clear_graph": args.clear_graph, "steps": []}
    failures: List[Dict[str, Any]] = []
    for module_name, _script in STEPS:
        try:
            total["steps"].append(run_step(module_name, clear_graph=args.clear_graph))
        except Exception as exc:
            LOGGER.exception("Step failed: %s", module_name)
            failure = {"module": module_name, "status": "failed", "error": str(exc)}
            total["steps"].append(failure)
            failures.append(failure)
            break
        finally:
            args.clear_graph = False
    total["reports"] = {
        "catproduct_extract_report": read_result_json("catproduct_extract_report.json", {}),
        "pdf_extract_report": read_result_json("pdf_extract_report.json", {}),
        "process_build_report": read_result_json("process_build_report.json", {}),
        "inspect_report": read_result_json("inspect_report.json", {}),
    }
    metric_keys = [
        "planning_object_count", "standard_part_count", "excluded_reference_count",
        "excluded_standard_part_count", "object_category_counts", "weak_requireProcess_count",
        "objects_without_category", "feature_quality_counts", "feature_used_for_process_constraint_count",
        "connect_count", "connect_created_by_shared_standard_part", "mayInterfere_count",
        "mayInterfere_created_by_obb", "obb_fallback_to_aabb",
        "planning_object_obb_total", "planning_object_obb_available",
        "pdf_matched_to_catproduct_count", "pdf_unmatched_structural_part_count",
        "pdf_standard_part_count",
    ]
    summary = {}
    for key in metric_keys:
        for report_name in ("inspect_report", "process_build_report", "pdf_extract_report"):
            report = total["reports"].get(report_name, {})
            if key in report:
                summary[key] = report[key]
                break
    total["planning_quality_summary"] = summary
    total["status"] = "failed" if failures else "ok"
    write_json(CONFIG.result_dir / "kg_total_report.json", total)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
