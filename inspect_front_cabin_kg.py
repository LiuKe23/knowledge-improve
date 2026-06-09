from __future__ import annotations

from typing import Any, Dict

from kg_config import KGConfig
from kg_utils import neo4j_session, read_result_json, run_read, setup_logging, write_json


CONFIG = KGConfig()
LOGGER = setup_logging("inspect_front_cabin_kg")


def main() -> None:
    report: Dict[str, Any] = {"warnings": []}
    with neo4j_session() as session:
        labels = run_read(session, """
        MATCH (n)
        UNWIND labels(n) AS label
        RETURN label, count(n) AS count
        ORDER BY label
        """)
        rels = run_read(session, """
        MATCH ()-[r]->()
        RETURN type(r) AS relationship, count(r) AS count
        ORDER BY relationship
        """)
        report["label_counts"] = {row["label"]: row["count"] for row in labels}
        report["relationship_counts"] = {row["relationship"]: row["count"] for row in rels}
        for label in ["Product", "SubAssembly", "Part", "Feature", "Process", "Resource"]:
            report.setdefault("required_label_counts", {})[label] = report["label_counts"].get(label, 0)
        for rel in ["requireProcess", "requiresResource", "precedes_def", "parallel_def", "connect", "mayInterfere", "constrainedBy"]:
            report.setdefault("required_relationship_counts", {})[rel] = report["relationship_counts"].get(rel, 0)

        bad_labels = [label for label in report["label_counts"] if label not in CONFIG.allowed_labels]
        bad_rels = [rel for rel in report["relationship_counts"] if rel not in CONFIG.allowed_relationships]
        if bad_labels:
            report["warnings"].append(f"Disallowed labels found: {bad_labels}")
        if bad_rels:
            report["warnings"].append(f"Disallowed relationship types found: {bad_rels}")

        object_stats = run_read(session, """
        MATCH (n)
        RETURN
          sum(CASE WHEN n.is_assembly_object = true THEN 1 ELSE 0 END) AS planning_object_count,
          sum(CASE WHEN n.is_standard_part = true THEN 1 ELSE 0 END) AS standard_part_count,
          sum(CASE WHEN n.is_reference_object = true THEN 1 ELSE 0 END) AS excluded_reference_count,
          sum(CASE WHEN n.is_standard_part = true THEN 1 ELSE 0 END) AS excluded_standard_part_count,
          sum(CASE WHEN (n:Part OR n:SubAssembly) AND coalesce(n.object_category, '') = '' THEN 1 ELSE 0 END) AS objects_without_category
        """)[0]
        report.update({key: object_stats.get(key) or 0 for key in object_stats})

        categories = run_read(session, """
        MATCH (n)
        WHERE n:Product OR n:SubAssembly OR n:Part
        RETURN coalesce(n.object_category, '') AS category, count(n) AS count
        ORDER BY category
        """)
        report["object_category_counts"] = {row["category"] or "": row["count"] for row in categories}

        feature_quality = run_read(session, """
        MATCH (f:Feature)
        RETURN coalesce(f.feature_quality, 'unknown') AS quality, count(f) AS count
        ORDER BY quality
        """)
        report["feature_quality_counts"] = {row["quality"]: row["count"] for row in feature_quality}
        used = run_read(session, "MATCH (f:Feature) WHERE f.use_for_process_constraint = true RETURN count(f) AS count")
        report["feature_used_for_process_constraint_count"] = used[0]["count"] if used else 0

        weak = run_read(session, "MATCH ()-[r:requireProcess]->() WHERE r.source = 'weak_rule_based_completion' RETURN count(r) AS count")
        report["weak_requireProcess_count"] = weak[0]["count"] if weak else 0
        support = run_read(session, """
        MATCH (a)-[r:constrainedBy]->(b)
        WHERE r.type = 'support'
          AND (a:Part OR a:SubAssembly)
          AND (b:Part OR b:SubAssembly)
        RETURN count(r) AS count
        """)
        report["support_constraint_count"] = support[0]["count"] if support else 0
        support_by_source = run_read(session, """
        MATCH (a)-[r:constrainedBy]->(b)
        WHERE r.type = 'support'
          AND (a:Part OR a:SubAssembly)
          AND (b:Part OR b:SubAssembly)
        RETURN coalesce(r.source, '') AS source, count(r) AS count
        ORDER BY source
        """)
        report["support_constraint_sources"] = {row["source"]: row["count"] for row in support_by_source}
        shared = run_read(session, "MATCH ()-[r:connect]->() WHERE r.source = 'shared_standard_part' RETURN count(r) AS count")
        report["connect_created_by_shared_standard_part"] = shared[0]["count"] if shared else 0
        obb = run_read(session, "MATCH ()-[r:mayInterfere]->() WHERE r.source IN ['OBB_interference_analysis', 'AABB_fallback_analysis', 'OBB_step_detection'] RETURN count(r) AS count")
        report["mayInterfere_created_by_obb"] = obb[0]["count"] if obb else 0
        report["connect_count"] = report["relationship_counts"].get("connect", 0)
        report["mayInterfere_count"] = report["relationship_counts"].get("mayInterfere", 0)
        report["obb_fallback_to_aabb"] = True

        aabb_stats = run_read(session, """
        MATCH (n)
        WHERE (n:Part OR n:SubAssembly)
          AND n.is_assembly_object = true
          AND coalesce(n.is_reference_object, false) <> true
          AND coalesce(n.is_standard_part, false) <> true
        RETURN count(n) AS total,
               sum(CASE WHEN n.aabb_json IS NULL OR n.aabb_json = '' THEN 0 ELSE 1 END) AS with_aabb
        """)
        if aabb_stats:
            report["planning_object_aabb_total"] = aabb_stats[0]["total"] or 0
            report["planning_object_aabb_available"] = aabb_stats[0]["with_aabb"] or 0
        obb_stats = run_read(session, """
        MATCH (n)
        WHERE (n:Part OR n:SubAssembly)
          AND n.is_assembly_object = true
          AND coalesce(n.is_reference_object, false) <> true
          AND coalesce(n.is_standard_part, false) <> true
        RETURN count(n) AS total,
               sum(CASE WHEN n.obb_json IS NULL OR n.obb_json = '' THEN 0 ELSE 1 END) AS with_obb
        """)
        if obb_stats:
            report["planning_object_obb_total"] = obb_stats[0]["total"] or 0
            report["planning_object_obb_available"] = obb_stats[0]["with_obb"] or 0

        null_materials = run_read(session, """
        MATCH (n)
        WHERE (n:Product OR n:SubAssembly OR n:Part) AND (n.material IS NULL OR n.material = '')
        RETURN count(n) AS count
        """)
        report["empty_material_objects"] = null_materials[0]["count"] if null_materials else 0
        report["material_note"] = "Empty material is allowed and is not treated as an error."
        report["connect_note"] = "connect is created from approved CATIA Contact report evidence and, when available, shared standard part evidence."
        report["mayInterfere_note"] = "mayInterfere is generated from CATIA Inertia OBB + step detection; AABB is fallback only."
        report["support_note"] = "Object-level support constraints are derived only from explicit process-text precedence and hasComponent during planning; they are not inferred from optimized sequences."

    pdf_report = read_result_json("pdf_extract_report.json", {})
    for key in ("pdf_matched_to_catproduct_count", "pdf_unmatched_structural_part_count", "pdf_standard_part_count"):
        report[key] = pdf_report.get(key, pdf_report.get("neo4j", {}).get(key, 0))

    write_json(CONFIG.result_dir / "inspect_report.json", report)
    LOGGER.info("Inspection complete: %s", report)


if __name__ == "__main__":
    main()
