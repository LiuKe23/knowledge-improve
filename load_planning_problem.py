from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from kg_config import Neo4jConfig


RESULT_DIR = Path(r"F:\proV1.8\front_cabin_planning_results")


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_obb_center(value: str | None) -> List[float] | None:
    if not value:
        return None
    try:
        data = json.loads(value)
        center = [float(v) for v in data.get("center", [])]
    except Exception:
        return None
    return center if len(center) == 3 else None


def node_key(row: Dict[str, Any], prefix: str = "") -> str:
    return row.get(f"{prefix}id") or row.get(f"{prefix}name") or ""


def fetch_problem(uri: str, user: str, password: str) -> Dict[str, Any]:
    from neo4j import GraphDatabase

    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        driver.verify_connectivity()
        with driver.session() as session:
            objects = [dict(r) for r in session.run("""
            MATCH (o)
            WHERE (o:Part OR o:SubAssembly)
              AND coalesce(o.is_assembly_object,false) = true
              AND coalesce(o.is_reference_object,false) <> true
              AND coalesce(o.is_standard_part,false) <> true
            RETURN o.id AS id, labels(o)[0] AS label, o.name AS name,
                   o.display_name AS display_name, o.part_number AS part_number,
                   o.object_category AS category, o.obb_json AS obb_json
            ORDER BY o.id
            """)]
            for row in objects:
                row["obb_center"] = parse_obb_center(row.pop("obb_json", None))
            object_ids = [row["id"] for row in objects]

            process_rows = [dict(r) for r in session.run("""
            MATCH (o)-[:requireProcess]->(p:Process)
            WHERE o.id IN $object_ids
            OPTIONAL MATCH (p)-[:requiresResource]->(res:Resource)
            RETURN o.id AS object_id, p.id AS process_id, p.name AS process_name,
                   p.process_type AS process_type, collect(DISTINCT res.id) AS resource_ids,
                   collect(DISTINCT res.name) AS resource_names
            ORDER BY o.id, p.id
            """, object_ids=object_ids)]

            product_rows = [dict(r) for r in session.run("""
            MATCH (prod:Product)-[:requireProcess]->(p:Process)
            WHERE coalesce(prod.is_final_product,false) = true
            OPTIONAL MATCH (p)-[:requiresResource]->(res:Resource)
            RETURN prod.id AS product_id, prod.name AS product_name,
                   p.id AS process_id, p.name AS process_name,
                   collect(DISTINCT res.id) AS resource_ids,
                   collect(DISTINCT res.name) AS resource_names
            ORDER BY p.id
            """)]

            support_rows = [dict(r) for r in session.run("""
            MATCH (dep)-[rel:constrainedBy]->(support)
            WHERE rel.type = 'support'
              AND dep.id IN $object_ids AND support.id IN $object_ids
            RETURN support.id AS before, dep.id AS after,
                   rel.source AS source, rel.evidence AS evidence
            ORDER BY before, after
            """, object_ids=object_ids)]

            hierarchy_rows = [dict(r) for r in session.run("""
            MATCH (parent)-[:hasComponent]->(child)
            WHERE parent.id IN $object_ids AND child.id IN $object_ids
            RETURN child.id AS before, parent.id AS after
            ORDER BY before, after
            """, object_ids=object_ids)]

            connect_rows = [dict(r) for r in session.run("""
            MATCH (a)-[rel:connect]-(b)
            WHERE a.id IN $object_ids AND b.id IN $object_ids AND a.id < b.id
            RETURN a.id AS a, b.id AS b, properties(rel) AS properties
            ORDER BY a, b
            """, object_ids=object_ids)]

            may_rows = [dict(r) for r in session.run("""
            MATCH (a)-[rel:mayInterfere]-(b)
            WHERE a.id IN $object_ids AND b.id IN $object_ids AND a.id < b.id
            RETURN a.id AS a, b.id AS b, rel.directions AS directions,
                   rel.clearance AS clearance, rel.overlap AS overlap,
                   properties(rel) AS properties
            ORDER BY a, b
            """, object_ids=object_ids)]

            precedence_rows = [dict(r) for r in session.run("""
            MATCH (a:Process)-[:precedes_def]->(b:Process)
            RETURN a.id AS before, a.name AS before_name, b.id AS after, b.name AS after_name
            ORDER BY before, after
            """)]
            parallel_rows = [dict(r) for r in session.run("""
            MATCH (a:Process)-[:parallel_def]-(b:Process)
            WHERE a.id < b.id
            RETURN a.id AS a, a.name AS a_name, b.id AS b, b.name AS b_name
            ORDER BY a, b
            """)]

    processes_by_object: Dict[str, List[Dict[str, Any]]] = {oid: [] for oid in object_ids}
    resources_by_object: Dict[str, List[str]] = {oid: [] for oid in object_ids}
    object_centers: Dict[str, List[float]] = {row["id"]: row["obb_center"] for row in objects if row.get("obb_center")}
    for row in process_rows:
        processes_by_object[row["object_id"]].append({
            "id": row["process_id"],
            "name": row["process_name"],
            "type": row.get("process_type") or "",
            "resource_ids": row.get("resource_ids") or [],
            "resource_names": row.get("resource_names") or [],
        })
        resources_by_object[row["object_id"]] = sorted(set(resources_by_object[row["object_id"]]) | set(row.get("resource_ids") or []))

    hard_edges = []
    seen_edges = set()
    for row in support_rows + hierarchy_rows:
        edge = (row["before"], row["after"])
        if edge[0] and edge[1] and edge[0] != edge[1] and edge not in seen_edges:
            seen_edges.add(edge)
            hard_edges.append({"before": edge[0], "after": edge[1], "source": row.get("source", "hasComponent")})

    problem = {
        "neo4j": {"uri": uri, "user": user},
        "result_dir": str(RESULT_DIR),
        "objects": objects,
        "object_ids": object_ids,
        "object_centers": object_centers,
        "processes_by_object": processes_by_object,
        "resources_by_object": resources_by_object,
        "hard_precedence_edges": hard_edges,
        "support_edges": support_rows,
        "has_component_edges": hierarchy_rows,
        "connect_edges": connect_rows,
        "may_interfere_edges": may_rows,
        "product_processes": product_rows,
        "process_precedence_defs": precedence_rows,
        "process_parallel_defs": parallel_rows,
        "quality": {
            "object_count": len(object_ids),
            "hard_precedence_count": len(hard_edges),
            "connect_count": len(connect_rows),
            "mayInterfere_count": len(may_rows),
            "objects_with_obb_center": len(object_centers),
            "objects_without_process": [oid for oid in object_ids if not processes_by_object.get(oid)],
            "notes": [],
        },
    }
    if not connect_rows:
        problem["quality"]["notes"].append("connect missing: continuity objective falls back to shared Process only.")
    if not may_rows:
        problem["quality"]["notes"].append("mayInterfere missing: direction-switch objective has no active object-pair evidence.")
    if len(hard_edges) < max(1, len(object_ids) // 2):
        problem["quality"]["notes"].append("hard constraints sparse.")
    return problem


def load_problem(snapshot_path: Path | None = None) -> Dict[str, Any]:
    path = snapshot_path or RESULT_DIR / "problem_snapshot.json"
    return read_json(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    cfg = Neo4jConfig()
    parser.add_argument("--uri", default=cfg.uri)
    parser.add_argument("--user", default=cfg.user)
    parser.add_argument("--password", default=cfg.password)
    parser.add_argument("--output", default=str(RESULT_DIR / "problem_snapshot.json"))
    args = parser.parse_args()
    problem = fetch_problem(args.uri, args.user, args.password)
    write_json(Path(args.output), problem)
    print(f"problem_snapshot written: {args.output}")
    print(json.dumps(problem["quality"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
