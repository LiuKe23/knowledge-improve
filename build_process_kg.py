from __future__ import annotations

import json
import math
import os
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from itertools import combinations
from typing import Any, Dict, List, Tuple

from kg_config import (
    KGConfig,
    PARALLEL_DEF,
    PRECEDES_DEF,
    PROCESS_DEFS,
    PROCESS_FEATURE_CONSTRAINTS,
    PROCESS_TEXT,
    REQUIRES_RESOURCE,
    RESOURCE_DEFS,
)
from kg_utils import merge_node, merge_relationship, neo4j_session, run_read, run_write, setup_logging, write_json


CONFIG = KGConfig()
LOGGER = setup_logging("build_process_kg")
INTERFERENCE_CLEARANCE_THRESHOLD = 0.5
OBB_CLEARANCE_THRESHOLD_M = 0.015
OBB_STEP_SIZE_M = 0.003
CATIA_CONTACT_REPORT = CONFIG.project_dir / "1" / "CATTemp" / "tempFile.xml"

STANDARD_KEYWORDS = ("铆钉", "螺栓", "螺钉", "螺母", "垫圈", "托板螺母", "紧固件", "rivet", "bolt", "screw", "nut", "washer", "fastener", "NAS", "MS", "BAC", "ABS", "HL", "CFBL")
STRUCTURAL_KEYWORDS = (
    "辅助梁", "前缘舱肋", "肋", "角材", "角片", "缘条", "壁板", "蒙皮",
    "封面板", "夹芯板", "连接带板", "支撑件", "支架",
    "beam", "rib", "angle", "stringer", "skin", "panel", "support", "bracket",
)

PROCESS_BY_CATEGORY = {
    "辅助梁": ["PROC_定位前缘组件", "PROC_制连接孔", "PROC_安装紧固件"],
    "连接带板": ["PROC_手工定位角材", "PROC_制连接孔", "PROC_安装紧固件"],
    "前缘舱肋": ["PROC_定位前缘舱肋", "PROC_制初孔", "PROC_制连接孔", "PROC_安装紧固件"],
    "肋": ["PROC_定位前缘舱肋", "PROC_制初孔", "PROC_制连接孔", "PROC_安装紧固件"],
    "角材": ["PROC_手工定位角材", "PROC_制连接孔", "PROC_安装紧固件"],
    "缘条": ["PROC_手工定位角材", "PROC_制连接孔", "PROC_安装紧固件"],
    "壁板": ["PROC_定位两侧壁板", "PROC_制初孔", "PROC_制连接孔", "PROC_安装紧固件"],
    "蒙皮": ["PROC_定位两侧壁板", "PROC_制初孔", "PROC_制连接孔", "PROC_安装紧固件"],
    "支撑件/支架": ["PROC_制连接孔", "PROC_安装部分支架"],
    "支撑件": ["PROC_制连接孔", "PROC_安装部分支架"],
    "支架": ["PROC_制连接孔", "PROC_安装部分支架"],
}
PRODUCT_PROCESSES = ["PROC_吊装移站", "PROC_下架", "PROC_补铆安装紧固件", "PROC_安装部分支架", "PROC_涂胶密封", "PROC_清洗排故"]
DEFAULT_PROCESSES = ["PROC_定位前缘组件"]

# Object-level precedence derived from the explicit station process text.
# The relationship direction follows the planning convention:
#   dependent -[:constrainedBy {type:"support"}]-> support
POSITIONING_ORDER = {
    "辅助梁": 1,
    "前缘舱组件": 1,
    "前缘舱肋": 2,
    "肋": 2,
    "角材": 2,
    "缘条": 2,
    "连接带板": 2,
    "壁板": 3,
    "蒙皮": 3,
}


def preferred_text(obj: Dict[str, Any]) -> str:
    fields = ("object_category", "display_name", "name_cn", "name", "part_number")
    return " ".join(str(obj.get(key) or "") for key in fields)


def is_standard_text(text: str) -> bool:
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in STANDARD_KEYWORDS)


def classify_category(text: str) -> str:
    lowered = text.lower()
    # Priority matters: 肋角材 should be angle.
    if "角材" in text or "角片" in text or "angle" in lowered:
        return "角材"
    if "缘条" in text or "stringer" in lowered:
        return "缘条"
    if "封面板" in text or "夹芯板" in text or "壁板" in text or "蒙皮" in text or "skin" in lowered or "panel" in lowered:
        return "壁板"
    if "连接带板" in text:
        return "连接带板"
    if "辅助梁" in text or "beam" in lowered:
        return "辅助梁"
    if "前缘舱肋" in text or "肋" in text or "rib" in lowered:
        return "前缘舱肋"
    if "支撑件" in text or "支架" in text or "support" in lowered or "bracket" in lowered:
        return "支撑件/支架"
    if is_standard_text(text):
        return "标准件/紧固件"
    return ""


def classify_processes(obj: Dict[str, Any]) -> Tuple[List[str], str, float]:
    if obj.get("label") == "Product":
        return PRODUCT_PROCESSES, "process_text_rule", 1.0
    category = obj.get("object_category") or classify_category(preferred_text(obj))
    if category in PROCESS_BY_CATEGORY:
        return PROCESS_BY_CATEGORY[category], "rule_based_initial_process_assignment", 0.85
    return DEFAULT_PROCESSES, "weak_rule_based_completion", 0.3


def parse_aabb(value: str) -> Dict[str, List[float]]:
    if not value:
        return {}
    try:
        data = json.loads(value)
        mins = [float(v) for v in data.get("min", [])]
        maxs = [float(v) for v in data.get("max", [])]
        if len(mins) == 3 and len(maxs) == 3:
            return {"min": mins, "max": maxs}
    except Exception:
        return {}
    return {}


def parse_obb(value: str) -> Dict[str, Any]:
    if not value:
        return {}
    try:
        data = json.loads(value)
        center = [float(v) for v in data.get("center", [])]
        axes = [[float(v) for v in axis] for axis in data.get("axes", [])]
        half_extents = [float(v) for v in data.get("half_extents", [])]
        if len(center) == 3 and len(axes) == 3 and all(len(axis) == 3 for axis in axes) and len(half_extents) == 3:
            return {"center": center, "axes": axes, "half_extents": half_extents, "source": data.get("source", "")}
    except Exception:
        return {}
    return {}


def dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def sub(a: List[float], b: List[float]) -> List[float]:
    return [x - y for x, y in zip(a, b)]


def cross(a: List[float], b: List[float]) -> List[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def norm(a: List[float]) -> float:
    return math.sqrt(dot(a, a))


def normalize(a: List[float]) -> List[float]:
    n = norm(a)
    if n <= 1e-12:
        return []
    return [v / n for v in a]


def project_radius(obb: Dict[str, Any], axis: List[float], inflate: float = 0.0) -> float:
    extents = obb["half_extents"]
    return sum((extents[i] + inflate) * abs(dot(obb["axes"][i], axis)) for i in range(3))


def obb_gap_on_axis(a: Dict[str, Any], b: Dict[str, Any], axis: List[float], inflate: float = 0.0) -> float:
    distance = abs(dot(sub(b["center"], a["center"]), axis))
    return distance - project_radius(a, axis, inflate) - project_radius(b, axis, inflate)


def direction_from_vector(vector: List[float]) -> str:
    labels = ["X", "Y", "Z"]
    idx = max(range(3), key=lambda i: abs(vector[i]))
    return ("+" if vector[idx] >= 0 else "-") + labels[idx]


def obb_step_clearance(a: Dict[str, Any], b: Dict[str, Any], threshold: float = OBB_CLEARANCE_THRESHOLD_M, step_size: float = OBB_STEP_SIZE_M) -> Tuple[float, bool, List[str], int]:
    axes: List[List[float]] = []
    axes.extend(a["axes"])
    axes.extend(b["axes"])
    for ax in a["axes"]:
        for bx in b["axes"]:
            c = normalize(cross(ax, bx))
            if c:
                axes.append(c)

    max_gap = 0.0
    nearest_axis: List[float] = []
    overlap = True
    for axis in axes:
        gap = obb_gap_on_axis(a, b, axis)
        if gap > 0:
            overlap = False
            if gap > max_gap:
                max_gap = gap
                nearest_axis = axis
    if overlap:
        return 0.0, True, ["X", "Y", "Z"], 0

    steps = int(math.ceil(threshold / step_size))
    detected = False
    for step in range(1, steps + 1):
        inflate = step * step_size
        if all(obb_gap_on_axis(a, b, axis, inflate) <= 0 for axis in axes):
            detected = True
            steps = step
            break
    center_vec = sub(b["center"], a["center"])
    if nearest_axis and dot(center_vec, nearest_axis) < 0:
        nearest_axis = [-v for v in nearest_axis]
    direction = direction_from_vector(nearest_axis or center_vec)
    return float(max_gap), detected, [direction], steps


def aabb_clearance(a: Dict[str, List[float]], b: Dict[str, List[float]]) -> Tuple[float, bool, List[str]]:
    gaps = []
    directions = []
    overlap = True
    for axis, name in enumerate(("X", "Y", "Z")):
        if a["max"][axis] < b["min"][axis]:
            gap = b["min"][axis] - a["max"][axis]
            gaps.append(gap)
            directions.append(f"+{name}")
            overlap = False
        elif b["max"][axis] < a["min"][axis]:
            gap = a["min"][axis] - b["max"][axis]
            gaps.append(gap)
            directions.append(f"-{name}")
            overlap = False
        else:
            gaps.append(0.0)
    if overlap:
        return 0.0, True, ["X", "Y", "Z"]
    clearance = sum(gap * gap for gap in gaps) ** 0.5
    return clearance, False, directions


def prepare_planning_objects(session: Any) -> Dict[str, Any]:
    run_write(session, "MATCH (f:Feature) WHERE f.feature_quality = 'low' DETACH DELETE f")
    run_write(session, "MATCH ()-[r:connect]-() DELETE r")
    run_write(session, "MATCH ()-[r:mayInterfere]-() DELETE r")
    run_write(session, "MATCH ()-[r:constrainedBy]->() WHERE r.type = 'support' DELETE r")
    run_write(session, """
    MATCH ()-[r:requireProcess]->()
    WHERE r.source IN ['weak_rule_based_completion', 'rule_based_completion', 'rule_based_initial_process_assignment']
    DELETE r
    """)
    run_write(session, """
    MATCH (n)
    WHERE n:Product OR n:SubAssembly OR n:Part
    REMOVE n.is_assembly_object, n.is_final_product, n.is_reference_object, n.is_standard_part
    """)
    objects = run_read(session, """
    MATCH (n)
    WHERE n:Product OR n:SubAssembly OR n:Part
    RETURN n.id AS id, labels(n)[0] AS label, n.name AS name, n.part_number AS part_number,
           n.display_name AS display_name, n.name_cn AS name_cn, n.object_category AS object_category,
           n.pdf_evidence AS pdf_evidence
    """)
    category_counts: Dict[str, int] = {}
    objects_without_category = 0
    for obj in objects:
        text = preferred_text(obj)
        category = obj.get("object_category") or classify_category(text)
        if obj["label"] == "Product":
            run_write(session, """
            MATCH (n {id: $id})
            SET n.is_final_product = true,
                n.object_category = CASE WHEN coalesce(n.object_category, '') <> '' THEN n.object_category ELSE '前缘舱组件' END
            """, id=obj["id"])
            category = "前缘舱组件"
        else:
            is_reference = str(obj.get("part_number") or "").startswith("R_") or str(obj.get("name") or "").startswith("R_") or "参考" in text or "坐标系" in text
            is_standard = is_standard_text(text) or category == "标准件/紧固件"
            has_pdf_match = bool(obj.get("pdf_evidence"))
            is_planning_object = has_pdf_match and not is_reference and not is_standard
            if not category:
                objects_without_category += 1
            run_write(session, """
            MATCH (n {id: $id})
            SET n.object_category = CASE WHEN $category <> '' THEN $category ELSE coalesce(n.object_category, '') END
            FOREACH (_ IN CASE WHEN $is_reference THEN [1] ELSE [] END | SET n.is_reference_object = true)
            FOREACH (_ IN CASE WHEN $is_standard THEN [1] ELSE [] END | SET n.is_standard_part = true)
            FOREACH (_ IN CASE WHEN $is_planning_object THEN [1] ELSE [] END | SET n.is_assembly_object = true)
            """, id=obj["id"], category=category, is_reference=is_reference, is_standard=is_standard, is_planning_object=is_planning_object)
        if category:
            category_counts[category] = category_counts.get(category, 0) + 1
    counts = run_read(session, """
    MATCH (n)
    RETURN
      sum(CASE WHEN n.is_assembly_object = true THEN 1 ELSE 0 END) AS planning_object_count,
      sum(CASE WHEN n.is_standard_part = true THEN 1 ELSE 0 END) AS standard_part_count,
      sum(CASE WHEN n.is_reference_object = true THEN 1 ELSE 0 END) AS excluded_reference_count,
      sum(CASE WHEN n.is_standard_part = true THEN 1 ELSE 0 END) AS excluded_standard_part_count
    """)[0]
    return {
        "planning_object_count": counts["planning_object_count"] or 0,
        "standard_part_count": counts["standard_part_count"] or 0,
        "excluded_reference_count": counts["excluded_reference_count"] or 0,
        "excluded_standard_part_count": counts["excluded_standard_part_count"] or 0,
        "object_category_counts": category_counts,
        "objects_without_category": objects_without_category,
    }


def process_order_rank(category: str) -> int:
    return POSITIONING_ORDER.get(category, 0)


def station_group(obj: Dict[str, Any]) -> int:
    text = preferred_text(obj)
    match = re.search(r"(\d+)号", text)
    if match:
        return int(match.group(1))
    part_number = str(obj.get("part_number") or obj.get("name") or "")
    match = re.search(r"5536C10(\d)", part_number)
    if match:
        return int(match.group(1))
    return 0


def station_subtype_rank(obj: Dict[str, Any]) -> int:
    text = preferred_text(obj)
    category = obj.get("object_category") or classify_category(text)
    if category == "前缘舱肋" or ("肋" in text and "角" not in text and "缘条" not in text):
        return 1
    if category == "角材" or "角片" in text:
        return 2
    if category == "缘条":
        return 3
    return 9


def create_reference_station_support_constraints(session: Any) -> Dict[str, Any]:
    rows = run_read(session, """
    MATCH (n)
    WHERE (n:Part OR n:SubAssembly) AND n.is_assembly_object = true
    RETURN n.id AS id, n.name AS name, n.part_number AS part_number,
           n.display_name AS display_name, n.name_cn AS name_cn,
           n.object_category AS object_category
    """)
    station_objects = []
    for row in rows:
        group = station_group(row)
        subtype = station_subtype_rank(row)
        if 1 <= group <= 6 and subtype < 9:
            station_objects.append({**row, "station_group": group, "station_subtype": subtype})
    created = 0
    examples = []
    for dep in station_objects:
        for support in station_objects:
            if dep["id"] == support["id"]:
                continue
            same_group_later_subtype = dep["station_group"] == support["station_group"] and dep["station_subtype"] > support["station_subtype"]
            if not same_group_later_subtype:
                continue
            merge_relationship(session, dep["id"], support["id"], "constrainedBy", {
                "type": "support",
                "source": "user_reference_process_order",
                "evidence": "same-station rib-before-angle/stringer order supplied by user as process reference",
                "confidence": 0.9,
            })
            created += 1
            if len(examples) < 20:
                examples.append({
                    "dependent": dep["id"],
                    "dependent_group": dep["station_group"],
                    "support": support["id"],
                    "support_group": support["station_group"],
                })
    return {
        "support_constraints_from_user_reference_order": created,
        "support_user_reference_examples": examples,
    }


def create_process_text_support_constraints(session: Any) -> Dict[str, Any]:
    rows = run_read(session, """
    MATCH (n)
    WHERE (n:Part OR n:SubAssembly)
      AND n.is_assembly_object = true
      AND coalesce(n.is_reference_object, false) <> true
      AND coalesce(n.is_standard_part, false) <> true
    RETURN n.id AS id, n.name AS name, n.object_category AS object_category
    ORDER BY n.id
    """)
    objects = []
    for row in rows:
        category = row.get("object_category") or ""
        rank = process_order_rank(category)
        if rank:
            objects.append({**row, "rank": rank})

    created = 0
    examples: List[Dict[str, str]] = []
    for dep in objects:
        for support in objects:
            if dep["id"] == support["id"]:
                continue
            if support["rank"] < dep["rank"]:
                merge_relationship(session, dep["id"], support["id"], "constrainedBy", {
                    "type": "support",
                    "source": "process_text_object_precedence",
                    "confidence": 0.75,
                    "evidence": "工艺文本明确定位顺序：前缘组件/辅助梁 -> 前缘舱肋 -> 角材/缘条 -> 壁板；仅用于硬约束，不来自规划结果。",
                })
                created += 1
                if len(examples) < 20:
                    examples.append({
                        "dependent": dep["id"],
                        "dependent_category": dep.get("object_category") or "",
                        "support": support["id"],
                        "support_category": support.get("object_category") or "",
                    })

    categories = sorted({obj.get("object_category") or "" for obj in objects})
    return {
        "support_constraints_from_process_text": created,
        "support_constraint_categories": categories,
        "support_constraint_examples": examples,
    }


def create_shared_standard_connect(session: Any) -> int:
    rows = run_read(session, """
    MATCH (n)
    WHERE n:Part OR n:SubAssembly
    RETURN n.id AS id, n.part_number AS part_number, n.aabb_json AS aabb_json,
           n.is_assembly_object AS is_assembly_object, n.is_standard_part AS is_standard_part
    """)
    structures = []
    standards = []
    for row in rows:
        box = parse_aabb(row.get("aabb_json") or "")
        if not box:
            continue
        if row.get("is_assembly_object") is True:
            structures.append({**row, "box": box})
        elif row.get("is_standard_part") is True:
            standards.append({**row, "box": box})
    pair_connectors: Dict[Tuple[str, str], List[str]] = {}
    for standard in standards:
        near = []
        for structure in structures:
            clearance, overlap, _directions = aabb_clearance(standard["box"], structure["box"])
            if overlap or clearance <= INTERFERENCE_CLEARANCE_THRESHOLD:
                near.append(structure["id"])
        for a_id, b_id in combinations(sorted(set(near)), 2):
            pair_connectors.setdefault((a_id, b_id), []).append(standard["id"])
    for (a_id, b_id), connector_ids in pair_connectors.items():
        merge_relationship(session, a_id, b_id, "connect", {
            "source": "shared_standard_part",
            "connect_type": "shared_fastener",
            "confidence": 0.8,
            "connector_part_ids": connector_ids,
            "evidence": "shared standard part",
        }, undirected=True)
    return len(pair_connectors)


def normalize_catia_product_code(value: str) -> str:
    if not value:
        return ""
    text = str(value).strip()
    text = os.path.splitext(os.path.basename(text))[0]
    text = text.split()[0]
    text = text.split("(")[0]
    text = re.sub(r"\.\d+$", "", text)
    return text.strip()


def catia_product_candidates(product: Any) -> List[str]:
    candidates: List[str] = []
    description = product.get("DescriptionID") or ""
    if description:
        candidates.extend(normalize_catia_product_code(part) for part in description.split("--++--") if part.strip())
    alias = product.get("Alias") or ""
    if alias:
        candidates.append(normalize_catia_product_code(alias))
    shape_source = product.get("ShapeSource") or ""
    if shape_source:
        candidates.append(normalize_catia_product_code(shape_source))
    unique: List[str] = []
    for candidate in candidates:
        if candidate and candidate not in unique:
            unique.append(candidate)
    return unique


def create_catia_contact_connect(session: Any) -> Dict[str, Any]:
    if not CATIA_CONTACT_REPORT.exists():
        return {
            "connect_created_by_catia_contact_report": 0,
            "connect_contact_report_found": False,
            "connect_contact_report": str(CATIA_CONTACT_REPORT),
        }
    rows = run_read(session, """
    MATCH (n)
    WHERE (n:Part OR n:SubAssembly) AND n.is_assembly_object = true
    RETURN n.id AS id, n.part_number AS part_number
    """)
    by_part_number = {row["part_number"]: row for row in rows if row.get("part_number")}
    root = ET.parse(CATIA_CONTACT_REPORT).getroot()
    pair_records: Dict[Tuple[str, str], Dict[str, Any]] = defaultdict(lambda: {"interference_nums": [], "result_types": set()})
    stats = {
        "connect_contact_report_found": True,
        "connect_contact_xml_records": 0,
        "connect_contact_records": 0,
        "connect_contact_matched_two_objects": 0,
        "connect_contact_self_mapped_records": 0,
        "connect_contact_one_side_or_zero_records": 0,
    }
    for interference in root.iter("Interference"):
        stats["connect_contact_xml_records"] += 1
        if interference.get("ResultTypeNoNLS") != "Contact":
            continue
        stats["connect_contact_records"] += 1
        products = interference.findall("Product")
        mapped = []
        for product in products[:2]:
            hit = None
            for candidate in catia_product_candidates(product):
                if candidate in by_part_number:
                    hit = by_part_number[candidate]
                    break
            mapped.append(hit)
        if len(mapped) < 2 or not mapped[0] or not mapped[1]:
            stats["connect_contact_one_side_or_zero_records"] += 1
            continue
        stats["connect_contact_matched_two_objects"] += 1
        if mapped[0]["id"] == mapped[1]["id"]:
            stats["connect_contact_self_mapped_records"] += 1
            continue
        a_id, b_id = sorted([mapped[0]["id"], mapped[1]["id"]])
        record = pair_records[(a_id, b_id)]
        record["interference_nums"].append(str(interference.get("NumInterf")))
        record["result_types"].add(interference.get("ResultTypeNoNLS"))
    for (a_id, b_id), record in pair_records.items():
        merge_relationship(session, a_id, b_id, "connect", {
            "source": "CATIA_clash_contact_report",
            "connect_type": "geometric_contact",
            "evidence_file": str(CATIA_CONTACT_REPORT),
            "confidence": 0.75,
            "record_count": len(record["interference_nums"]),
            "interference_nums": record["interference_nums"],
            "result_types": sorted(record["result_types"]),
            "created_by": "connect_from_catia_contact_report",
        }, undirected=True)
    stats["connect_created_by_catia_contact_report"] = len(pair_records)
    stats["connect_contact_report"] = str(CATIA_CONTACT_REPORT)
    return stats


def create_obb_step_may_interfere(session: Any) -> int:
    rows = run_read(session, """
    MATCH (n)
    WHERE (n:Part OR n:SubAssembly) AND n.is_assembly_object = true
    RETURN n.id AS id, n.obb_json AS obb_json
    """)
    objects = []
    for row in rows:
        box = parse_obb(row.get("obb_json") or "")
        if box:
            objects.append({**row, "box": box})
    created = 0
    for a, b in combinations(objects, 2):
        clearance, detected, directions, steps = obb_step_clearance(a["box"], b["box"])
        if detected:
            merge_relationship(session, a["id"], b["id"], "mayInterfere", {
                "source": "OBB_step_detection",
                "directions": directions,
                "clearance": float(round(clearance, 8)),
                "overlap": clearance <= 0.0,
                "confidence": 0.72,
                "detection_method": "OBB_SAT_with_step_expansion",
                "step_size_m": OBB_STEP_SIZE_M,
                "clearance_threshold_m": OBB_CLEARANCE_THRESHOLD_M,
                "steps": steps,
            }, undirected=True)
            created += 1
    return created


def create_aabb_may_interfere(session: Any) -> int:
    rows = run_read(session, """
    MATCH (n)
    WHERE (n:Part OR n:SubAssembly) AND n.is_assembly_object = true
    RETURN n.id AS id, n.aabb_json AS aabb_json
    """)
    objects = []
    for row in rows:
        box = parse_aabb(row.get("aabb_json") or "")
        if box:
            objects.append({**row, "box": box})
    created = 0
    for a, b in combinations(objects, 2):
        clearance, overlap, directions = aabb_clearance(a["box"], b["box"])
        if overlap or clearance <= INTERFERENCE_CLEARANCE_THRESHOLD:
            merge_relationship(session, a["id"], b["id"], "mayInterfere", {
                "source": "AABB_fallback_analysis",
                "directions": directions,
                "clearance": float(round(clearance, 6)),
                "overlap": overlap,
                "confidence": 0.6,
            }, undirected=True)
            created += 1
    return created


def collect_planning_evidence_gaps(session: Any) -> Dict[str, Any]:
    aabb_stats = run_read(session, """
    MATCH (n)
    WHERE (n:Part OR n:SubAssembly)
      AND n.is_assembly_object = true
      AND coalesce(n.is_reference_object, false) <> true
      AND coalesce(n.is_standard_part, false) <> true
    RETURN count(n) AS total,
           sum(CASE WHEN n.aabb_json IS NULL OR n.aabb_json = '' THEN 0 ELSE 1 END) AS with_aabb
    """)[0]
    obb_stats = run_read(session, """
    MATCH (n)
    WHERE (n:Part OR n:SubAssembly)
      AND n.is_assembly_object = true
      AND coalesce(n.is_reference_object, false) <> true
      AND coalesce(n.is_standard_part, false) <> true
    RETURN count(n) AS total,
           sum(CASE WHEN n.obb_json IS NULL OR n.obb_json = '' THEN 0 ELSE 1 END) AS with_obb
    """)[0]
    support_count = run_read(session, """
    MATCH (a)-[r:constrainedBy]->(b)
    WHERE r.type = 'support'
      AND (a:Part OR a:SubAssembly)
      AND (b:Part OR b:SubAssembly)
    RETURN count(r) AS count
    """)[0]["count"]
    return {
        "planning_object_aabb_total": aabb_stats.get("total") or 0,
        "planning_object_aabb_available": aabb_stats.get("with_aabb") or 0,
        "planning_object_obb_total": obb_stats.get("total") or 0,
        "planning_object_obb_available": obb_stats.get("with_obb") or 0,
        "support_constraint_count": support_count or 0,
        "connect_evidence_note": "connect is not synthesized from planning results; it is created only from approved CATIA Contact report evidence and, when available, shared standard-part evidence.",
        "mayInterfere_evidence_note": "mayInterfere is generated from CATIA Inertia OBB + step detection when OBB evidence exists; AABB is only a fallback.",
    }


def collect_quality_metrics(session: Any) -> Dict[str, Any]:
    feature_quality = run_read(session, """
    MATCH (f:Feature)
    RETURN coalesce(f.feature_quality, 'unknown') AS quality, count(f) AS count
    ORDER BY quality
    """)
    used = run_read(session, "MATCH (f:Feature) WHERE f.use_for_process_constraint = true RETURN count(f) AS count")
    rel_counts = run_read(session, """
    MATCH ()-[r]->()
    WHERE type(r) IN ['connect', 'mayInterfere', 'requireProcess']
    RETURN type(r) AS rel_type, r.source AS source, count(r) AS count
    """)
    metrics = {
        "feature_quality_counts": {row["quality"]: row["count"] for row in feature_quality},
        "feature_used_for_process_constraint_count": used[0]["count"] if used else 0,
        "connect_count": 0,
        "connect_created_by_shared_standard_part": 0,
        "mayInterfere_count": 0,
        "mayInterfere_created_by_obb": 0,
        "weak_requireProcess_count": 0,
    }
    for row in rel_counts:
        if row["rel_type"] == "connect":
            metrics["connect_count"] += row["count"]
            if row["source"] == "shared_standard_part":
                metrics["connect_created_by_shared_standard_part"] += row["count"]
        elif row["rel_type"] == "mayInterfere":
            metrics["mayInterfere_count"] += row["count"]
            if row["source"] in ("OBB_interference_analysis", "AABB_fallback_analysis", "OBB_step_detection"):
                metrics["mayInterfere_created_by_obb"] += row["count"]
        elif row["rel_type"] == "requireProcess" and row["source"] == "weak_rule_based_completion":
            metrics["weak_requireProcess_count"] += row["count"]
    return metrics


def main() -> None:
    report: Dict[str, Any] = {
        "process_text": PROCESS_TEXT,
        "process_nodes": 0,
        "resource_nodes": 0,
        "precedes_def": 0,
        "parallel_def": 0,
        "requiresResource": 0,
        "process_feature_constraints": 0,
        "requireProcess": 0,
        "warnings": [],
        "obb_fallback_to_aabb": True,
    }
    with neo4j_session() as session:
        LOGGER.info("Creating Process and Resource template layer")
        for proc in PROCESS_DEFS:
            merge_node(session, "Process", proc["id"], {"id": proc["id"], "name": proc["name"], "process_type": proc["process_type"], "source": "process_text_extracted"})
            report["process_nodes"] += 1
        for res in RESOURCE_DEFS:
            merge_node(session, "Resource", res["id"], {"id": res["id"], "name": res["name"], "resource_type": res["resource_type"], "source": "process_text_extracted"})
            report["resource_nodes"] += 1
        for start, end in PRECEDES_DEF:
            merge_relationship(session, start, end, "precedes_def", {"source": "process_text_extracted"})
            report["precedes_def"] += 1
        for start, end in PARALLEL_DEF:
            merge_relationship(session, start, end, "parallel_def", {"source": "process_text_extracted"}, undirected=True)
            report["parallel_def"] += 1
        for proc_id, res_id in REQUIRES_RESOURCE:
            merge_relationship(session, proc_id, res_id, "requiresResource", {"source": "process_text_extracted"})
            report["requiresResource"] += 1

        LOGGER.info("Creating semantic process Feature constraints")
        for proc_id, feature_id, feature_name in PROCESS_FEATURE_CONSTRAINTS:
            merge_node(session, "Feature", feature_id, {
                "id": feature_id,
                "name": feature_name,
                "type": "semantic_feature",
                "feature_quality": "semantic",
                "use_for_process_constraint": True,
                "source": "process_text_feature",
                "owner_part_number": "",
                "owner_object_id": "",
                "is_semantic_feature": True,
            })
            merge_relationship(session, proc_id, feature_id, "constrainedBy", {
                "type": "process_feature_constraint",
                "source": "process_text_feature",
                "confidence": 0.85,
            })
            report["process_feature_constraints"] += 1

        report.update(prepare_planning_objects(session))
        LOGGER.info("Rebuilding requireProcess from object_category/display_name/name_cn")
        objects = run_read(session, """
        MATCH (n)
        WHERE n:Product OR n:SubAssembly OR n:Part
        RETURN n.id AS id, labels(n)[0] AS label, n.name AS name, n.part_number AS part_number,
               n.display_name AS display_name, n.name_cn AS name_cn, n.object_category AS object_category,
               n.is_assembly_object AS is_assembly_object, n.is_standard_part AS is_standard_part,
               n.is_reference_object AS is_reference_object
        """)
        for obj in objects:
            if obj.get("label") != "Product" and (obj.get("is_standard_part") is True or obj.get("is_reference_object") is True):
                continue
            if obj.get("label") != "Product" and obj.get("is_assembly_object") is not True:
                continue
            proc_ids, source, confidence = classify_processes(obj)
            for proc_id in proc_ids:
                merge_relationship(session, obj["id"], proc_id, "requireProcess", {"source": source, "confidence": confidence})
                report["requireProcess"] += 1

        report.update(create_process_text_support_constraints(session))
        report.update(create_reference_station_support_constraints(session))
        report["support_user_reference_note"] = "Only same-station rib-before-angle/stringer precedence is used as hard support; full station order is used as KG-IGA guidance, not as hard precedence."
        report["connect_created_by_shared_standard_part"] = create_shared_standard_connect(session)
        report.update(create_catia_contact_connect(session))
        report["mayInterfere_created_by_obb"] = create_obb_step_may_interfere(session)
        if report["mayInterfere_created_by_obb"] == 0:
            report["mayInterfere_created_by_obb"] = create_aabb_may_interfere(session)
        report.update(collect_planning_evidence_gaps(session))
        report.update(collect_quality_metrics(session))
    write_json(CONFIG.result_dir / "process_build_report.json", report)
    LOGGER.info("Process KG build complete: %s", report)


if __name__ == "__main__":
    main()
