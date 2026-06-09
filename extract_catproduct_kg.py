from __future__ import annotations

import argparse
import json
import math
from typing import Any, Dict, List, Optional

from kg_config import KGConfig
from kg_utils import (
    clean_com_value,
    ensure_allowed_input,
    json_dumps_safe,
    merge_node,
    merge_relationship,
    neo4j_session,
    run_write,
    setup_logging,
    stable_id,
    write_json,
)


CONFIG = KGConfig()
LOGGER = setup_logging("extract_catproduct_kg")
CATA_SYSTEM_SERVICE = None


GEOMETRY_KEYWORDS = (
    "datum", "基准", "定位", "腹板", "边缘", "孔", "hole", "axis", "轴",
    "plane", "平面", "point", "点", "surface", "面", "cylinder", "圆柱",
    "publication", "reference",
)
LOW_FEATURE_KEYWORDS = ("parameter", "sketch", "草图", "构造", "路径", "相交", "活动")
STANDARD_KEYWORDS = ("铆钉", "螺栓", "螺钉", "螺母", "垫圈", "托板螺母", "紧固件", "rivet", "bolt", "screw", "nut", "washer", "fastener", "NAS", "MS", "BAC", "ABS", "HL", "CFBL")


def com_count(collection: Any) -> int:
    try:
        return int(collection.Count)
    except Exception:
        return 0


def com_item(collection: Any, index: int) -> Any:
    return collection.Item(index)


def get_attr(obj: Any, name: str, default: Any = "") -> Any:
    try:
        return getattr(obj, name)
    except Exception:
        return default


def get_source_document(product: Any) -> str:
    for attr in ("ReferenceProduct",):
        try:
            ref = getattr(product, attr)
            parent = getattr(ref, "Parent", None)
            full_name = clean_com_value(getattr(parent, "FullName", ""))
            if full_name:
                return full_name
        except Exception:
            pass
    parent = get_attr(product, "Parent", None)
    return clean_com_value(get_attr(parent, "FullName", ""))


def get_material(product: Any) -> Dict[str, str]:
    candidates = []
    try:
        candidates.append(product.UserRefProperties.Item("Material").ValueAsString())
    except Exception:
        pass
    try:
        candidates.append(product.ReferenceProduct.UserRefProperties.Item("Material").ValueAsString())
    except Exception:
        pass
    for candidate in candidates:
        material = clean_com_value(candidate)
        if material:
            return {"material": material, "material_source": "CATProduct_COM"}
    return {"material": "", "material_source": ""}


def read_user_ref_properties(product: Any) -> Dict[str, str]:
    props: Dict[str, str] = {}
    for source in (product, get_attr(product, "ReferenceProduct", None)):
        if source is None:
            continue
        try:
            collection = source.UserRefProperties
            for idx in range(1, com_count(collection) + 1):
                item = com_item(collection, idx)
                name = clean_com_value(get_attr(item, "Name", f"property_{idx}"))
                value = ""
                try:
                    value = clean_com_value(item.ValueAsString())
                except Exception:
                    value = clean_com_value(get_attr(item, "Value", ""))
                if name and value:
                    props[name] = value
        except Exception:
            continue
    return props


def read_parameters(product: Any) -> Dict[str, str]:
    params: Dict[str, str] = {}
    part = None
    try:
        part = product.ReferenceProduct.Parent.Part
    except Exception:
        return params
    try:
        collection = part.Parameters
        for idx in range(1, min(com_count(collection), 5000) + 1):
            item = com_item(collection, idx)
            name = clean_com_value(get_attr(item, "Name", f"parameter_{idx}"))
            if not name:
                continue
            text = name.lower()
            if not any(keyword.lower() in text for keyword in GEOMETRY_KEYWORDS):
                continue
            value = ""
            try:
                value = clean_com_value(item.ValueAsString())
            except Exception:
                value = clean_com_value(get_attr(item, "Value", ""))
            params[name] = value
    except Exception:
        return params
    return params


def extract_features(product: Any, owner_id: str, owner_part_number: str) -> List[Dict[str, Any]]:
    features: List[Dict[str, Any]] = []
    seen = set()

    def quality_for_feature(name: str, feature_type: str, source: str) -> str:
        text = f"{name} {feature_type} {source}".lower()
        if any(keyword.lower() in text for keyword in LOW_FEATURE_KEYWORDS):
            return "low"
        if "publication" in text or "datum" in text or "annotation" in text:
            return "high"
        if any(keyword.lower() in text for keyword in ("hybridshape", "hybridbody", "geometrical", "point", "axis", "plane", "surface", "hole", "cylinder", "基准", "点", "轴", "平面", "面", "孔")):
            return "medium"
        return "low"

    def add_feature(name: str, feature_type: str, source: str) -> None:
        clean_name = clean_com_value(name)
        if not clean_name:
            return
        lower = clean_name.lower()
        typed = f"{clean_name} {feature_type}".lower()
        if not any(keyword.lower() in typed for keyword in GEOMETRY_KEYWORDS):
            return
        quality = quality_for_feature(clean_name, feature_type, source)
        if quality == "low":
            return
        feature_id = stable_id("Feature", owner_part_number, clean_name, feature_type)
        if feature_id in seen:
            return
        seen.add(feature_id)
        features.append({
            "id": feature_id,
            "name": clean_name,
            "type": feature_type,
            "feature_quality": quality,
            "use_for_process_constraint": True,
            "source": source,
            "owner_part_number": owner_part_number,
            "owner_object_id": owner_id,
            "is_semantic_feature": False,
        })

    try:
        publications = product.Publications
        for idx in range(1, com_count(publications) + 1):
            add_feature(get_attr(com_item(publications, idx), "Name", ""), "Publication", "CATProduct_COM_publication")
    except Exception:
        pass

    try:
        part = product.ReferenceProduct.Parent.Part
        bodies = part.HybridBodies
        for body_idx in range(1, min(com_count(bodies), 200) + 1):
            body = com_item(bodies, body_idx)
            add_feature(get_attr(body, "Name", ""), "Geometrical Set", "CATPart_COM_geometrical_set")
            shapes = get_attr(body, "HybridShapes", None)
            for shape_idx in range(1, min(com_count(shapes), 2000) + 1):
                shape = com_item(shapes, shape_idx)
                add_feature(get_attr(shape, "Name", ""), clean_com_value(get_attr(shape, "Type", "HybridShape")), "CATPart_COM_hybrid_shape")
    except Exception:
        pass

    return features


def is_standard_part(name: str, part_number: str) -> bool:
    text = f"{name} {part_number}".lower()
    return any(keyword.lower() in text for keyword in STANDARD_KEYWORDS)


def try_get_aabb(product: Any) -> Dict[str, Any]:
    for target in (product, get_attr(product, "ReferenceProduct", None), get_attr(get_attr(product, "ReferenceProduct", None), "Parent", None)):
        if target is None:
            continue
        try:
            box = [0.0] * 6
            target.GetBoundingBox(box)
            return {"min": box[:3], "max": box[3:], "source": "CATIA_COM_GetBoundingBox"}
        except Exception:
            pass
        try:
            analyze = target.Analyze
            box = [0.0] * 6
            analyze.GetBoundingBox(box)
            return {"min": box[:3], "max": box[3:], "source": "CATIA_COM_Analyze_GetBoundingBox"}
        except Exception:
            pass
    return {}


def get_system_service() -> Any:
    global CATA_SYSTEM_SERVICE
    if CATA_SYSTEM_SERVICE is not None:
        return CATA_SYSTEM_SERVICE
    try:
        from pycatia import catia
        CATA_SYSTEM_SERVICE = catia().system_service
        return CATA_SYSTEM_SERVICE
    except Exception:
        return None


def evaluate_catvba(function_name: str, body: str, args: List[Any]) -> Any:
    service = get_system_service()
    if service is None:
        return None
    try:
        return service.evaluate(body, 0, function_name, args)
    except Exception:
        return None


def normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm <= 1e-12:
        return []
    return [v / norm for v in vec]


def try_get_obb(product: Any) -> Dict[str, Any]:
    try:
        inertia = product.GetTechnologicalObject("Inertia")
    except Exception:
        return {}
    cog_code = """
    Public Function get_cog_position(inertia)
        Dim oCoordinates (2)
        inertia.GetCOGPosition oCoordinates
        get_cog_position = oCoordinates
    End Function
    """
    axes_code = """
    Public Function get_principal_axes(inertia)
        Dim oComponents (8)
        inertia.GetPrincipalAxes oComponents
        get_principal_axes = oComponents
    End Function
    """
    moments_code = """
    Public Function get_principal_moments(inertia)
        Dim oValues (2)
        inertia.GetPrincipalMoments oValues
        get_principal_moments = oValues
    End Function
    """
    cog = evaluate_catvba("get_cog_position", cog_code, [inertia])
    axes_raw = evaluate_catvba("get_principal_axes", axes_code, [inertia])
    moments = evaluate_catvba("get_principal_moments", moments_code, [inertia])
    try:
        mass = float(inertia.Mass)
    except Exception:
        mass = 0.0
    if not cog or not axes_raw or not moments or mass <= 1e-12:
        return {}
    cog_vals = [float(v) for v in cog]
    axes_vals = [float(v) for v in axes_raw]
    moment_vals = [max(0.0, float(v)) for v in moments]
    axes = [
        normalize([axes_vals[0], axes_vals[3], axes_vals[6]]),
        normalize([axes_vals[1], axes_vals[4], axes_vals[7]]),
        normalize([axes_vals[2], axes_vals[5], axes_vals[8]]),
    ]
    if any(not axis for axis in axes):
        return {}
    i1, i2, i3 = moment_vals
    length_sq = [
        max(1e-12, 6.0 * (i2 + i3 - i1) / mass),
        max(1e-12, 6.0 * (i1 + i3 - i2) / mass),
        max(1e-12, 6.0 * (i1 + i2 - i3) / mass),
    ]
    half_extents = [0.5 * math.sqrt(v) for v in length_sq]
    if any(not math.isfinite(v) or v <= 0.0 for v in half_extents):
        return {}
    return {
        "center": cog_vals,
        "axes": axes,
        "half_extents": half_extents,
        "mass": mass,
        "principal_moments": moment_vals,
        "source": "CATIA_Inertia_principal_axes_equivalent_OBB",
        "unit": "m",
    }


def classify_reference(name: str, part_number: str) -> bool:
    text = f"{name} {part_number}".strip()
    return text.startswith("R_") or "参考" in text or "坐标系" in text or "reference" in text.lower()


def traverse_product(product: Any, parent_id: Optional[str], depth: int, index_path: str, rows: List[Dict[str, Any]], rels: List[Dict[str, str]], features: List[Dict[str, Any]]) -> str:
    children = get_attr(product, "Products", None)
    child_count = com_count(children)
    part_number = clean_com_value(get_attr(product, "PartNumber", ""))
    instance_name = clean_com_value(get_attr(product, "Name", ""))
    name = part_number or instance_name or f"product_{index_path}"
    label = "Product" if depth == 0 else ("SubAssembly" if child_count > 0 else "Part")
    node_id = stable_id(label, index_path, part_number, instance_name)
    material = get_material(product)
    user_props = read_user_ref_properties(product)
    is_reference = classify_reference(instance_name, part_number)
    standard = is_standard_part(instance_name, part_number)
    aabb = try_get_aabb(product)
    obb = try_get_obb(product)
    props = {
        "id": node_id,
        "name": name,
        "part_number": part_number,
        "display_name": name,
        "instance_name": instance_name,
        "node_type": label,
        "source_document": get_source_document(product),
        "source": "CATProduct_COM",
    }
    if material["material"]:
        props["material"] = material["material"]
    if depth == 0:
        props["is_final_product"] = True
    if is_reference:
        props["is_reference_object"] = True
    if standard:
        props["is_standard_part"] = True
    if aabb:
        props["aabb_json"] = json.dumps(aabb, ensure_ascii=False)
    if obb:
        props["obb_json"] = json.dumps(obb, ensure_ascii=False)
    rows.append({"label": label, "id": node_id, "props": props})
    if parent_id:
        rels.append({"parent_id": parent_id, "child_id": node_id})
    for feature in extract_features(product, node_id, part_number):
        features.append(feature)
    for idx in range(1, child_count + 1):
        try:
            traverse_product(com_item(children, idx), node_id, depth + 1, f"{index_path}.{idx}", rows, rels, features)
        except Exception as exc:
            LOGGER.warning("Failed to traverse child %s under %s: %s", idx, node_id, exc)
    return node_id


def extract(clear_graph: bool) -> Dict[str, Any]:
    catproduct = ensure_allowed_input(CONFIG.catproduct_path)
    report: Dict[str, Any] = {"input": str(catproduct), "warnings": [], "nodes": 0, "features": 0, "hasComponent": 0}
    try:
        import win32com.client
    except Exception as exc:
        raise RuntimeError("pywin32 is required for CATIA COM extraction.") from exc

    LOGGER.info("Opening CATProduct through CATIA COM: %s", catproduct)
    catia = win32com.client.Dispatch("CATIA.Application")
    document = catia.Documents.Open(str(catproduct))
    root_product = document.Product

    nodes: List[Dict[str, Any]] = []
    rels: List[Dict[str, str]] = []
    features: List[Dict[str, Any]] = []
    traverse_product(root_product, None, 0, "1", nodes, rels, features)

    with neo4j_session() as session:
        if clear_graph:
            LOGGER.info("Clearing Neo4j graph with MATCH (n) DETACH DELETE n")
            run_write(session, "MATCH (n) DETACH DELETE n")
        for node in nodes:
            merge_node(session, node["label"], node["id"], node["props"])
        for rel in rels:
            merge_relationship(session, rel["parent_id"], rel["child_id"], "hasComponent", {"source": "CATProduct_COM"})
        for feature in features:
            merge_node(session, "Feature", feature["id"], feature)
            merge_relationship(session, feature["id"], feature["owner_object_id"], "constrainedBy", {"type": "feature_on_object", "source": feature["source"]})

    report.update({
        "nodes": len(nodes),
        "features": len(features),
        "hasComponent": len(rels),
        "obb_available": sum(1 for row in nodes if row["props"].get("obb_json")),
        "feature_note": "Feature extraction uses CATIA Publications, HybridBodies/HybridShapes, and filtered Parameters when available.",
    })
    write_json(CONFIG.result_dir / "catproduct_extract_report.json", report)
    LOGGER.info("CATProduct extraction complete: %s", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear-graph", action="store_true", default=False)
    args = parser.parse_args()
    extract(clear_graph=args.clear_graph)


if __name__ == "__main__":
    main()
