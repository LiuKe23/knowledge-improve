from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Sequence, Set, Tuple

from load_planning_problem import RESULT_DIR, load_problem, write_json


def edge_pairs(problem: Dict[str, Any]) -> List[Tuple[str, str]]:
    return [(e["before"], e["after"]) for e in problem.get("hard_precedence_edges", [])]


def is_valid_sequence(sequence: Sequence[str], edges: Sequence[Tuple[str, str]]) -> bool:
    if len(sequence) != len(set(sequence)):
        return False
    pos = {node: i for i, node in enumerate(sequence)}
    return all(pos.get(before, 10**9) < pos.get(after, -1) for before, after in edges)


def build_graph(nodes: Sequence[str], edges: Sequence[Tuple[str, str]]) -> Tuple[Dict[str, Set[str]], Dict[str, int]]:
    succ: Dict[str, Set[str]] = {n: set() for n in nodes}
    indeg: Dict[str, int] = {n: 0 for n in nodes}
    for before, after in edges:
        if before in succ and after in succ and after not in succ[before]:
            succ[before].add(after)
            indeg[after] += 1
    return succ, indeg


def random_topological_sort(nodes: Sequence[str], edges: Sequence[Tuple[str, str]], rng: random.Random, bias: Dict[str, float] | None = None) -> List[str]:
    succ, indeg = build_graph(nodes, edges)
    available = [n for n in nodes if indeg[n] == 0]
    result: List[str] = []
    while available:
        if bias:
            weights = [max(0.001, bias.get(n, 1.0)) for n in available]
            chosen = rng.choices(available, weights=weights, k=1)[0]
        else:
            chosen = rng.choice(available)
        available.remove(chosen)
        result.append(chosen)
        for nxt in sorted(succ[chosen]):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                available.append(nxt)
    if len(result) != len(nodes):
        raise ValueError("hard precedence edges contain a cycle; no feasible topological sequence exists")
    return result


def repair_to_feasible(order: Sequence[str], nodes: Sequence[str], edges: Sequence[Tuple[str, str]]) -> List[str]:
    priority = {node: i for i, node in enumerate(order)}
    for node in nodes:
        priority.setdefault(node, len(priority))
    succ, indeg = build_graph(nodes, edges)
    available = [n for n in nodes if indeg[n] == 0]
    result: List[str] = []
    while available:
        chosen = min(available, key=lambda n: priority.get(n, 10**9))
        available.remove(chosen)
        result.append(chosen)
        for nxt in sorted(succ[chosen]):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                available.append(nxt)
    if len(result) != len(nodes):
        raise ValueError("hard precedence edges contain a cycle; repair failed")
    return result


def pair_key(a: str, b: str) -> str:
    return "||".join(sorted((a, b)))


ALL_DIRECTIONS = {"+X", "-X", "+Y", "-Y", "+Z", "-Z"}


def normalize_blocked_directions(directions: Sequence[Any]) -> Set[str]:
    blocked: Set[str] = set()
    for direction in directions or []:
        value = str(direction).strip().upper()
        if value in ALL_DIRECTIONS:
            blocked.add(value)
    return blocked


def geometry_feasibility_details(sequence: Sequence[str], problem: Dict[str, Any]) -> Dict[str, Any]:
    maps = problem.get("_objective_maps_cache")
    if maps is None:
        maps = build_objective_maps(problem)
        problem["_objective_maps_cache"] = maps
    assembled: List[str] = []
    steps = []
    feasible = True
    first_blocked_step = None
    for index, node in enumerate(sequence, start=1):
        blocked: Set[str] = set()
        evidence_pairs = []
        for previous in assembled:
            key = pair_key(node, previous)
            dirs = maps["directions"].get(key)
            if not dirs:
                continue
            normalized = sorted(normalize_blocked_directions(dirs))
            if not normalized:
                continue
            blocked.update(normalized)
            evidence_pairs.append({
                "with_object": previous,
                "directions": list(dirs),
                "blocked_directions": normalized,
            })
        available = sorted(ALL_DIRECTIONS - blocked)
        step_feasible = bool(available)
        if not step_feasible and feasible:
            first_blocked_step = index
        feasible = feasible and step_feasible
        steps.append({
            "sequence_index": index,
            "object_id": node,
            "blocked_directions": sorted(blocked),
            "available_directions": available,
            "is_geometry_feasible": step_feasible,
            "evidence_pair_count": len(evidence_pairs),
            "evidence_pairs": evidence_pairs,
        })
        assembled.append(node)
    return {
        "is_geometry_feasible": feasible,
        "first_blocked_step": first_blocked_step,
        "steps": steps,
    }


def is_geometry_feasible_sequence(sequence: Sequence[str], problem: Dict[str, Any]) -> bool:
    return bool(geometry_feasibility_details(sequence, problem)["is_geometry_feasible"])


def euclidean_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def compute_spatial_scale(centers: Dict[str, Sequence[float]]) -> Dict[str, float]:
    values = list(centers.values())
    distances = [
        euclidean_distance(values[i], values[j])
        for i in range(len(values))
        for j in range(i + 1, len(values))
    ]
    nonzero = [d for d in distances if d > 1e-9]
    if nonzero:
        reference = median(nonzero)
        threshold = reference * 1.75
    else:
        reference = 1.0
        threshold = 1.75
    ranges = []
    for dim in range(3):
        coords = [point[dim] for point in values]
        ranges.append((max(coords) - min(coords)) if coords else 0.0)
    flow_axis = max(range(3), key=lambda dim: ranges[dim]) if values else 0
    projections = [point[flow_axis] for point in values]
    projection_min = min(projections) if projections else 0.0
    projection_max = max(projections) if projections else 1.0
    projection_reference = max(projection_max - projection_min, reference, 1e-9)
    return {
        "reference_distance": max(reference, 1e-9),
        "spatial_jump_threshold": max(threshold, 1e-9),
        "flow_axis": flow_axis,
        "projection_min": projection_min,
        "projection_max": projection_max,
        "projection_reference": projection_reference,
    }


def build_objective_maps(problem: Dict[str, Any]) -> Dict[str, Any]:
    connect = set()
    if not problem.get("_disable_connect_objective"):
        connect = {pair_key(e["a"], e["b"]) for e in problem.get("connect_edges", [])}
    directions = {pair_key(e["a"], e["b"]): tuple(e.get("directions") or []) for e in problem.get("may_interfere_edges", []) if e.get("directions")}
    resources = {oid: set(vals) for oid, vals in problem.get("resources_by_object", {}).items()}
    centers = {
        oid: tuple(center)
        for oid, center in problem.get("object_centers", {}).items()
        if isinstance(center, list) and len(center) == 3
    }
    spatial_scale = compute_spatial_scale(centers)
    categories = {obj["id"]: str(obj.get("category") or "") for obj in problem.get("objects", [])}
    flow_object_ids = {
        oid for oid in centers
        if "壁板" not in categories.get(oid, "")
        and "辅助梁" not in categories.get(oid, "")
    }
    flow_projections = [centers[oid][spatial_scale["flow_axis"]] for oid in flow_object_ids]
    if flow_projections:
        spatial_scale["projection_min"] = min(flow_projections)
        spatial_scale["projection_max"] = max(flow_projections)
        spatial_scale["projection_reference"] = max(spatial_scale["projection_max"] - spatial_scale["projection_min"], spatial_scale["reference_distance"], 1e-9)
    return {
        "connect": connect,
        "directions": directions,
        "resources": resources,
        "centers": centers,
        "projections": {oid: center[spatial_scale["flow_axis"]] for oid, center in centers.items()},
        "flow_object_ids": flow_object_ids,
        **spatial_scale,
    }


def evaluate_sequence(sequence: Sequence[str], problem: Dict[str, Any], weights: Dict[str, float] | None = None) -> Dict[str, Any]:
    weights = weights or {
        "direction": 1.5,
        "tool": 2.0,
        "kg": 1.0,
        "connect": 1.0,
        "spatial": 1.0,
        "flow": 30.0,
        "spatial_alpha": 1.0,
    }
    maps = problem.get("_objective_maps_cache")
    if maps is None:
        maps = build_objective_maps(problem)
        problem["_objective_maps_cache"] = maps
    direction_switches = tool_switches = 0
    connect_break_penalty = spatial_jump_penalty = spatial_flow_penalty = 0.0
    direction_maintains = tool_maintains = continuity_maintains = 0
    spatial_pairs_scored = spatial_pairs_missing = 0
    previous_dirs = None
    for pair_index, (a, b) in enumerate(zip(sequence, sequence[1:])):
        key = pair_key(a, b)
        dirs = maps["directions"].get(key)
        if dirs:
            if previous_dirs is not None:
                if dirs == previous_dirs:
                    direction_maintains += 1
                else:
                    direction_switches += 1
            previous_dirs = dirs
        resources_a = maps["resources"].get(a, set())
        resources_b = maps["resources"].get(b, set())
        if resources_a == resources_b:
            tool_maintains += 1
        else:
            tool_switches += max(1, len(resources_a ^ resources_b))
        if not problem.get("_disable_connect_objective") and key in maps["connect"]:
            continuity_maintains += 1
        elif not problem.get("_disable_connect_objective"):
            connect_break_penalty += 1.0
        center_a = maps["centers"].get(a)
        center_b = maps["centers"].get(b)
        if center_a and center_b:
            distance = euclidean_distance(center_a, center_b)
            d_norm = distance / maps["reference_distance"]
            if distance <= maps["spatial_jump_threshold"]:
                spatial_jump_penalty += d_norm
            else:
                spatial_jump_penalty += d_norm + weights["spatial_alpha"] * (d_norm ** 2)
            spatial_pairs_scored += 1
            proj_a = maps["projections"].get(a)
            proj_b = maps["projections"].get(b)
            if proj_a is not None and proj_b is not None:
                reverse = proj_a - proj_b
                tolerance = 0.03 * maps["projection_reference"]
                if pair_index > 0 and a in maps["flow_object_ids"] and b in maps["flow_object_ids"] and reverse > tolerance:
                    rev_norm = reverse / maps["projection_reference"]
                    spatial_flow_penalty += 2.0 * rev_norm + rev_norm ** 2
                if pair_index == 0 and len(sequence) > 2 and b in maps["flow_object_ids"]:
                    start_offset = max(0.0, proj_b - maps["projection_min"])
                    start_norm = start_offset / maps["projection_reference"]
                    spatial_flow_penalty += start_norm + start_norm ** 2
        else:
            spatial_pairs_missing += 1
    spatial_jump_penalty += weights["flow"] * spatial_flow_penalty
    kg_continuity_penalty = (
        (0.0 if problem.get("_disable_connect_objective") else weights["connect"] * connect_break_penalty)
        + weights["spatial"] * spatial_jump_penalty
    )
    fitness = (
        weights["direction"] * direction_switches
        + weights["tool"] * tool_switches
        + weights["kg"] * kg_continuity_penalty
    )
    return {
        "fitness": float(fitness),
        "direction_switches": direction_switches,
        "tool_switches": tool_switches,
        "continuity_breaks": kg_continuity_penalty,
        "kg_continuity_penalty": kg_continuity_penalty,
        "connect_break_penalty": connect_break_penalty,
        "spatial_jump_penalty": spatial_jump_penalty,
        "spatial_flow_penalty": spatial_flow_penalty,
        "reference_distance": maps["reference_distance"],
        "spatial_jump_threshold": maps["spatial_jump_threshold"],
        "flow_axis": maps["flow_axis"],
        "spatial_pairs_scored": spatial_pairs_scored,
        "spatial_pairs_missing": spatial_pairs_missing,
        "direction_maintains": direction_maintains,
        "tool_maintains": tool_maintains,
        "continuity_maintains": continuity_maintains,
    }


def generate_candidate_pool(problem: Dict[str, Any], pool_size: int = 5000, seed: int = 42) -> Tuple[List[List[str]], Dict[str, Any]]:
    rng = random.Random(seed)
    nodes = problem["object_ids"]
    edges = edge_pairs(problem)
    candidates: List[List[str]] = []
    seen = set()
    duplicate_removed = invalid_removed = geometry_invalid_removed = attempts = 0
    max_attempts = max(pool_size * 20, pool_size)
    geometry_check = bool(problem.get("_enable_geometry_hard_check"))
    while len(candidates) < pool_size and attempts < max_attempts:
        attempts += 1
        try:
            seq = random_topological_sort(nodes, edges, rng)
        except ValueError:
            invalid_removed += 1
            break
        key = tuple(seq)
        if key in seen:
            duplicate_removed += 1
            continue
        if not is_valid_sequence(seq, edges):
            invalid_removed += 1
            continue
        if geometry_check and not is_geometry_feasible_sequence(seq, problem):
            geometry_invalid_removed += 1
            continue
        seen.add(key)
        candidates.append(seq)
    scores = [evaluate_sequence(seq, problem)["fitness"] for seq in candidates]
    maps = problem.get("_objective_maps_cache") or build_objective_maps(problem)
    report = {
        "object_count": len(nodes),
        "hard_precedence_count": len(edges),
        "candidate_pool_size": len(candidates),
        "generation_attempts": attempts,
        "duplicate_removed": duplicate_removed,
        "invalid_removed": invalid_removed,
        "geometry_feasibility_enabled": geometry_check,
        "geometry_invalid_removed": geometry_invalid_removed,
        "best_candidate_fitness": min(scores) if scores else None,
        "worst_candidate_fitness": max(scores) if scores else None,
        "reference_distance": maps.get("reference_distance"),
        "spatial_jump_threshold": maps.get("spatial_jump_threshold"),
        "flow_axis": maps.get("flow_axis"),
        "objects_with_obb_center": len(maps.get("centers", {})),
        "notes": list(problem.get("quality", {}).get("notes", [])),
    }
    if len(edges) < max(1, len(nodes) // 2):
        report["notes"].append("hard constraints sparse")
    return candidates, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", default=str(RESULT_DIR / "problem_snapshot.json"))
    parser.add_argument("--pool-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    problem = load_problem(Path(args.snapshot))
    candidates, report = generate_candidate_pool(problem, args.pool_size, args.seed)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(RESULT_DIR / "candidate_pool.json", {"sequences": candidates})
    write_json(RESULT_DIR / "candidate_pool_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
