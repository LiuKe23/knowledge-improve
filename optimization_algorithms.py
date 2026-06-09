from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence, Tuple

from candidate_sequence_generator import (
    build_objective_maps,
    edge_pairs,
    evaluate_sequence,
    geometry_feasibility_details,
    is_geometry_feasible_sequence,
    is_valid_sequence,
    random_topological_sort,
    repair_to_feasible,
)
from load_planning_problem import RESULT_DIR, load_problem, write_json


def sequence_score(sequence: Sequence[str], problem: Dict[str, Any]) -> float:
    return evaluate_sequence(sequence, problem)["fitness"]


def v5_scoring_problem(problem: Dict[str, Any]) -> Dict[str, Any]:
    scoped = dict(problem)
    scoped["_disable_connect_objective"] = True
    scoped["_disable_connect_guidance"] = True
    scoped.pop("_objective_maps_cache", None)
    return scoped


def score_for_algorithm(sequence: Sequence[str], problem: Dict[str, Any], algorithm: str) -> Dict[str, Any]:
    if algorithm == "KG-IGAv5":
        return evaluate_sequence(sequence, v5_scoring_problem(problem))
    return evaluate_sequence(sequence, problem)


def sequence_score_for_algorithm(sequence: Sequence[str], problem: Dict[str, Any], algorithm: str) -> float:
    return score_for_algorithm(sequence, problem, algorithm)["fitness"]


KNOWLEDGE_USAGE_FIELDS = [
    "algorithm",
    "run",
    "initial_best_fitness",
    "initial_avg_fitness",
    "total_crossover_count",
    "guided_crossover_count",
    "total_mutation_count",
    "guided_mutation_count",
    "bad_position_detect_count",
    "connect_query_count",
    "obb_query_count",
    "mayinterfere_query_count",
    "resource_query_count",
    "repair_count",
    "repair_changed_count",
    "guided_accept_count",
    "guided_reject_count",
    "prescreen_count",
    "prescreen_reject_count",
    "direct_guide_attempt_count",
    "direct_guide_accept_count",
    "direct_guide_illegal_reject_count",
    "connect_candidate_count",
    "obb_fallback_count",
    "constraint_mutation_attempt_count",
    "constraint_mutation_accept_count",
    "constraint_mutation_shift_count",
    "constraint_mutation_reject_count",
    "no_repair_crossover_retry_count",
    "no_repair_illegal_child_count",
    "no_repair_valid_child_count",
    "no_repair_local_reject_count",
    "geometry_valid_initial_count",
    "geometry_invalid_initial_count",
    "geometry_valid_child_count",
    "geometry_invalid_child_count",
    "geometry_valid_neighbor_count",
    "geometry_invalid_neighbor_count",
]


KNOWLEDGE_USAGE_SUMMARY_FIELDS = [
    "algorithm",
    "runs",
    "avg_guided_crossover_count",
    "avg_guided_mutation_count",
    "avg_knowledge_query_count",
    "avg_repair_changed_count",
    "guided_accept_rate_percent",
    "avg_prescreen_reject_count",
    "avg_initial_best_fitness",
    "avg_direct_guide_attempt_count",
    "avg_direct_guide_accept_count",
    "avg_direct_guide_illegal_reject_count",
    "avg_connect_candidate_count",
    "avg_obb_fallback_count",
    "avg_constraint_mutation_attempt_count",
    "avg_constraint_mutation_accept_count",
    "avg_constraint_mutation_shift_count",
    "avg_constraint_mutation_reject_count",
    "avg_no_repair_crossover_retry_count",
    "avg_no_repair_illegal_child_count",
    "avg_no_repair_valid_child_count",
    "avg_no_repair_local_reject_count",
    "avg_geometry_valid_initial_count",
    "avg_geometry_invalid_initial_count",
    "avg_geometry_valid_child_count",
    "avg_geometry_invalid_child_count",
    "avg_geometry_valid_neighbor_count",
    "avg_geometry_invalid_neighbor_count",
]


def empty_knowledge_usage(algorithm: str = "", run: int = 0) -> Dict[str, Any]:
    row = {field: 0 for field in KNOWLEDGE_USAGE_FIELDS}
    row["algorithm"] = algorithm
    row["run"] = run
    return row


def add_usage(target: Dict[str, Any], key: str, value: int = 1) -> None:
    target[key] = int(target.get(key, 0)) + value


def knowledge_query_total(row: Dict[str, Any]) -> int:
    return (
        int(row.get("connect_query_count", 0))
        + int(row.get("obb_query_count", 0))
        + int(row.get("mayinterfere_query_count", 0))
        + int(row.get("resource_query_count", 0))
    )


def summarize_knowledge_usage(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result = []
    algorithms = sorted({str(row.get("algorithm", "")) for row in rows if row.get("algorithm")})
    for algorithm in algorithms:
        group = [row for row in rows if row.get("algorithm") == algorithm]
        accepts = sum(int(row.get("guided_accept_count", 0)) for row in group)
        rejects = sum(int(row.get("guided_reject_count", 0)) for row in group)
        attempts = accepts + rejects
        result.append({
            "algorithm": algorithm,
            "runs": len(group),
            "avg_guided_crossover_count": round(mean(int(row.get("guided_crossover_count", 0)) for row in group), 3),
            "avg_guided_mutation_count": round(mean(int(row.get("guided_mutation_count", 0)) for row in group), 3),
            "avg_knowledge_query_count": round(mean(knowledge_query_total(row) for row in group), 3),
            "avg_repair_changed_count": round(mean(int(row.get("repair_changed_count", 0)) for row in group), 3),
            "guided_accept_rate_percent": round(100.0 * accepts / attempts, 2) if attempts else 0.0,
            "avg_prescreen_reject_count": round(mean(int(row.get("prescreen_reject_count", 0)) for row in group), 3),
            "avg_initial_best_fitness": round(mean(float(row.get("initial_best_fitness", 0) or 0) for row in group), 6),
            "avg_direct_guide_attempt_count": round(mean(int(row.get("direct_guide_attempt_count", 0)) for row in group), 3),
            "avg_direct_guide_accept_count": round(mean(int(row.get("direct_guide_accept_count", 0)) for row in group), 3),
            "avg_direct_guide_illegal_reject_count": round(mean(int(row.get("direct_guide_illegal_reject_count", 0)) for row in group), 3),
            "avg_connect_candidate_count": round(mean(int(row.get("connect_candidate_count", 0)) for row in group), 3),
            "avg_obb_fallback_count": round(mean(int(row.get("obb_fallback_count", 0)) for row in group), 3),
            "avg_constraint_mutation_attempt_count": round(mean(int(row.get("constraint_mutation_attempt_count", 0)) for row in group), 3),
            "avg_constraint_mutation_accept_count": round(mean(int(row.get("constraint_mutation_accept_count", 0)) for row in group), 3),
            "avg_constraint_mutation_shift_count": round(mean(int(row.get("constraint_mutation_shift_count", 0)) for row in group), 3),
            "avg_constraint_mutation_reject_count": round(mean(int(row.get("constraint_mutation_reject_count", 0)) for row in group), 3),
            "avg_no_repair_crossover_retry_count": round(mean(int(row.get("no_repair_crossover_retry_count", 0)) for row in group), 3),
            "avg_no_repair_illegal_child_count": round(mean(int(row.get("no_repair_illegal_child_count", 0)) for row in group), 3),
            "avg_no_repair_valid_child_count": round(mean(int(row.get("no_repair_valid_child_count", 0)) for row in group), 3),
            "avg_no_repair_local_reject_count": round(mean(int(row.get("no_repair_local_reject_count", 0)) for row in group), 3),
            "avg_geometry_valid_initial_count": round(mean(int(row.get("geometry_valid_initial_count", 0)) for row in group), 3),
            "avg_geometry_invalid_initial_count": round(mean(int(row.get("geometry_invalid_initial_count", 0)) for row in group), 3),
            "avg_geometry_valid_child_count": round(mean(int(row.get("geometry_valid_child_count", 0)) for row in group), 3),
            "avg_geometry_invalid_child_count": round(mean(int(row.get("geometry_invalid_child_count", 0)) for row in group), 3),
            "avg_geometry_valid_neighbor_count": round(mean(int(row.get("geometry_valid_neighbor_count", 0)) for row in group), 3),
            "avg_geometry_invalid_neighbor_count": round(mean(int(row.get("geometry_invalid_neighbor_count", 0)) for row in group), 3),
        })
    return result


def graphplan(problem: Dict[str, Any], seed: int) -> Tuple[List[str], List[float]]:
    rng = random.Random(seed)
    nodes = problem["object_ids"]
    edges = edge_pairs(problem)
    succ = {n: set() for n in nodes}
    indeg = {n: 0 for n in nodes}
    for before, after in edges:
        succ[before].add(after)
        indeg[after] += 1
    available = [n for n in nodes if indeg[n] == 0]
    sequence: List[str] = []
    curve: List[float] = []
    while available:
        if not sequence:
            chosen = sorted(available)[0]
        else:
            last = sequence[-1]
            scored = []
            for node in available:
                trial = sequence + [node]
                local = evaluate_sequence([last, node], problem)
                scored.append((local["fitness"], rng.random(), node))
            chosen = min(scored)[2]
        available.remove(chosen)
        sequence.append(chosen)
        curve.append(sequence_score(repair_to_feasible(sequence + [n for n in nodes if n not in sequence], nodes, edges), problem))
        for nxt in sorted(succ[chosen]):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                available.append(nxt)
    return sequence, curve


def mutate_insert(seq: List[str], rng: random.Random) -> List[str]:
    new = list(seq)
    if len(new) < 2:
        return new
    i, j = rng.sample(range(len(new)), 2)
    item = new.pop(i)
    new.insert(j, item)
    return new


def precedence_sets(nodes: Sequence[str], edges: Sequence[Tuple[str, str]]) -> Tuple[Dict[str, set], Dict[str, set]]:
    predecessors = {node: set() for node in nodes}
    successors = {node: set() for node in nodes}
    for before, after in edges:
        if before in predecessors and after in successors:
            predecessors[after].add(before)
            successors[before].add(after)
    return predecessors, successors


def mutate_insert_constraint_aware(seq: List[str], nodes: Sequence[str], edges: Sequence[Tuple[str, str]], rng: random.Random, usage: Dict[str, Any] | None = None) -> List[str]:
    if usage is not None:
        add_usage(usage, "constraint_mutation_attempt_count")
    if len(seq) < 2:
        if usage is not None:
            add_usage(usage, "constraint_mutation_reject_count")
        return list(seq)
    item = rng.choice(seq)
    target = rng.randrange(len(seq))
    without = [node for node in seq if node != item]
    pos = {node: idx for idx, node in enumerate(without)}
    predecessors, successors = precedence_sets(nodes, edges)
    lower = 0
    upper = len(without)
    present_preds = [pos[pred] for pred in predecessors.get(item, set()) if pred in pos]
    present_succs = [pos[succ] for succ in successors.get(item, set()) if succ in pos]
    if present_preds:
        lower = max(present_preds) + 1
    if present_succs:
        upper = min(present_succs)
    if lower > upper:
        if usage is not None:
            add_usage(usage, "constraint_mutation_reject_count")
        return list(seq)
    shifted_target = min(max(target, lower), upper)
    if usage is not None and shifted_target != target:
        add_usage(usage, "constraint_mutation_shift_count")
    mutated = list(without)
    mutated.insert(shifted_target, item)
    if is_valid_sequence(mutated, edges):
        if usage is not None:
            add_usage(usage, "constraint_mutation_accept_count")
        return mutated
    if usage is not None:
        add_usage(usage, "constraint_mutation_reject_count")
    return list(seq)


def crossover_order(a: Sequence[str], b: Sequence[str], rng: random.Random) -> List[str]:
    n = len(a)
    i, j = sorted(rng.sample(range(n), 2))
    child = [None] * n
    child[i:j] = a[i:j]
    fill = [x for x in b if x not in child]
    k = 0
    for idx in range(n):
        if child[idx] is None:
            child[idx] = fill[k]
            k += 1
    return [str(x) for x in child]


def tournament(pop: List[List[str]], problem: Dict[str, Any], rng: random.Random, k: int = 3) -> List[str]:
    sample = rng.sample(pop, min(k, len(pop)))
    return min(sample, key=lambda seq: sequence_score(seq, problem))


def object_lookup(problem: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {obj["id"]: obj for obj in problem.get("objects", [])}


def part_number_rank(value: str) -> Tuple[int, str]:
    match = re.search(r"(\d+)", value or "")
    return (int(match.group(1)) if match else 10**9, value or "")


def station_number(obj: Dict[str, Any]) -> int:
    text = f"{obj.get('display_name') or ''} {obj.get('part_number') or ''}"
    match = re.search(r"(\d+)", text)
    if match:
        return int(match.group(1))
    code_match = re.search(r"5536C(\d{3})", obj.get("part_number") or "")
    if code_match:
        return int(code_match.group(1))
    return 999


def subtype_rank(obj: Dict[str, Any]) -> int:
    text = f"{obj.get('category') or ''} {obj.get('display_name') or ''} {obj.get('name') or ''}".lower()
    if "beam" in text or "aux" in text:
        return 0
    if "rib" in text:
        return 1
    if "angle" in text:
        return 2
    if "stringer" in text:
        return 3
    if "panel" in text or "skin" in text or "plate" in text:
        return 4
    return 9


def process_phase_rank(obj: Dict[str, Any]) -> int:
    category = obj.get("category") or ""
    name = (obj.get("display_name") or obj.get("name") or "")
    text = f"{category} {name}".lower()
    if "beam" in text or "aux" in text:
        return 1
    if "rib" in text or "angle" in text or "stringer" in text or "plate" in text:
        return 2
    if "panel" in text or "skin" in text:
        return 3
    return 9


def knowledge_rank(oid: str, lookup: Dict[str, Dict[str, Any]]) -> Tuple[int, int, int, int, str]:
    obj = lookup.get(oid, {})
    return (
        process_phase_rank(obj),
        station_number(obj),
        subtype_rank(obj),
        part_number_rank(obj.get("part_number") or obj.get("name") or "")[0],
        oid,
    )


def knowledge_topological_sort(problem: Dict[str, Any], rng: random.Random, jitter: float = 0.05) -> List[str]:
    nodes = problem["object_ids"]
    edges = edge_pairs(problem)
    lookup = object_lookup(problem)
    succ = {n: set() for n in nodes}
    indeg = {n: 0 for n in nodes}
    for before, after in edges:
        if before in succ and after in succ and after not in succ[before]:
            succ[before].add(after)
            indeg[after] += 1
    available = [n for n in nodes if indeg[n] == 0]
    sequence: List[str] = []
    while available:
        scored = []
        for node in available:
            local = 0.0 if not sequence else evaluate_sequence([sequence[-1], node], problem)["fitness"]
            rank = knowledge_rank(node, lookup)
            scored.append((local, *rank, rng.random() * jitter, node))
        chosen = min(scored)[-1]
        available.remove(chosen)
        sequence.append(chosen)
        for nxt in sorted(succ[chosen]):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                available.append(nxt)
    if len(sequence) != len(nodes):
        raise ValueError("hard precedence edges contain a cycle; knowledge topological sort failed")
    return sequence


def make_initial_population(problem: Dict[str, Any], size: int, rng: random.Random, knowledge: bool = False) -> List[List[str]]:
    nodes = problem["object_ids"]
    edges = edge_pairs(problem)
    pop = []
    for _ in range(size):
        # KG-IGA deliberately uses the same random feasible initialization as GA.
        # The knowledge graph is queried only during lightweight repair/local search,
        # so the experiment reflects convergence improvement rather than initial-value advantage.
        pop.append(random_topological_sort(nodes, edges, rng))
    return pop


def record_initial_usage(pop: List[List[str]], problem: Dict[str, Any], usage: Dict[str, Any] | None) -> None:
    if usage is None or not pop:
        return
    scores = [sequence_score(seq, problem) for seq in pop]
    usage["initial_best_fitness"] = round(min(scores), 6)
    usage["initial_avg_fitness"] = round(mean(scores), 6)


def genetic_algorithm(
    problem: Dict[str, Any],
    seed: int,
    population_size: int,
    iterations: int,
    knowledge: bool = False,
    usage: Dict[str, Any] | None = None,
    initial_population: List[List[str]] | None = None,
) -> Tuple[List[str], List[float]]:
    rng = random.Random(seed)
    nodes = problem["object_ids"]
    edges = edge_pairs(problem)
    pop = [list(seq) for seq in initial_population] if initial_population is not None else make_initial_population(problem, population_size, rng, knowledge=knowledge)
    record_initial_usage(pop, problem, usage)
    best = min(pop, key=lambda s: sequence_score(s, problem))
    curve = [sequence_score(best, problem)]
    for _ in range(iterations):
        elite_count = max(2, population_size // 10)
        new_pop = sorted(pop, key=lambda s: sequence_score(s, problem))[:elite_count]
        while len(new_pop) < population_size:
            p1 = tournament(pop, problem, rng)
            p2 = tournament(pop, problem, rng)
            child = crossover_order(p1, p2, rng)
            if rng.random() < (0.55 if knowledge else 0.30):
                child = mutate_insert(child, rng)
                if knowledge and usage is not None:
                    add_usage(usage, "guided_mutation_count")
            raw_child = list(child)
            child = repair_to_feasible(child, nodes, edges)
            if usage is not None:
                add_usage(usage, "repair_count")
                if child != raw_child:
                    add_usage(usage, "repair_changed_count")
            if knowledge and rng.random() < 0.25:
                child = local_search(child, problem, rng, steps=3, usage=usage)
            new_pop.append(child)
        pop = new_pop
        current = min(pop, key=lambda s: sequence_score(s, problem))
        if sequence_score(current, problem) < sequence_score(best, problem):
            best = current
        if knowledge:
            best = local_search(best, problem, rng, steps=8, usage=usage)
            pop[0] = best
        curve.append(sequence_score(best, problem))
    return best, curve


def genetic_algorithm_constraint_mutation(problem: Dict[str, Any], seed: int, population_size: int, iterations: int, usage: Dict[str, Any]) -> Tuple[List[str], List[float]]:
    rng = random.Random(seed)
    nodes = problem["object_ids"]
    edges = edge_pairs(problem)
    pop = make_initial_population(problem, population_size, rng, knowledge=False)
    record_initial_usage(pop, problem, usage)
    best = min(pop, key=lambda s: sequence_score(s, problem))
    curve = [sequence_score(best, problem)]
    for _ in range(iterations):
        elite_count = max(2, population_size // 10)
        new_pop = sorted(pop, key=lambda s: sequence_score(s, problem))[:elite_count]
        while len(new_pop) < population_size:
            p1 = tournament(pop, problem, rng)
            p2 = tournament(pop, problem, rng)
            child = crossover_order(p1, p2, rng)
            if rng.random() < 0.55:
                child = mutate_insert_constraint_aware(child, nodes, edges, rng, usage)
                add_usage(usage, "guided_mutation_count")
            raw_child = list(child)
            child = repair_to_feasible(child, nodes, edges)
            add_usage(usage, "repair_count")
            if child != raw_child:
                add_usage(usage, "repair_changed_count")
            if rng.random() < 0.25:
                child = local_search(child, problem, rng, steps=3, usage=usage)
            new_pop.append(child)
        pop = new_pop
        current = min(pop, key=lambda s: sequence_score(s, problem))
        if sequence_score(current, problem) < sequence_score(best, problem):
            best = current
        best = local_search(best, problem, rng, steps=8, usage=usage)
        pop[0] = best
        curve.append(sequence_score(best, problem))
    return best, curve


def crossover_valid_or_retry(pop: List[List[str]], problem: Dict[str, Any], rng: random.Random, usage: Dict[str, Any], max_retries: int = 20) -> List[str] | None:
    edges = edge_pairs(problem)
    for attempt in range(max_retries):
        p1 = tournament(pop, problem, rng)
        p2 = tournament(pop, problem, rng)
        child = crossover_order(p1, p2, rng)
        if is_valid_sequence(child, edges):
            add_usage(usage, "no_repair_valid_child_count")
            return child
        add_usage(usage, "no_repair_illegal_child_count")
        if attempt < max_retries - 1:
            add_usage(usage, "no_repair_crossover_retry_count")
    return None


def local_search_no_repair(sequence: List[str], problem: Dict[str, Any], rng: random.Random, steps: int, usage: Dict[str, Any]) -> List[str]:
    nodes = problem["object_ids"]
    edges = edge_pairs(problem)
    best = list(sequence)
    best_score = sequence_score(best, problem)
    for _ in range(steps):
        if rng.random() < 0.60:
            neighbor = mutate_insert_constraint_aware(best, nodes, edges, rng, usage)
        elif len(best) > 2:
            neighbor = list(best)
            i = rng.randrange(len(neighbor) - 1)
            neighbor[i], neighbor[i + 1] = neighbor[i + 1], neighbor[i]
        else:
            neighbor = list(best)
        if not is_valid_sequence(neighbor, edges):
            add_usage(usage, "no_repair_local_reject_count")
            continue
        score = sequence_score(neighbor, problem)
        if score < best_score:
            add_usage(usage, "guided_accept_count")
            best, best_score = neighbor, score
        else:
            add_usage(usage, "guided_reject_count")
    return best


def genetic_algorithm_constraint_mutation_no_repair(problem: Dict[str, Any], seed: int, population_size: int, iterations: int, usage: Dict[str, Any]) -> Tuple[List[str], List[float]]:
    rng = random.Random(seed)
    nodes = problem["object_ids"]
    edges = edge_pairs(problem)
    pop = make_initial_population(problem, population_size, rng, knowledge=False)
    record_initial_usage(pop, problem, usage)
    best = min(pop, key=lambda s: sequence_score(s, problem))
    curve = [sequence_score(best, problem)]
    for _ in range(iterations):
        elite_count = max(2, population_size // 10)
        elites = sorted(pop, key=lambda s: sequence_score(s, problem))[:elite_count]
        new_pop = [list(seq) for seq in elites]
        while len(new_pop) < population_size:
            child = crossover_valid_or_retry(pop, problem, rng, usage)
            if child is None:
                child = list(rng.choice(elites))
            if rng.random() < 0.55:
                add_usage(usage, "guided_mutation_count")
                mutated = mutate_insert_constraint_aware(child, nodes, edges, rng, usage)
                if is_valid_sequence(mutated, edges):
                    child = mutated
                else:
                    add_usage(usage, "no_repair_illegal_child_count")
            if rng.random() < 0.25:
                child = local_search_no_repair(child, problem, rng, steps=3, usage=usage)
            if is_valid_sequence(child, edges):
                new_pop.append(child)
            else:
                add_usage(usage, "no_repair_illegal_child_count")
                new_pop.append(list(rng.choice(elites)))
        pop = new_pop
        current = min(pop, key=lambda s: sequence_score(s, problem))
        if sequence_score(current, problem) < sequence_score(best, problem):
            best = current
        best = local_search_no_repair(best, problem, rng, steps=8, usage=usage)
        pop[0] = best
        curve.append(sequence_score(best, problem))
    return best, curve


def pso(problem: Dict[str, Any], seed: int, population_size: int, iterations: int, initial_population: List[List[str]] | None = None) -> Tuple[List[str], List[float]]:
    rng = random.Random(seed)
    nodes = problem["object_ids"]
    edges = edge_pairs(problem)
    particles = [list(seq) for seq in initial_population] if initial_population is not None else make_initial_population(problem, population_size, rng)
    pbest = [list(p) for p in particles]
    gbest = min(pbest, key=lambda s: sequence_score(s, problem))
    curve = [sequence_score(gbest, problem)]
    for _ in range(iterations):
        for idx, particle in enumerate(particles):
            proposal = list(particle)
            if rng.random() < 0.55:
                proposal = crossover_order(proposal, pbest[idx], rng)
            if rng.random() < 0.55:
                proposal = crossover_order(proposal, gbest, rng)
            if rng.random() < 0.40:
                proposal = mutate_insert(proposal, rng)
            proposal = repair_to_feasible(proposal, nodes, edges)
            particles[idx] = proposal
            if sequence_score(proposal, problem) < sequence_score(pbest[idx], problem):
                pbest[idx] = proposal
        current = min(pbest, key=lambda s: sequence_score(s, problem))
        if sequence_score(current, problem) < sequence_score(gbest, problem):
            gbest = current
        curve.append(sequence_score(gbest, problem))
    return gbest, curve


def rl_action_score(previous: str | None, node: str, problem: Dict[str, Any], q_values: Dict[Tuple[str, str], float]) -> float:
    q_key = (previous or "__START__", node)
    if previous is None:
        return q_values.get(q_key, 0.0)
    local_cost = evaluate_sequence([previous, node], problem)["fitness"]
    return q_values.get(q_key, 0.0) - 0.05 * local_cost


def rl_construct_sequence(problem: Dict[str, Any], rng: random.Random, q_values: Dict[Tuple[str, str], float], epsilon: float) -> List[str]:
    nodes = problem["object_ids"]
    edges = edge_pairs(problem)
    succ = {n: set() for n in nodes}
    indeg = {n: 0 for n in nodes}
    for before, after in edges:
        if before in succ and after in succ and after not in succ[before]:
            succ[before].add(after)
            indeg[after] += 1
    available = [n for n in nodes if indeg[n] == 0]
    sequence: List[str] = []
    previous: str | None = None
    while available:
        if rng.random() < epsilon:
            chosen = rng.choice(available)
        else:
            chosen = max(
                available,
                key=lambda node: (rl_action_score(previous, node, problem, q_values), -available.index(node), rng.random()),
            )
        available.remove(chosen)
        sequence.append(chosen)
        previous = chosen
        for nxt in sorted(succ[chosen]):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                available.append(nxt)
    if len(sequence) != len(nodes):
        raise ValueError("hard precedence edges contain a cycle; RL construction failed")
    return sequence


def rl_update_q_values(sequence: Sequence[str], fitness: float, q_values: Dict[Tuple[str, str], float], alpha: float) -> None:
    reward = 1.0 / (1.0 + max(0.0, fitness))
    previous = "__START__"
    for node in sequence:
        key = (previous, node)
        q_values[key] = q_values.get(key, 0.0) + alpha * (reward - q_values.get(key, 0.0))
        previous = node


def reinforcement_learning_algorithm(
    problem: Dict[str, Any],
    seed: int,
    population_size: int,
    iterations: int,
    usage: Dict[str, Any] | None = None,
    initial_population: List[List[str]] | None = None,
) -> Tuple[List[str], List[float]]:
    rng = random.Random(seed)
    nodes = problem["object_ids"]
    edges = edge_pairs(problem)
    initial = [list(seq) for seq in initial_population] if initial_population is not None else make_initial_population(problem, population_size, rng)
    record_initial_usage(initial, problem, usage)
    q_values: Dict[Tuple[str, str], float] = {}
    for seq in initial:
        rl_update_q_values(seq, sequence_score(seq, problem), q_values, alpha=0.35)
    best = min(initial, key=lambda seq: sequence_score(seq, problem))
    curve = [sequence_score(best, problem)]
    for iteration in range(iterations):
        epsilon = max(0.05, 0.45 * (1.0 - iteration / max(1, iterations)))
        alpha = max(0.05, 0.30 * (1.0 - iteration / max(1, iterations)))
        candidates = [list(best)]
        for _ in range(max(1, population_size - 1)):
            candidate = rl_construct_sequence(problem, rng, q_values, epsilon)
            if rng.random() < 0.25:
                candidate = mutate_insert_constraint_aware(candidate, nodes, edges, rng, usage)
            candidate = repair_to_feasible(candidate, nodes, edges)
            if rng.random() < 0.20:
                candidate = local_search(candidate, problem, rng, steps=2, usage=usage)
            candidates.append(candidate)
        for candidate in candidates:
            score = sequence_score(candidate, problem)
            rl_update_q_values(candidate, score, q_values, alpha)
            if score < sequence_score(best, problem):
                best = candidate
        if usage is not None:
            add_usage(usage, "guided_mutation_count", max(1, population_size - 1))
            add_usage(usage, "guided_accept_count")
        curve.append(sequence_score(best, problem))
    if not is_valid_sequence(best, edges):
        best = repair_to_feasible(best, nodes, edges)
    return best, curve


def drl_feature_vector(
    previous: str | None,
    node: str,
    step_index: int,
    total_steps: int,
    available_count: int,
    problem: Dict[str, Any],
    node_index: Dict[str, int],
) -> List[float]:
    maps = problem.get("_objective_maps_cache")
    if maps is None:
        maps = build_objective_maps(problem)
        problem["_objective_maps_cache"] = maps
    reference = maps.get("reference_distance", 1.0) or 1.0
    projection_reference = maps.get("projection_reference", reference) or reference
    node_count = max(1, len(node_index))
    local_cost = 0.0
    distance_norm = 0.0
    reverse_flow = 0.0
    connected = 0.0
    resource_diff = 0.0
    if previous is not None:
        local_cost = min(10.0, evaluate_sequence([previous, node], problem)["fitness"] / 50.0)
        key = "||".join(sorted((previous, node)))
        connected = 1.0 if key in maps.get("connect", set()) else 0.0
        prev_center = maps.get("centers", {}).get(previous)
        node_center = maps.get("centers", {}).get(node)
        if prev_center and node_center:
            distance_norm = min(10.0, sum((prev_center[dim] - node_center[dim]) ** 2 for dim in range(3)) ** 0.5 / reference)
        prev_projection = maps.get("projections", {}).get(previous)
        node_projection = maps.get("projections", {}).get(node)
        if prev_projection is not None and node_projection is not None:
            reverse_flow = max(0.0, (prev_projection - node_projection) / projection_reference)
        resources_prev = maps.get("resources", {}).get(previous, set())
        resources_node = maps.get("resources", {}).get(node, set())
        resource_diff = min(10.0, len(resources_prev ^ resources_node) / 5.0)
    return [
        step_index / max(1, total_steps - 1),
        available_count / max(1, total_steps),
        node_index.get(node, 0) / node_count,
        local_cost,
        distance_norm,
        reverse_flow,
        connected,
        resource_diff,
    ]


def drl_construct_sequence(
    problem: Dict[str, Any],
    rng: random.Random,
    model: Any,
    torch: Any,
    node_index: Dict[str, int],
    epsilon: float,
) -> List[str]:
    nodes = problem["object_ids"]
    edges = edge_pairs(problem)
    succ = {n: set() for n in nodes}
    indeg = {n: 0 for n in nodes}
    for before, after in edges:
        if before in succ and after in succ and after not in succ[before]:
            succ[before].add(after)
            indeg[after] += 1
    available = [n for n in nodes if indeg[n] == 0]
    sequence: List[str] = []
    previous: str | None = None
    while available:
        if rng.random() < epsilon:
            chosen = rng.choice(available)
        else:
            features = [
                drl_feature_vector(previous, node, len(sequence), len(nodes), len(available), problem, node_index)
                for node in available
            ]
            with torch.no_grad():
                scores = model(torch.tensor(features, dtype=torch.float32)).reshape(-1).tolist()
            chosen = available[max(range(len(available)), key=lambda idx: (scores[idx], rng.random()))]
        available.remove(chosen)
        sequence.append(chosen)
        previous = chosen
        for nxt in sorted(succ[chosen]):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                available.append(nxt)
    if len(sequence) != len(nodes):
        raise ValueError("hard precedence edges contain a cycle; DRL construction failed")
    return sequence


def drl_sequence_features(sequence: Sequence[str], problem: Dict[str, Any], node_index: Dict[str, int]) -> List[List[float]]:
    features = []
    previous: str | None = None
    total = len(sequence)
    for step, node in enumerate(sequence):
        features.append(drl_feature_vector(previous, node, step, total, max(1, total - step), problem, node_index))
        previous = node
    return features


def drl_train_on_sequences(
    model: Any,
    optimizer: Any,
    torch: Any,
    sequences: Sequence[Sequence[str]],
    problem: Dict[str, Any],
    node_index: Dict[str, int],
) -> None:
    batch_x: List[List[float]] = []
    batch_y: List[float] = []
    for seq in sequences:
        fitness = sequence_score(seq, problem)
        reward = 1.0 / (1.0 + max(0.0, fitness))
        batch_x.extend(drl_sequence_features(seq, problem, node_index))
        batch_y.extend([reward] * len(seq))
    if not batch_x:
        return
    x = torch.tensor(batch_x, dtype=torch.float32)
    y = torch.tensor(batch_y, dtype=torch.float32).reshape(-1, 1)
    for _ in range(2):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()


def deep_reinforcement_learning_algorithm(
    problem: Dict[str, Any],
    seed: int,
    population_size: int,
    iterations: int,
    usage: Dict[str, Any] | None = None,
    initial_population: List[List[str]] | None = None,
) -> Tuple[List[str], List[float]]:
    import torch

    rng = random.Random(seed)
    torch.manual_seed(seed)
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    nodes = problem["object_ids"]
    edges = edge_pairs(problem)
    node_index = {node: idx for idx, node in enumerate(sorted(nodes))}
    initial = [list(seq) for seq in initial_population] if initial_population is not None else make_initial_population(problem, population_size, rng)
    record_initial_usage(initial, problem, usage)
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 1),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    drl_train_on_sequences(model, optimizer, torch, initial, problem, node_index)
    best = min(initial, key=lambda seq: sequence_score(seq, problem))
    curve = [sequence_score(best, problem)]
    replay: List[List[str]] = sorted((list(seq) for seq in initial), key=lambda seq: sequence_score(seq, problem))[: max(4, population_size // 4)]
    for iteration in range(iterations):
        epsilon = max(0.05, 0.40 * (1.0 - iteration / max(1, iterations)))
        candidates = [list(best)]
        for _ in range(max(1, population_size - 1)):
            candidate = drl_construct_sequence(problem, rng, model, torch, node_index, epsilon)
            if rng.random() < 0.20:
                candidate = mutate_insert_constraint_aware(candidate, nodes, edges, rng, usage)
            candidate = repair_to_feasible(candidate, nodes, edges)
            if rng.random() < 0.15:
                candidate = local_search(candidate, problem, rng, steps=2, usage=usage)
            candidates.append(candidate)
        candidates = sorted(candidates, key=lambda seq: sequence_score(seq, problem))
        if sequence_score(candidates[0], problem) < sequence_score(best, problem):
            best = candidates[0]
        replay = sorted(replay + candidates[: max(4, population_size // 4)], key=lambda seq: sequence_score(seq, problem))[: max(8, population_size // 2)]
        drl_train_on_sequences(model, optimizer, torch, replay, problem, node_index)
        if usage is not None:
            add_usage(usage, "guided_mutation_count", max(1, population_size - 1))
            add_usage(usage, "guided_accept_count")
        curve.append(sequence_score(best, problem))
    if not is_valid_sequence(best, edges):
        best = repair_to_feasible(best, nodes, edges)
    return best, curve


def kg_lightweight_neighbor(sequence: List[str], problem: Dict[str, Any], rng: random.Random) -> List[str]:
    maps = problem.get("_objective_maps_cache")
    if maps is None:
        maps = build_objective_maps(problem)
        problem["_objective_maps_cache"] = maps
    connect = set() if problem.get("_disable_connect_guidance") else maps.get("connect", set())
    centers = maps.get("centers", {})
    projections = maps.get("projections", {})
    reference_distance = maps.get("reference_distance", 1.0) or 1.0
    if len(sequence) < 3:
        return list(sequence)
    idx = rng.randrange(len(sequence))
    node = sequence[idx]
    candidates = []
    for j, other in enumerate(sequence):
        if j == idx:
            continue
        key = "||".join(sorted((node, other)))
        score = 0
        if not problem.get("_disable_connect_guidance") and key in connect:
            score -= 3
        if node in centers and other in centers:
            distance = sum((centers[node][dim] - centers[other][dim]) ** 2 for dim in range(3)) ** 0.5
            score += distance / reference_distance
        if node in projections and other in projections:
            score += max(0.0, projections[node] - projections[other])
        candidates.append((score, abs(idx - j), j))
    if not candidates:
        return mutate_insert(sequence, rng)
    _, _, target = min(candidates)
    new = list(sequence)
    item = new.pop(idx)
    insert_at = target if target < idx else target
    if rng.random() < 0.5:
        insert_at += 1
    insert_at = max(0, min(len(new), insert_at))
    new.insert(insert_at, item)
    return new


def spatial_projection_neighbor(sequence: List[str], problem: Dict[str, Any], rng: random.Random, window: int = 4) -> List[str]:
    maps = problem.get("_objective_maps_cache")
    if maps is None:
        maps = build_objective_maps(problem)
        problem["_objective_maps_cache"] = maps
    projections = maps.get("projections", {})
    centers = maps.get("centers", {})
    reference_distance = maps.get("reference_distance", 1.0) or 1.0
    threshold = maps.get("spatial_jump_threshold", reference_distance * 1.75)
    projection_reference = maps.get("projection_reference", reference_distance)
    if len(sequence) < 3:
        return list(sequence)
    bad_pairs = []
    for idx, (a, b) in enumerate(zip(sequence, sequence[1:])):
        if a not in centers or b not in centers:
            continue
        distance = sum((centers[a][dim] - centers[b][dim]) ** 2 for dim in range(3)) ** 0.5
        reverse = projections.get(a, 0.0) - projections.get(b, 0.0)
        if distance > threshold or (idx > 0 and reverse > 0.03 * projection_reference):
            bad_pairs.append((distance / reference_distance + max(0.0, reverse), idx))
    if not bad_pairs:
        return list(sequence)
    _, pivot = max(bad_pairs)
    left = max(0, pivot - window)
    right = min(len(sequence), pivot + window + 2)
    new = list(sequence)
    local = new[left:right]
    original_pos = {node: pos for pos, node in enumerate(new)}
    local.sort(key=lambda node: (projections.get(node, 10**9), original_pos[node], rng.random()))
    new[left:right] = local
    return new


def local_search(sequence: List[str], problem: Dict[str, Any], rng: random.Random, steps: int = 50, usage: Dict[str, Any] | None = None) -> List[str]:
    nodes = problem["object_ids"]
    edges = edge_pairs(problem)
    best = list(sequence)
    best_score = sequence_score(best, problem)
    for _ in range(steps):
        neighbor = list(best)
        if rng.random() < 0.35:
            neighbor = spatial_projection_neighbor(neighbor, problem, rng)
            if usage is not None:
                add_usage(usage, "guided_mutation_count")
                add_usage(usage, "obb_query_count")
        elif rng.random() < 0.65:
            neighbor = kg_lightweight_neighbor(neighbor, problem, rng)
            if usage is not None:
                add_usage(usage, "guided_mutation_count")
                if not problem.get("_disable_connect_guidance"):
                    add_usage(usage, "connect_query_count")
                add_usage(usage, "obb_query_count")
        elif len(neighbor) > 2 and rng.random() < 0.45:
            i = rng.randrange(len(neighbor) - 1)
            neighbor[i], neighbor[i + 1] = neighbor[i + 1], neighbor[i]
        else:
            neighbor = mutate_insert(neighbor, rng)
        if usage is not None:
            add_usage(usage, "repair_count")
        raw_neighbor = list(neighbor)
        neighbor = repair_to_feasible(neighbor, nodes, edges)
        if usage is not None and neighbor != raw_neighbor:
            add_usage(usage, "repair_changed_count")
        score = sequence_score(neighbor, problem)
        if score < best_score:
            if usage is not None:
                add_usage(usage, "guided_accept_count")
            best, best_score = neighbor, score
        elif usage is not None:
            add_usage(usage, "guided_reject_count")
    return best


def local_search_geometry_hard(sequence: List[str], problem: Dict[str, Any], rng: random.Random, steps: int = 50, usage: Dict[str, Any] | None = None) -> List[str]:
    nodes = problem["object_ids"]
    edges = edge_pairs(problem)
    score_problem = v5_scoring_problem(problem)
    best = list(sequence)
    best_score = sequence_score(best, score_problem)
    for _ in range(steps):
        neighbor = list(best)
        roll = rng.random()
        if roll < 0.35:
            neighbor = spatial_projection_neighbor(neighbor, score_problem, rng)
            if usage is not None:
                add_usage(usage, "guided_mutation_count")
                add_usage(usage, "obb_query_count")
        elif roll < 0.65:
            neighbor = kg_lightweight_neighbor(neighbor, score_problem, rng)
            if usage is not None:
                add_usage(usage, "guided_mutation_count")
                add_usage(usage, "obb_query_count")
        elif len(neighbor) > 2 and rng.random() < 0.45:
            i = rng.randrange(len(neighbor) - 1)
            neighbor[i], neighbor[i + 1] = neighbor[i + 1], neighbor[i]
        else:
            neighbor = mutate_insert(neighbor, rng)
        if usage is not None:
            add_usage(usage, "repair_count")
        raw_neighbor = list(neighbor)
        neighbor = repair_to_feasible(neighbor, nodes, edges)
        if usage is not None and neighbor != raw_neighbor:
            add_usage(usage, "repair_changed_count")
        if not is_geometry_feasible_sequence(neighbor, problem):
            if usage is not None:
                add_usage(usage, "geometry_invalid_neighbor_count")
                add_usage(usage, "guided_reject_count")
            continue
        if usage is not None:
            add_usage(usage, "geometry_valid_neighbor_count")
        score = sequence_score(neighbor, score_problem)
        if score < best_score:
            if usage is not None:
                add_usage(usage, "guided_accept_count")
            best, best_score = neighbor, score
        elif usage is not None:
            add_usage(usage, "guided_reject_count")
    return best


def guided_rate(iteration: int, iterations: int) -> float:
    if iterations <= 1:
        return 0.60
    return min(0.60, 0.15 + 0.45 * (iteration / max(1, iterations - 1)))


def local_pair_penalty(a: str, b: str, previous_dirs: Tuple[Any, ...] | None, problem: Dict[str, Any], usage: Dict[str, Any]) -> float:
    maps = problem.get("_objective_maps_cache")
    if maps is None:
        maps = build_objective_maps(problem)
        problem["_objective_maps_cache"] = maps
    key = "||".join(sorted((a, b)))
    penalty = 0.0
    if not problem.get("_disable_connect_guidance") and not problem.get("_disable_connect_objective"):
        add_usage(usage, "connect_query_count")
        if key not in maps.get("connect", set()):
            penalty += 1.0
    add_usage(usage, "obb_query_count")
    center_a = maps.get("centers", {}).get(a)
    center_b = maps.get("centers", {}).get(b)
    if center_a and center_b:
        distance = sum((center_a[dim] - center_b[dim]) ** 2 for dim in range(3)) ** 0.5
        penalty += distance / (maps.get("reference_distance", 1.0) or 1.0)
    add_usage(usage, "mayinterfere_query_count")
    dirs = maps.get("directions", {}).get(key)
    if dirs and previous_dirs is not None and dirs != previous_dirs:
        penalty += 1.0
    add_usage(usage, "resource_query_count")
    resources_a = maps.get("resources", {}).get(a, set())
    resources_b = maps.get("resources", {}).get(b, set())
    if resources_a != resources_b:
        penalty += 0.5 * max(1, len(resources_a ^ resources_b))
    return penalty


def bad_pair_indices(sequence: Sequence[str], problem: Dict[str, Any], usage: Dict[str, Any], top_k: int = 4, sample_size: int = 10, rng: random.Random | None = None) -> List[int]:
    scored = []
    previous_dirs = None
    maps = problem.get("_objective_maps_cache")
    if maps is None:
        maps = build_objective_maps(problem)
        problem["_objective_maps_cache"] = maps
    indices = list(range(max(0, len(sequence) - 1)))
    if rng is not None and len(indices) > sample_size:
        indices = rng.sample(indices, sample_size)
    for idx in indices:
        a, b = sequence[idx], sequence[idx + 1]
        score = local_pair_penalty(a, b, previous_dirs, problem, usage)
        dirs = maps.get("directions", {}).get("||".join(sorted((a, b))))
        if dirs:
            previous_dirs = dirs
        scored.append((score, idx))
    bad = [idx for score, idx in sorted(scored, reverse=True)[:top_k] if score > 0]
    if bad:
        add_usage(usage, "bad_position_detect_count", len(bad))
    return bad


def guided_local_adjustment(sequence: List[str], problem: Dict[str, Any], rng: random.Random, usage: Dict[str, Any], window: int = 3) -> List[str]:
    if len(sequence) < 4:
        add_usage(usage, "guided_reject_count")
        return list(sequence)
    base_score = sequence_score(sequence, problem)
    bad_indices = bad_pair_indices(sequence, problem, usage, rng=rng)
    if not bad_indices:
        add_usage(usage, "guided_reject_count")
        return list(sequence)
    pivot = rng.choice(bad_indices)
    left = max(0, pivot - window)
    right = min(len(sequence), pivot + window + 2)
    best = list(sequence)
    best_score = base_score
    candidates: List[List[str]] = []

    if pivot + 1 < len(sequence):
        swapped = list(sequence)
        swapped[pivot], swapped[pivot + 1] = swapped[pivot + 1], swapped[pivot]
        candidates.append(swapped)

    local_nodes = sequence[left:right]
    for source in range(left, right):
        for target in range(left, right):
            if source == target:
                continue
            moved = list(sequence)
            item = moved.pop(source)
            insert_at = target if target < source else target - 1
            insert_at = max(left, min(right - 1, insert_at))
            moved.insert(insert_at, item)
            candidates.append(moved)

    if local_nodes:
        maps = problem.get("_objective_maps_cache")
        if maps is None:
            maps = build_objective_maps(problem)
            problem["_objective_maps_cache"] = maps
        guided_local = list(sequence)
        original_pos = {node: pos for pos, node in enumerate(sequence)}
        ordered = sorted(local_nodes, key=lambda node: (
            maps.get("projections", {}).get(node, 10**9),
            original_pos[node],
            rng.random(),
        ))
        guided_local[left:right] = ordered
        candidates.append(guided_local)

    nodes = problem["object_ids"]
    edges = edge_pairs(problem)
    rng.shuffle(candidates)
    candidates = candidates[:8]
    for candidate in candidates:
        add_usage(usage, "repair_count")
        repaired = repair_to_feasible(candidate, nodes, edges)
        if repaired != candidate:
            add_usage(usage, "repair_changed_count")
        score = sequence_score(repaired, problem)
        if score < best_score:
            best, best_score = repaired, score
    if best_score < base_score:
        add_usage(usage, "guided_accept_count")
        return best
    add_usage(usage, "guided_reject_count")
    return list(sequence)


def sampled_local_penalty(sequence: Sequence[str], problem: Dict[str, Any], rng: random.Random, usage: Dict[str, Any], sample_size: int = 4) -> float:
    if len(sequence) < 2:
        return 0.0
    indices = list(range(len(sequence) - 1))
    if len(indices) > sample_size:
        indices = rng.sample(indices, sample_size)
    total = 0.0
    previous_dirs = None
    maps = problem.get("_objective_maps_cache")
    if maps is None:
        maps = build_objective_maps(problem)
        problem["_objective_maps_cache"] = maps
    for idx in indices:
        a, b = sequence[idx], sequence[idx + 1]
        total += local_pair_penalty(a, b, previous_dirs, problem, usage)
        dirs = maps.get("directions", {}).get("||".join(sorted((a, b))))
        if dirs:
            previous_dirs = dirs
    return total / max(1, len(indices))


def prescreen_tournament(pop: List[List[str]], problem: Dict[str, Any], rng: random.Random, usage: Dict[str, Any], k: int = 3) -> List[str]:
    sample = rng.sample(pop, min(k, len(pop)))
    add_usage(usage, "prescreen_count", len(sample))
    penalties = [(sampled_local_penalty(seq, problem, rng, usage), sequence_score(seq, problem), seq) for seq in sample]
    threshold = sorted(p for p, _, _ in penalties)[len(penalties) // 2]
    reasonable = [(p, score, seq) for p, score, seq in penalties if p <= threshold]
    add_usage(usage, "prescreen_reject_count", len(penalties) - len(reasonable))
    return list(min(reasonable or penalties, key=lambda item: (item[1], item[0]))[2])


def prescreen_mutation(child: List[str], problem: Dict[str, Any], rng: random.Random, usage: Dict[str, Any], attempts: int = 3) -> List[str]:
    base_penalty = sampled_local_penalty(child, problem, rng, usage)
    candidates = []
    for _ in range(attempts):
        add_usage(usage, "prescreen_count")
        proposal = mutate_insert(child, rng)
        penalty = sampled_local_penalty(proposal, problem, rng, usage)
        if penalty <= base_penalty * 1.10 + 0.05:
            candidates.append((penalty, proposal))
        else:
            add_usage(usage, "prescreen_reject_count")
    if not candidates:
        add_usage(usage, "guided_reject_count")
        return child
    add_usage(usage, "guided_accept_count")
    return min(candidates, key=lambda item: item[0])[1]


def resource_distance(a: str, b: str, maps: Dict[str, Any]) -> int:
    resources_a = maps.get("resources", {}).get(a, set())
    resources_b = maps.get("resources", {}).get(b, set())
    return len(resources_a ^ resources_b)


def center_distance(a: str, b: str, maps: Dict[str, Any]) -> float:
    center_a = maps.get("centers", {}).get(a)
    center_b = maps.get("centers", {}).get(b)
    if not center_a or not center_b:
        return 10**9
    return sum((center_a[dim] - center_b[dim]) ** 2 for dim in range(3)) ** 0.5


def recommended_neighbor(anchor: str, sequence: Sequence[str], problem: Dict[str, Any], usage: Dict[str, Any]) -> str | None:
    maps = problem.get("_objective_maps_cache")
    if maps is None:
        maps = build_objective_maps(problem)
        problem["_objective_maps_cache"] = maps
    connect = set() if problem.get("_disable_connect_guidance") else maps.get("connect", set())
    others = [node for node in sequence if node != anchor]
    if not problem.get("_disable_connect_guidance"):
        add_usage(usage, "connect_query_count", len(others))
        connected = [node for node in others if "||".join(sorted((anchor, node))) in connect]
        if connected:
            add_usage(usage, "connect_candidate_count", len(connected))
            add_usage(usage, "obb_query_count", len(connected))
            return min(connected, key=lambda node: (center_distance(anchor, node, maps), resource_distance(anchor, node, maps), sequence.index(node)))

    centers = maps.get("centers", {})
    if anchor not in centers:
        return None
    reference = maps.get("reference_distance", 1.0) or 1.0
    threshold = maps.get("spatial_jump_threshold", reference * 1.75)
    spatial_candidates = [node for node in others if node in centers and center_distance(anchor, node, maps) <= threshold]
    add_usage(usage, "obb_query_count", len(others))
    if not spatial_candidates:
        return None
    add_usage(usage, "obb_fallback_count")
    add_usage(usage, "resource_query_count", len(spatial_candidates))
    return min(spatial_candidates, key=lambda node: (center_distance(anchor, node, maps), resource_distance(anchor, node, maps), sequence.index(node)))


def move_neighbor_next_to_anchor(sequence: Sequence[str], anchor: str, neighbor: str, rng: random.Random) -> List[str]:
    new = list(sequence)
    anchor_index = new.index(anchor)
    new.remove(neighbor)
    anchor_index = new.index(anchor)
    insert_at = anchor_index + (0 if rng.random() < 0.5 else 1)
    new.insert(insert_at, neighbor)
    return new


def direct_kg_guidance(sequence: List[str], problem: Dict[str, Any], rng: random.Random, usage: Dict[str, Any]) -> List[str]:
    if len(sequence) < 3:
        add_usage(usage, "guided_reject_count")
        return list(sequence)
    add_usage(usage, "direct_guide_attempt_count")
    anchor = rng.choice(sequence)
    neighbor = recommended_neighbor(anchor, sequence, problem, usage)
    if neighbor is None:
        add_usage(usage, "guided_reject_count")
        return list(sequence)
    guided = move_neighbor_next_to_anchor(sequence, anchor, neighbor, rng)
    if is_valid_sequence(guided, edge_pairs(problem)):
        add_usage(usage, "direct_guide_accept_count")
        add_usage(usage, "guided_accept_count")
        return guided
    add_usage(usage, "direct_guide_illegal_reject_count")
    add_usage(usage, "guided_reject_count")
    return list(sequence)


def genetic_algorithm_v2(problem: Dict[str, Any], seed: int, population_size: int, iterations: int, usage: Dict[str, Any]) -> Tuple[List[str], List[float]]:
    rng = random.Random(seed)
    nodes = problem["object_ids"]
    edges = edge_pairs(problem)
    pop = make_initial_population(problem, population_size, rng, knowledge=False)
    record_initial_usage(pop, problem, usage)
    best = min(pop, key=lambda s: sequence_score(s, problem))
    curve = [sequence_score(best, problem)]
    for iteration in range(iterations):
        rate = 0.35
        elite_count = max(2, population_size // 10)
        new_pop = sorted(pop, key=lambda s: sequence_score(s, problem))[:elite_count]
        while len(new_pop) < population_size:
            p1 = tournament(pop, problem, rng)
            p2 = tournament(pop, problem, rng)
            add_usage(usage, "total_crossover_count")
            child = crossover_order(p1, p2, rng)
            if rng.random() < 0.30:
                add_usage(usage, "total_mutation_count")
                child = mutate_insert(child, rng)
            add_usage(usage, "repair_count")
            raw_child = list(child)
            child = repair_to_feasible(child, nodes, edges)
            if child != raw_child:
                add_usage(usage, "repair_changed_count")
            if rng.random() < rate:
                add_usage(usage, "guided_mutation_count")
                child = direct_kg_guidance(child, problem, rng, usage)
            new_pop.append(child)
        pop = new_pop
        current = min(pop, key=lambda s: sequence_score(s, problem))
        if sequence_score(current, problem) < sequence_score(best, problem):
            best = current
        curve.append(sequence_score(best, problem))
    return best, curve


def genetic_algorithm_v3(problem: Dict[str, Any], seed: int, population_size: int, iterations: int, usage: Dict[str, Any]) -> Tuple[List[str], List[float]]:
    rng = random.Random(seed)
    nodes = problem["object_ids"]
    edges = edge_pairs(problem)
    pop = make_initial_population(problem, population_size, rng, knowledge=False)
    record_initial_usage(pop, problem, usage)
    best = min(pop, key=lambda s: sequence_score(s, problem))
    curve = [sequence_score(best, problem)]
    for iteration in range(iterations):
        rate = guided_rate(iteration, iterations)
        elite_count = max(2, population_size // 10)
        new_pop = sorted(pop, key=lambda s: sequence_score(s, problem))[:elite_count]
        while len(new_pop) < population_size:
            p1 = prescreen_tournament(pop, problem, rng, usage)
            p2 = prescreen_tournament(pop, problem, rng, usage)
            add_usage(usage, "total_crossover_count")
            child = crossover_order(p1, p2, rng)
            if rng.random() < rate:
                add_usage(usage, "guided_crossover_count")
                before = sampled_local_penalty(child, problem, rng, usage)
                alternative = crossover_order(p2, p1, rng)
                after = sampled_local_penalty(alternative, problem, rng, usage)
                if after <= before:
                    child = alternative
                    add_usage(usage, "guided_accept_count")
                else:
                    add_usage(usage, "guided_reject_count")
            if rng.random() < 0.30:
                add_usage(usage, "total_mutation_count")
                child = mutate_insert(child, rng)
            if rng.random() < rate:
                add_usage(usage, "total_mutation_count")
                add_usage(usage, "guided_mutation_count")
                child = prescreen_mutation(child, problem, rng, usage)
            raw_child = list(child)
            add_usage(usage, "repair_count")
            child = repair_to_feasible(child, nodes, edges)
            if child != raw_child:
                add_usage(usage, "repair_changed_count")
            if iteration >= int(iterations * 0.50) and rng.random() < 0.20:
                child = local_search(child, problem, rng, steps=2, usage=usage)
            new_pop.append(child)
        pop = new_pop
        current = min(pop, key=lambda s: sequence_score(s, problem))
        if sequence_score(current, problem) < sequence_score(best, problem):
            best = current
        curve.append(sequence_score(best, problem))
    return best, curve


def make_geometry_feasible_population(problem: Dict[str, Any], size: int, rng: random.Random, usage: Dict[str, Any]) -> List[List[str]]:
    nodes = problem["object_ids"]
    edges = edge_pairs(problem)
    pop: List[List[str]] = []
    attempts = 0
    max_attempts = max(size * 200, 1000)
    while len(pop) < size and attempts < max_attempts:
        attempts += 1
        seq = random_topological_sort(nodes, edges, rng)
        if is_geometry_feasible_sequence(seq, problem):
            add_usage(usage, "geometry_valid_initial_count")
            pop.append(seq)
        else:
            add_usage(usage, "geometry_invalid_initial_count")
    if len(pop) < size:
        raise RuntimeError(f"KG-IGAv5 could only initialize {len(pop)}/{size} geometry-feasible individuals")
    return pop


def genetic_algorithm_v5(
    problem: Dict[str, Any],
    seed: int,
    population_size: int,
    iterations: int,
    usage: Dict[str, Any],
    initial_population: List[List[str]] | None = None,
) -> Tuple[List[str], List[float]]:
    rng = random.Random(seed)
    nodes = problem["object_ids"]
    edges = edge_pairs(problem)
    score_problem = v5_scoring_problem(problem)
    if initial_population is None:
        pop = make_geometry_feasible_population(problem, population_size, rng, usage)
    else:
        pop = []
        for seq in initial_population:
            if is_geometry_feasible_sequence(seq, problem):
                add_usage(usage, "geometry_valid_initial_count")
                pop.append(list(seq))
            else:
                add_usage(usage, "geometry_invalid_initial_count")
        if len(pop) != len(initial_population):
            raise RuntimeError(f"KG-IGAv5 rejected {len(initial_population) - len(pop)} shared initial individuals")
    record_initial_usage(pop, score_problem, usage)
    best = min(pop, key=lambda s: sequence_score(s, score_problem))
    curve = [sequence_score(best, score_problem)]
    for _ in range(iterations):
        elite_count = max(2, population_size // 10)
        new_pop = sorted(pop, key=lambda s: sequence_score(s, score_problem))[:elite_count]
        while len(new_pop) < population_size:
            p1 = tournament(pop, score_problem, rng)
            p2 = tournament(pop, score_problem, rng)
            child = crossover_order(p1, p2, rng)
            if rng.random() < 0.55:
                child = mutate_insert(child, rng)
                add_usage(usage, "guided_mutation_count")
            raw_child = list(child)
            child = repair_to_feasible(child, nodes, edges)
            add_usage(usage, "repair_count")
            if child != raw_child:
                add_usage(usage, "repair_changed_count")
            if rng.random() < 0.25:
                child = local_search(child, score_problem, rng, steps=3, usage=usage)
            new_pop.append(child)
        pop = new_pop
        current = min(pop, key=lambda s: sequence_score(s, score_problem))
        if sequence_score(current, score_problem) < sequence_score(best, score_problem):
            best = current
        best = local_search(best, score_problem, rng, steps=8, usage=usage)
        pop[0] = best
        curve.append(sequence_score(best, score_problem))
    return best, curve


def convergence_iteration(curve: Sequence[float]) -> int:
    if not curve:
        return 0
    best = min(curve)
    for idx, value in enumerate(curve):
        if value == best:
            return idx
    return len(curve)


def run_algorithm(
    name: str,
    problem: Dict[str, Any],
    seed: int,
    population_size: int,
    iterations: int,
    usage: Dict[str, Any] | None = None,
    initial_population: List[List[str]] | None = None,
) -> Tuple[List[str], List[float]]:
    if name == "GraphPlan":
        return graphplan(problem, seed)
    if name == "PSO":
        return pso(problem, seed, population_size, iterations, initial_population=initial_population)
    if name == "GA":
        return genetic_algorithm(problem, seed, population_size, iterations, knowledge=False, usage=usage, initial_population=initial_population)
    if name == "KG-IGA":
        return genetic_algorithm(problem, seed, population_size, iterations, knowledge=True, usage=usage, initial_population=initial_population)
    if name == "RL":
        return reinforcement_learning_algorithm(problem, seed, population_size, iterations, usage=usage, initial_population=initial_population)
    if name == "DRL":
        return deep_reinforcement_learning_algorithm(problem, seed, population_size, iterations, usage=usage, initial_population=initial_population)
    if name == "KG-IGA-CM":
        return genetic_algorithm_constraint_mutation(problem, seed, population_size, iterations, usage or empty_knowledge_usage(name))
    if name == "KG-IGA-CM-NR":
        return genetic_algorithm_constraint_mutation_no_repair(problem, seed, population_size, iterations, usage or empty_knowledge_usage(name))
    if name == "KG-IGAv2":
        return genetic_algorithm_v2(problem, seed, population_size, iterations, usage or empty_knowledge_usage(name))
    if name == "KG-IGAv3":
        return genetic_algorithm_v3(problem, seed, population_size, iterations, usage or empty_knowledge_usage(name))
    if name == "KG-IGAv5":
        return genetic_algorithm_v5(problem, seed, population_size, iterations, usage or empty_knowledge_usage(name))
    raise ValueError(name)


def run_all(problem: Dict[str, Any], runs: int = 20, population_size: int = 40, iterations: int = 80, seed: int = 42) -> Dict[str, Any]:
    algorithms = ["GraphPlan", "PSO", "GA", "KG-IGA", "RL", "DRL", "KG-IGA-CM", "KG-IGA-CM-NR", "KG-IGAv2", "KG-IGAv3", "KG-IGAv5"]
    details: List[Dict[str, Any]] = []
    knowledge_usage_detail: List[Dict[str, Any]] = []
    best_sequences: Dict[str, Dict[str, Any]] = {}
    curves: Dict[str, List[List[float]]] = {name: [] for name in algorithms}
    shared_initial_by_run = {
        run: make_initial_population(problem, population_size, random.Random(seed + run * 101))
        for run in range(1, runs + 1)
    }
    for algorithm in algorithms:
        for run in range(1, runs + 1):
            usage = empty_knowledge_usage(algorithm, run)
            started = time.time()
            initial_population = shared_initial_by_run[run] if algorithm in {"PSO", "GA", "KG-IGA", "RL", "DRL"} else None
            seq, curve = run_algorithm(algorithm, problem, seed + run * 101 + len(algorithm), population_size, iterations, usage, initial_population=initial_population)
            elapsed = time.time() - started
            if not is_valid_sequence(seq, edge_pairs(problem)):
                raise AssertionError(f"{algorithm} produced invalid sequence")
            score = score_for_algorithm(seq, problem, algorithm)
            geometry = geometry_feasibility_details(seq, problem)
            row = {
                "algorithm": algorithm,
                "run": run,
                "population_size": 1 if algorithm == "GraphPlan" else population_size,
                "iterations": len(curve) if algorithm == "GraphPlan" else iterations,
                "convergence_iteration": convergence_iteration(curve),
                "direction_maintains": score["direction_maintains"],
                "tool_maintains": score["tool_maintains"],
                "continuity_maintains": score["continuity_maintains"],
                "connect_break_penalty": round(score["connect_break_penalty"], 6),
                "spatial_jump_penalty": round(score["spatial_jump_penalty"], 6),
                "spatial_flow_penalty": round(score["spatial_flow_penalty"], 6),
                "kg_continuity_penalty": round(score["kg_continuity_penalty"], 6),
                "fitness": score["fitness"],
                "runtime_s": round(elapsed, 6),
                "sequence": json.dumps(seq, ensure_ascii=False),
                "geometry_feasible": geometry["is_geometry_feasible"],
                "geometry_first_blocked_step": geometry["first_blocked_step"] or "",
            }
            details.append(row)
            curves[algorithm].append(curve)
            best = best_sequences.get(algorithm)
            if not best or score["fitness"] < best["score"]["fitness"]:
                best_sequences[algorithm] = {"sequence": seq, "score": score}
            knowledge_usage_detail.append(usage)

    global_best = min((row["fitness"] for row in details), default=0)
    summary = []
    for algorithm in algorithms:
        rows = [row for row in details if row["algorithm"] == algorithm]
        best_value = min(row["fitness"] for row in rows)
        hit_rate = 100.0 * sum(1 for row in rows if row["fitness"] == global_best) / len(rows)
        best_row = min(rows, key=lambda row: row["fitness"])
        summary.append({
            "algorithm": algorithm,
            "population_size": best_row["population_size"],
            "iterations": best_row["iterations"],
            "avg_convergence_iteration": round(mean(row["convergence_iteration"] for row in rows), 3),
            "direction_maintains": best_row["direction_maintains"],
            "tool_maintains": best_row["tool_maintains"],
            "continuity_maintains": best_row["continuity_maintains"],
            "connect_break_penalty": best_row["connect_break_penalty"],
            "spatial_jump_penalty": best_row["spatial_jump_penalty"],
            "spatial_flow_penalty": best_row["spatial_flow_penalty"],
            "kg_continuity_penalty": best_row["kg_continuity_penalty"],
            "best_fitness": best_value,
            "avg_fitness": round(mean(row["fitness"] for row in rows), 6),
            "best_hit_rate_percent": round(hit_rate, 2),
            "avg_runtime_s": round(mean(row["runtime_s"] for row in rows), 6),
        })
    return {
        "details": details,
        "summary": summary,
        "best_sequences": best_sequences,
        "curves": curves,
        "knowledge_usage_detail": knowledge_usage_detail,
        "knowledge_usage_summary": summarize_knowledge_usage(knowledge_usage_detail),
    }

def run_rl_vs_kgiga_shared_initial(problem: Dict[str, Any], runs: int = 20, population_size: int = 40, iterations: int = 80, seed: int = 42) -> Dict[str, Any]:
    algorithms = ["KG-IGA", "RL", "DRL"]
    details: List[Dict[str, Any]] = []
    knowledge_usage_detail: List[Dict[str, Any]] = []
    best_sequences: Dict[str, Dict[str, Any]] = {}
    curves: Dict[str, List[List[float]]] = {name: [] for name in algorithms}
    for run in range(1, runs + 1):
        run_seed = seed + run * 101
        initial_pop = make_initial_population(problem, population_size, random.Random(run_seed))
        initial_best = min(sequence_score(seq, problem) for seq in initial_pop)
        for algorithm in algorithms:
            usage = empty_knowledge_usage(algorithm, run)
            started = time.time()
            seq, curve = run_algorithm(
                algorithm,
                problem,
                run_seed + len(algorithm),
                population_size,
                iterations,
                usage,
                initial_population=initial_pop,
            )
            elapsed = time.time() - started
            if not curve or abs(curve[0] - initial_best) > 1e-9:
                raise AssertionError(f"{algorithm} does not share the same initial best fitness in run {run}")
            if not is_valid_sequence(seq, edge_pairs(problem)):
                raise AssertionError(f"{algorithm} produced invalid sequence")
            score = score_for_algorithm(seq, problem, algorithm)
            geometry = geometry_feasibility_details(seq, problem)
            row = {
                "algorithm": algorithm,
                "run": run,
                "population_size": population_size,
                "iterations": iterations,
                "convergence_iteration": convergence_iteration(curve),
                "direction_maintains": score["direction_maintains"],
                "tool_maintains": score["tool_maintains"],
                "continuity_maintains": score["continuity_maintains"],
                "connect_break_penalty": round(score["connect_break_penalty"], 6),
                "spatial_jump_penalty": round(score["spatial_jump_penalty"], 6),
                "spatial_flow_penalty": round(score["spatial_flow_penalty"], 6),
                "kg_continuity_penalty": round(score["kg_continuity_penalty"], 6),
                "fitness": score["fitness"],
                "runtime_s": round(elapsed, 6),
                "shared_initial_best_fitness": round(initial_best, 6),
                "geometry_feasible": geometry["is_geometry_feasible"],
                "geometry_first_blocked_step": geometry["first_blocked_step"] or "",
                "sequence": json.dumps(seq, ensure_ascii=False),
            }
            details.append(row)
            curves[algorithm].append(curve)
            best = best_sequences.get(algorithm)
            if not best or score["fitness"] < best["score"]["fitness"]:
                best_sequences[algorithm] = {"sequence": seq, "score": score}
            knowledge_usage_detail.append(usage)

    global_best = min((row["fitness"] for row in details), default=0)
    summary = []
    for algorithm in algorithms:
        rows = [row for row in details if row["algorithm"] == algorithm]
        best_value = min(row["fitness"] for row in rows)
        best_row = min(rows, key=lambda row: row["fitness"])
        summary.append({
            "algorithm": algorithm,
            "population_size": population_size,
            "iterations": iterations,
            "avg_convergence_iteration": round(mean(row["convergence_iteration"] for row in rows), 3),
            "direction_maintains": best_row["direction_maintains"],
            "tool_maintains": best_row["tool_maintains"],
            "continuity_maintains": best_row["continuity_maintains"],
            "connect_break_penalty": best_row["connect_break_penalty"],
            "spatial_jump_penalty": best_row["spatial_jump_penalty"],
            "spatial_flow_penalty": best_row["spatial_flow_penalty"],
            "kg_continuity_penalty": best_row["kg_continuity_penalty"],
            "best_fitness": best_value,
            "avg_fitness": round(mean(row["fitness"] for row in rows), 6),
            "best_hit_rate_percent": round(100.0 * sum(1 for row in rows if row["fitness"] == global_best) / len(rows), 2),
            "avg_runtime_s": round(mean(row["runtime_s"] for row in rows), 6),
        })
    return {
        "details": details,
        "summary": summary,
        "best_sequences": best_sequences,
        "curves": curves,
        "knowledge_usage_detail": knowledge_usage_detail,
        "knowledge_usage_summary": summarize_knowledge_usage(knowledge_usage_detail),
    }

def run_no_connect_shared_initial_comparison(problem: Dict[str, Any], runs: int = 20, population_size: int = 40, iterations: int = 80, seed: int = 42) -> Dict[str, Any]:
    scoped = dict(problem)
    scoped["_disable_connect_objective"] = True
    scoped["_disable_connect_guidance"] = True
    scoped.pop("_objective_maps_cache", None)
    algorithms = ["GA-no-connect", "KG-IGA-no-connect", "KG-IGAv5"]
    details: List[Dict[str, Any]] = []
    knowledge_usage_detail: List[Dict[str, Any]] = []
    best_sequences: Dict[str, Dict[str, Any]] = {}
    curves: Dict[str, List[List[float]]] = {name: [] for name in algorithms}
    for run in range(1, runs + 1):
        run_seed = seed + run * 101
        initial_pop = make_initial_population(scoped, population_size, random.Random(run_seed))
        shared_initial_geometry_valid = sum(1 for seq in initial_pop if is_geometry_feasible_sequence(seq, scoped))
        for algorithm in algorithms:
            usage = empty_knowledge_usage(algorithm, run)
            usage["geometry_valid_initial_count"] = shared_initial_geometry_valid if algorithm != "KG-IGAv5" else 0
            usage["geometry_invalid_initial_count"] = population_size - shared_initial_geometry_valid if algorithm != "KG-IGAv5" else 0
            started = time.time()
            if algorithm == "GA-no-connect":
                seq, curve = genetic_algorithm(scoped, run_seed, population_size, iterations, knowledge=False, usage=usage, initial_population=initial_pop)
            elif algorithm == "KG-IGA-no-connect":
                seq, curve = genetic_algorithm(scoped, run_seed, population_size, iterations, knowledge=True, usage=usage, initial_population=initial_pop)
            else:
                seq, curve = genetic_algorithm_v5(scoped, run_seed, population_size, iterations, usage=usage, initial_population=initial_pop)
            elapsed = time.time() - started
            if not is_valid_sequence(seq, edge_pairs(scoped)):
                raise AssertionError(f"{algorithm} produced invalid sequence")
            score = evaluate_sequence(seq, scoped)
            geometry = geometry_feasibility_details(seq, scoped)
            row = {
                "algorithm": algorithm,
                "run": run,
                "population_size": population_size,
                "iterations": iterations,
                "convergence_iteration": convergence_iteration(curve),
                "direction_maintains": score["direction_maintains"],
                "tool_maintains": score["tool_maintains"],
                "connect_break_penalty": round(score["connect_break_penalty"], 6),
                "spatial_jump_penalty": round(score["spatial_jump_penalty"], 6),
                "kg_continuity_penalty": round(score["kg_continuity_penalty"], 6),
                "fitness": score["fitness"],
                "runtime_s": round(elapsed, 6),
                "geometry_feasible": geometry["is_geometry_feasible"],
                "geometry_first_blocked_step": geometry["first_blocked_step"] or "",
                "shared_initial_geometry_valid": shared_initial_geometry_valid,
                "sequence": json.dumps(seq, ensure_ascii=False),
            }
            details.append(row)
            curves[algorithm].append(curve)
            best = best_sequences.get(algorithm)
            if not best or score["fitness"] < best["score"]["fitness"]:
                best_sequences[algorithm] = {"sequence": seq, "score": score}
            knowledge_usage_detail.append(usage)
    global_best = min((row["fitness"] for row in details), default=0)
    summary = []
    for algorithm in algorithms:
        rows = [row for row in details if row["algorithm"] == algorithm]
        best_value = min(row["fitness"] for row in rows)
        best_row = min(rows, key=lambda row: row["fitness"])
        summary.append({
            "algorithm": algorithm,
            "population_size": population_size,
            "iterations": iterations,
            "avg_convergence_iteration": round(mean(row["convergence_iteration"] for row in rows), 3),
            "direction_maintains": best_row["direction_maintains"],
            "tool_maintains": best_row["tool_maintains"],
            "connect_break_penalty": best_row["connect_break_penalty"],
            "spatial_jump_penalty": best_row["spatial_jump_penalty"],
            "kg_continuity_penalty": best_row["kg_continuity_penalty"],
            "best_fitness": best_value,
            "avg_fitness": round(mean(row["fitness"] for row in rows), 6),
            "best_hit_rate_percent": round(100.0 * sum(1 for row in rows if row["fitness"] == global_best) / len(rows), 2),
            "avg_runtime_s": round(mean(row["runtime_s"] for row in rows), 6),
            "avg_shared_initial_geometry_valid": round(mean(row["shared_initial_geometry_valid"] for row in rows), 3),
            "final_geometry_feasible_percent": round(100.0 * sum(1 for row in rows if row["geometry_feasible"]) / len(rows), 2),
        })
    return {
        "details": details,
        "summary": summary,
        "best_sequences": best_sequences,
        "curves": curves,
        "knowledge_usage_detail": knowledge_usage_detail,
        "knowledge_usage_summary": summarize_knowledge_usage(knowledge_usage_detail),
    }

def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_convergence_png(curves: Dict[str, List[List[float]]], path: Path, include_algorithms: Sequence[str] | None = None, title: str = "Convergence Curve") -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 5))
    for name, run_curves in curves.items():
        if include_algorithms is not None and name not in include_algorithms:
            continue
        max_len = max((len(c) for c in run_curves), default=0)
        if not max_len:
            continue
        avg = []
        for i in range(max_len):
            vals = [c[min(i, len(c) - 1)] for c in run_curves if c]
            avg.append(mean(vals))
        plt.plot(range(0, len(avg)), avg, label=name)
    plt.xlabel("Iteration")
    plt.ylabel("Best fitness")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()


def write_best_sequences(problem: Dict[str, Any], best_sequences: Dict[str, Dict[str, Any]], path: Path) -> None:
    names = {obj["id"]: obj.get("display_name") or obj.get("name") or obj["id"] for obj in problem["objects"]}
    lines = []
    for alg, data in best_sequences.items():
        lines.append(f"[{alg}] fitness={data['score']['fitness']}")
        for idx, oid in enumerate(data["sequence"], start=1):
            lines.append(f"{idx:02d}. {oid} | {names.get(oid, oid)}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", default=str(RESULT_DIR / "problem_snapshot.json"))
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--population-size", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--rl-vs-kgiga-only", action="store_true", help="Run only RL and KG-IGA with shared initial populations.")
    args = parser.parse_args()
    problem = load_problem(Path(args.snapshot))
    result = run_rl_vs_kgiga_shared_initial(problem, args.runs, args.population_size, args.iterations) if args.rl_vs_kgiga_only else run_all(problem, args.runs, args.population_size, args.iterations)
    suffix = "_rl_vs_kgiga" if args.rl_vs_kgiga_only else ""
    summary_fields = [
        "algorithm", "population_size", "iterations", "avg_convergence_iteration",
        "direction_maintains", "tool_maintains", "continuity_maintains",
        "connect_break_penalty", "spatial_jump_penalty", "spatial_flow_penalty",
        "kg_continuity_penalty", "best_fitness", "avg_fitness",
        "best_hit_rate_percent", "avg_runtime_s",
    ]
    detail_fields = [
        "algorithm", "run", "population_size", "iterations", "convergence_iteration",
        "direction_maintains", "tool_maintains", "continuity_maintains",
        "connect_break_penalty", "spatial_jump_penalty", "spatial_flow_penalty",
        "kg_continuity_penalty", "fitness", "runtime_s", "shared_initial_best_fitness",
        "geometry_feasible", "geometry_first_blocked_step", "sequence",
    ]
    write_csv(RESULT_DIR / f"algorithm_summary{suffix}.csv", result["summary"], summary_fields)
    write_csv(RESULT_DIR / f"all_runs_detail{suffix}.csv", result["details"], detail_fields)
    write_csv(RESULT_DIR / f"kgiga_knowledge_usage_detail{suffix}.csv", result["knowledge_usage_detail"], KNOWLEDGE_USAGE_FIELDS)
    write_csv(RESULT_DIR / f"kgiga_knowledge_usage_summary{suffix}.csv", result["knowledge_usage_summary"], KNOWLEDGE_USAGE_SUMMARY_FIELDS)
    if not args.rl_vs_kgiga_only:
        write_csv(RESULT_DIR / "kgiga_v2_knowledge_usage_detail.csv", result["knowledge_usage_detail"], KNOWLEDGE_USAGE_FIELDS)
        write_csv(RESULT_DIR / "kgiga_v2_knowledge_usage_summary.csv", result["knowledge_usage_summary"], KNOWLEDGE_USAGE_SUMMARY_FIELDS)
    write_json(RESULT_DIR / f"optimization_result{suffix}.json", result)
    write_convergence_png(result["curves"], RESULT_DIR / f"convergence_curve{suffix}.png")
    write_convergence_png(result["curves"], RESULT_DIR / f"convergence_curve_all_algorithms{suffix}.png", title="Convergence Curve - All Algorithms")
    write_convergence_png(
        result["curves"],
        RESULT_DIR / f"convergence_curve_kg_variants{suffix}.png",
        include_algorithms=["GA", "KG-IGA", "RL", "DRL", "KG-IGA-CM", "KG-IGA-CM-NR", "KG-IGAv2", "KG-IGAv3"],
        title="Convergence Curve - KG Variants",
    )
    write_convergence_png(
        result["curves"],
        RESULT_DIR / f"convergence_curve_rl_vs_kgiga{suffix}.png",
        include_algorithms=["KG-IGA", "RL", "DRL"],
        title="Convergence Curve - RL/DRL vs KG-IGA",
    )
    write_best_sequences(problem, result["best_sequences"], RESULT_DIR / f"best_object_sequences{suffix}.txt")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()




