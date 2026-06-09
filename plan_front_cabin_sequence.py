from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from candidate_sequence_generator import generate_candidate_pool, geometry_feasibility_details
from infer_process_tasks import infer_tasks, write_task_explainability, write_tasks_csv, write_tasks_to_neo4j, write_tasks_txt
from load_planning_problem import RESULT_DIR, fetch_problem, load_problem, write_json
from optimization_algorithms import (
    KNOWLEDGE_USAGE_FIELDS,
    KNOWLEDGE_USAGE_SUMMARY_FIELDS,
    run_all,
    run_no_connect_shared_initial_comparison,
    write_best_sequences,
    write_convergence_png,
    write_csv,
)
from kg_config import Neo4jConfig


def build_report(problem: Dict[str, Any], candidate_report: Dict[str, Any], optimization: Dict[str, Any], tasks_count: int, selected_algorithm: str) -> Dict[str, Any]:
    kg_continuity = "w_spatial * spatial_jump_penalty"
    if not problem.get("_disable_connect_objective"):
        kg_continuity = "w_connect * connect_break_penalty + w_spatial * spatial_jump_penalty"
    return {
        "project": "前缘舱装配序列规划算法",
        "method": "two-stage object-sequence optimization then process-task inference",
        "selected_task_algorithm": selected_algorithm,
        "planning_objects": problem["objects"],
        "quality": problem.get("quality", {}),
        "hard_constraints": {
            "count": len(problem.get("hard_precedence_edges", [])),
            "edges": problem.get("hard_precedence_edges", []),
            "sources": sorted({e.get("source", "") for e in problem.get("hard_precedence_edges", [])}),
        },
        "objective_sources": {
            "direction_switches": "mayInterfere.directions; missing pairs are skipped",
            "tool_switches": "Object -> requireProcess -> Process -> requiresResource -> Resource",
            "kg_continuity_penalty": kg_continuity,
            "spatial_jump_penalty": "3D Euclidean distance between adjacent Object OBB centers, normalized by median pairwise OBB-center distance; large jumps and reverse movement on the dominant OBB-center flow axis use nonlinear penalty",
            "excluded": ["manual reference sequence as hard coding", "part_number/name based regional continuity"],
        },
        "candidate_pool_report": candidate_report,
        "algorithm_summary": optimization["summary"],
        "best_sequences": optimization["best_sequences"],
        "inferred_task_count": tasks_count,
        "output_dir": str(RESULT_DIR),
    }


def build_geometry_report(problem: Dict[str, Any], optimization: Dict[str, Any]) -> Dict[str, Any]:
    report = {
        "method": "sequence-level geometry feasibility hard check",
        "all_directions": ["+X", "-X", "+Y", "-Y", "+Z", "-Z"],
        "rule": "At each step, union mayInterfere.directions against assembled objects; the step is feasible if at least one direction remains available.",
        "algorithms": {},
    }
    for algorithm, payload in optimization.get("best_sequences", {}).items():
        details = geometry_feasibility_details(payload["sequence"], problem)
        blocked_steps = [step for step in details["steps"] if not step["is_geometry_feasible"]]
        report["algorithms"][algorithm] = {
            "is_geometry_feasible": details["is_geometry_feasible"],
            "first_blocked_step": details["first_blocked_step"],
            "blocked_step_count": len(blocked_steps),
            "blocked_steps": blocked_steps[:20],
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    cfg = Neo4jConfig()
    parser.add_argument("--uri", default=cfg.uri)
    parser.add_argument("--user", default=cfg.user)
    parser.add_argument("--password", default=cfg.password)
    parser.add_argument("--candidate-pool-size", type=int, default=5000)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--population-size", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--task-algorithm", default="KG-IGA", choices=["GraphPlan", "PSO", "GA", "KG-IGA", "RL", "DRL", "KG-IGA-CM", "KG-IGA-CM-NR", "KG-IGAv2", "KG-IGAv3", "KG-IGAv5"])
    parser.add_argument("--write-tasks-to-neo4j", action="store_true")
    parser.add_argument("--disable-connect-objective", action="store_true", help="Ignore connect_break_penalty in the sequence objective for no-connect ablation.")
    parser.add_argument("--disable-connect-guidance", action="store_true", help="Ignore connect relations in KG local guidance for no-connect ablation.")
    parser.add_argument("--problem-snapshot", default="", help="Optional existing problem_snapshot.json to run offline without reading Neo4j.")
    parser.add_argument("--shared-initial-no-connect-comparison", action="store_true", help="Run GA/KG-IGA/KG-IGAv5 with the same no-connect objective, same seed, and same initial population.")
    args = parser.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    problem = load_problem(Path(args.problem_snapshot)) if args.problem_snapshot else fetch_problem(args.uri, args.user, args.password)
    no_connect_mode = args.disable_connect_objective or args.disable_connect_guidance
    shared_initial_mode = args.shared_initial_no_connect_comparison
    kgigav5_mode = args.task_algorithm == "KG-IGAv5"
    suffix = "_shared_initial_no_connect" if shared_initial_mode else ("_no_connect" if no_connect_mode else ("_kgigav5" if kgigav5_mode else ""))
    if args.disable_connect_objective:
        problem["_disable_connect_objective"] = True
    if args.disable_connect_guidance:
        problem["_disable_connect_guidance"] = True
    if kgigav5_mode or shared_initial_mode:
        problem["_enable_geometry_hard_check"] = True
        problem["_disable_connect_objective"] = True
        problem["_disable_connect_guidance"] = True
    problem.pop("_objective_maps_cache", None)
    write_json(RESULT_DIR / f"problem_snapshot{suffix}.json", problem)

    candidates, candidate_report = generate_candidate_pool(problem, args.candidate_pool_size)
    write_json(RESULT_DIR / f"candidate_pool{suffix}.json", {"sequences": candidates})
    write_json(RESULT_DIR / f"candidate_pool_report{suffix}.json", candidate_report)

    optimization = run_no_connect_shared_initial_comparison(problem, args.runs, args.population_size, args.iterations) if shared_initial_mode else run_all(problem, args.runs, args.population_size, args.iterations)
    summary_fields = ["算法", "种群规模", "最大迭代次数", "平均收敛迭代次数", "装配方向维持次数", "装配工具维持次数", "装配工艺连贯性维持次数", "连接断裂惩罚", "空间跨度惩罚", "空间流线折返惩罚", "KG连贯性惩罚", "最优目标函数值", "平均目标函数值", "最优解命中率(%)", "平均运行时间/s"]
    detail_fields = ["算法", "run", "种群规模", "最大迭代次数", "收敛迭代次数", "装配方向维持次数", "装配工具维持次数", "装配工艺连贯性维持次数", "连接断裂惩罚", "空间跨度惩罚", "空间流线折返惩罚", "KG连贯性惩罚", "目标函数值", "运行时间/s", "sequence"]
    if shared_initial_mode:
        summary_fields = ["算法", "种群规模", "最大迭代次数", "平均收敛迭代次数", "干涉方向维持次数", "工具维持次数", "连接断裂惩罚", "空间跨度惩罚", "KG连贯性惩罚", "最优目标函数值", "平均目标函数值", "最优解命中率(%)", "平均运行时间/s", "平均共享初始几何可行数", "最终几何可行率(%)"]
        detail_fields = ["算法", "run", "种群规模", "最大迭代次数", "收敛迭代次数", "干涉方向维持次数", "工具维持次数", "连接断裂惩罚", "空间跨度惩罚", "KG连贯性惩罚", "目标函数值", "运行时间/s", "geometry_feasible", "geometry_first_blocked_step", "shared_initial_geometry_valid", "sequence"]
    write_csv(RESULT_DIR / f"algorithm_summary{suffix}.csv", optimization["summary"], summary_fields)
    write_csv(RESULT_DIR / f"all_runs_detail{suffix}.csv", optimization["details"], detail_fields)
    write_csv(RESULT_DIR / f"kgiga_knowledge_usage_detail{suffix}.csv", optimization["knowledge_usage_detail"], KNOWLEDGE_USAGE_FIELDS)
    write_csv(RESULT_DIR / f"kgiga_knowledge_usage_summary{suffix}.csv", optimization["knowledge_usage_summary"], KNOWLEDGE_USAGE_SUMMARY_FIELDS)
    if no_connect_mode:
        write_csv(RESULT_DIR / "kgiga_knowledge_usage_no_connect.csv", optimization["knowledge_usage_summary"], KNOWLEDGE_USAGE_SUMMARY_FIELDS)
    else:
        write_csv(RESULT_DIR / "kgiga_v2_knowledge_usage_detail.csv", optimization["knowledge_usage_detail"], KNOWLEDGE_USAGE_FIELDS)
        write_csv(RESULT_DIR / "kgiga_v2_knowledge_usage_summary.csv", optimization["knowledge_usage_summary"], KNOWLEDGE_USAGE_SUMMARY_FIELDS)
    write_json(RESULT_DIR / f"optimization_result{suffix}.json", optimization)
    write_convergence_png(optimization["curves"], RESULT_DIR / f"convergence_curve{suffix}.png")
    write_convergence_png(optimization["curves"], RESULT_DIR / f"convergence_curve_all_algorithms{suffix}.png", title="Convergence Curve - All Algorithms")
    write_convergence_png(
        optimization["curves"],
        RESULT_DIR / f"convergence_curve_kg_variants{suffix}.png",
        include_algorithms=["GA", "KG-IGA", "RL", "DRL", "KG-IGA-CM", "KG-IGA-CM-NR", "KG-IGAv2", "KG-IGAv3", "KG-IGAv5"],
        title="Convergence Curve - KG Variants",
    )
    write_convergence_png(
        optimization["curves"],
        RESULT_DIR / f"convergence_curve_rl_vs_kgiga{suffix}.png",
        include_algorithms=["KG-IGA", "RL", "DRL"],
        title="Convergence Curve - RL/DRL vs KG-IGA",
    )
    if kgigav5_mode:
        write_convergence_png(
            optimization["curves"],
            RESULT_DIR / "convergence_curve_kgigav5_vs_kgiga.png",
            include_algorithms=["KG-IGA", "KG-IGAv5"],
            title="Convergence Curve - KG-IGA vs KG-IGAv5",
        )
        write_json(RESULT_DIR / "geometry_feasibility_report_kgigav5.json", build_geometry_report(problem, optimization))
    write_best_sequences(problem, optimization["best_sequences"], RESULT_DIR / f"best_object_sequences{suffix}.txt")

    best_sequence = optimization["best_sequences"][args.task_algorithm]["sequence"]
    tasks = infer_tasks(problem, best_sequence)
    write_tasks_csv(tasks, RESULT_DIR / f"inferred_process_task_sequence{suffix}.csv")
    write_tasks_txt(tasks, RESULT_DIR / f"inferred_process_task_sequence{suffix}.txt")
    write_task_explainability(
        problem,
        tasks,
        RESULT_DIR / f"inferred_process_task_sequence_with_names{suffix}.csv",
        RESULT_DIR / f"task_inference_explanation{suffix}.txt",
        RESULT_DIR / f"task_inference_explanation{suffix}.json",
    )
    if args.write_tasks_to_neo4j:
        write_tasks_to_neo4j(tasks, args.uri, args.user, args.password)

    report = build_report(problem, candidate_report, optimization, len(tasks), args.task_algorithm)
    write_json(RESULT_DIR / f"planning_report{suffix}.json", report)
    print(json.dumps({
        "output_dir": str(RESULT_DIR),
        "object_count": len(problem["objects"]),
        "hard_precedence_count": len(problem["hard_precedence_edges"]),
        "candidate_pool_size": len(candidates),
        "task_count": len(tasks),
        "selected_algorithm": args.task_algorithm,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
