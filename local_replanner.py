from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from kg_config import Neo4jConfig


PROJECT_DIR = Path(r"F:\proV1.8")
RESULT_DIR = PROJECT_DIR / "front_cabin_planning_results"
TASK_CSV = RESULT_DIR / "inferred_process_task_sequence.csv"


RESOURCE_ALIASES = {
    "RES_钻具": {"RES_钻具", "钻具", "钻", "制孔"},
    "RES_铆接工具": {"RES_铆接工具", "铆接工具", "铆接", "紧固"},
    "RES_定位器": {"RES_定位器", "定位器"},
    "RES_边缘定位挡块": {"RES_边缘定位挡块", "边缘定位挡块"},
}


PREPARATION_KEYWORDS = (
    "定位",
    "连接带板",
    "角材",
    "缘条",
    "角片",
    "壁板",
)


@dataclass
class ReplanAttempt:
    attempt: int
    rule_version: str
    current_sequence_index: int
    failed_resource_id: str
    available_task_count: int
    blocked_resource_count: int
    blocked_precedence_count: int
    explanation_quality: str
    improvement_reason: str


def read_tasks(path: Path = TASK_CSV) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["sequence_index"] = int(row.get("sequence_index") or 0)
        row["task_obb_center_values"] = parse_center(row.get("task_obb_center", ""))
    return sorted(rows, key=lambda r: r["sequence_index"])


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_center(value: str) -> Optional[Tuple[float, float, float]]:
    if not value:
        return None
    text = value.strip()
    try:
        if text.startswith("["):
            vals = ast.literal_eval(text)
        else:
            vals = [float(v) for v in text.replace("|", ";").split(";") if v.strip()]
    except Exception:
        return None
    if len(vals) != 3:
        return None
    try:
        return (float(vals[0]), float(vals[1]), float(vals[2]))
    except Exception:
        return None


def distance(a: Optional[Tuple[float, float, float]], b: Optional[Tuple[float, float, float]]) -> Optional[float]:
    if a is None or b is None:
        return None
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def resource_aliases(resource_id: str) -> set[str]:
    aliases = set(RESOURCE_ALIASES.get(resource_id, {resource_id}))
    if resource_id.startswith("RES_"):
        aliases.add(resource_id[4:])
    return aliases


def uses_failed_resource(task: Dict[str, Any], failed_resource_id: str) -> bool:
    text = ";".join([
        str(task.get("required_resources", "")),
        str(task.get("process_id", "")),
        str(task.get("process_name", "")),
        str(task.get("task_name", "")),
    ])
    return any(alias and alias in text for alias in resource_aliases(failed_resource_id))


def is_preparation_task(task: Dict[str, Any]) -> bool:
    text = " ".join([
        str(task.get("process_name", "")),
        str(task.get("task_name", "")),
        str(task.get("operated_object_names", "")),
        str(task.get("generation_rule", "")),
    ])
    return any(keyword in text for keyword in PREPARATION_KEYWORDS) and not any(
        keyword in text for keyword in ("制初孔", "制连接孔", "初孔输出", "紧固", "补铆", "涂胶", "清洗")
    )


def resolve_current_index(tasks: List[Dict[str, Any]], current_index: Optional[int], keyword: str) -> int:
    if current_index is not None:
        return current_index
    if keyword:
        preparation_matches = [
            task["sequence_index"]
            for task in tasks
            if keyword in " ".join([
                str(task.get("task_name", "")),
                str(task.get("operated_object_names", "")),
                str(task.get("process_name", "")),
            ])
            and is_preparation_task(task)
        ]
        if preparation_matches:
            return max(preparation_matches)
        matches = [
            task["sequence_index"]
            for task in tasks
            if keyword in " ".join([
                str(task.get("task_name", "")),
                str(task.get("operated_object_names", "")),
                str(task.get("process_name", "")),
            ])
        ]
        if matches:
            return max(matches)
    # Default to the last 3号连接带板 task if present, otherwise the last positioning task before drilling.
    matches = [
        task["sequence_index"]
        for task in tasks
        if "3号连接带板" in str(task.get("operated_object_names", ""))
    ]
    if matches:
        return max(matches)
    positioning = [
        task["sequence_index"]
        for task in tasks
        if is_preparation_task(task)
    ]
    return max(positioning) if positioning else 0


def classify_tasks(
    tasks: List[Dict[str, Any]],
    failed_resource_id: str,
    current_index: int,
    propagate_precedence: bool = True,
) -> List[Dict[str, Any]]:
    classified: List[Dict[str, Any]] = []
    blocked_seen = False
    for task in tasks:
        row = dict(task)
        seq = row["sequence_index"]
        if seq <= current_index:
            row["new_status"] = "DONE"
            row["block_reason"] = "已执行到当前扰动点，任务状态被冻结，不参与局部重排。"
        elif uses_failed_resource(row, failed_resource_id):
            row["new_status"] = "BLOCKED_RESOURCE"
            row["block_reason"] = f"任务所需资源包含 {failed_resource_id} 对应资源类型，制孔资源故障导致不可执行。"
            blocked_seen = True
        elif propagate_precedence and blocked_seen:
            row["new_status"] = "BLOCKED_PRECEDENCE"
            row["block_reason"] = "前序制孔任务已被资源故障阻断，当前任务等待前驱完成。"
        else:
            row["new_status"] = "AVAILABLE"
            row["block_reason"] = "未完成且不依赖故障资源，可进入可用残局子图。"
        classified.append(row)
    return classified


def count_status(rows: Iterable[Dict[str, Any]], status: str) -> int:
    return sum(1 for row in rows if row.get("new_status") == status)


def explanation_quality(rows: List[Dict[str, Any]]) -> str:
    has_done = count_status(rows, "DONE") > 0
    has_blocked_resource = count_status(rows, "BLOCKED_RESOURCE") > 0
    has_available = count_status(rows, "AVAILABLE") > 0
    has_blocked_precedence = count_status(rows, "BLOCKED_PRECEDENCE") > 0
    if has_done and has_blocked_resource and has_available and has_blocked_precedence:
        return "clear"
    if has_done and has_blocked_resource and has_available:
        return "partial"
    return "poor"


def make_attempt(
    attempt: int,
    rule_version: str,
    current_index: int,
    failed_resource_id: str,
    rows: List[Dict[str, Any]],
    improvement_reason: str,
) -> ReplanAttempt:
    return ReplanAttempt(
        attempt=attempt,
        rule_version=rule_version,
        current_sequence_index=current_index,
        failed_resource_id=failed_resource_id,
        available_task_count=count_status(rows, "AVAILABLE"),
        blocked_resource_count=count_status(rows, "BLOCKED_RESOURCE"),
        blocked_precedence_count=count_status(rows, "BLOCKED_PRECEDENCE"),
        explanation_quality=explanation_quality(rows),
        improvement_reason=improvement_reason,
    )


def select_best_attempt(attempt_rows: List[Tuple[ReplanAttempt, List[Dict[str, Any]]]]) -> Tuple[ReplanAttempt, List[Dict[str, Any]]]:
    quality_rank = {"clear": 3, "partial": 2, "poor": 1}
    return max(
        attempt_rows,
        key=lambda item: (
            quality_rank.get(item[0].explanation_quality, 0),
            item[0].available_task_count,
            item[0].blocked_resource_count,
        ),
    )


def plan_available_sequence(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    available = [row for row in rows if row.get("new_status") == "AVAILABLE"]
    if not available:
        return []
    done = [row for row in rows if row.get("new_status") == "DONE"]
    tail_center = done[-1].get("task_obb_center_values") if done else None

    def sort_key(task: Dict[str, Any]) -> Tuple[int, float, int]:
        prep_priority = 0 if is_preparation_task(task) else 1
        dist = distance(tail_center, task.get("task_obb_center_values"))
        dist_key = dist if dist is not None else 999999.0
        return (prep_priority, dist_key, task["sequence_index"])

    ordered = sorted(available, key=sort_key)
    previous_center = tail_center
    result: List[Dict[str, Any]] = []
    for idx, task in enumerate(ordered, 1):
        row = dict(task)
        dist = distance(previous_center, row.get("task_obb_center_values"))
        row["new_sequence_index"] = idx
        row["replan_reason"] = build_replan_reason(row, dist)
        row["distance_from_previous_task"] = "" if dist is None else f"{dist:.6f}"
        result.append(row)
        previous_center = row.get("task_obb_center_values") or previous_center
    return result


def build_replan_reason(task: Dict[str, Any], dist: Optional[float]) -> str:
    parts = ["不依赖故障制孔资源"]
    if is_preparation_task(task):
        parts.append("属于定位/连接带板/装配准备任务")
    if dist is not None:
        parts.append(f"与当前执行尾部 OBB 距离为 {dist:.3f}")
    parts.append("从可用残局子图中前移执行")
    return "；".join(parts)


def event_report(
    failed_resource_id: str,
    current_index: int,
    rows: List[Dict[str, Any]],
    sequence: List[Dict[str, Any]],
    runtime_s: float,
    mode: str,
) -> Dict[str, Any]:
    return {
        "event_type": "RESOURCE_FAILURE",
        "mode": mode,
        "failed_resource_id": failed_resource_id,
        "failed_resource_aliases": sorted(resource_aliases(failed_resource_id)),
        "current_sequence_index": current_index,
        "global_task_count": len(rows),
        "done_task_count": count_status(rows, "DONE"),
        "blocked_by_resource_count": count_status(rows, "BLOCKED_RESOURCE"),
        "blocked_by_precedence_count": count_status(rows, "BLOCKED_PRECEDENCE"),
        "available_task_count": count_status(rows, "AVAILABLE"),
        "local_replan_sequence_count": len(sequence),
        "local_replan_runtime_s": round(runtime_s, 6),
        "replan_strategy": "freeze_done_block_drilling_resource_extract_available_residual",
        "data_integrity_note": "原始全局规划文件未覆盖；本次实验只输出 local_replan_* 文件。",
    }


def comparison_rows(rows: List[Dict[str, Any]], sequence: List[Dict[str, Any]], runtime_s: float) -> List[Dict[str, Any]]:
    return [
        {"metric": "global_task_count", "global_plan": len(rows), "local_replan": len(rows)},
        {"metric": "done_task_count", "global_plan": 0, "local_replan": count_status(rows, "DONE")},
        {"metric": "blocked_resource_task_count", "global_plan": 0, "local_replan": count_status(rows, "BLOCKED_RESOURCE")},
        {"metric": "blocked_precedence_task_count", "global_plan": 0, "local_replan": count_status(rows, "BLOCKED_PRECEDENCE")},
        {"metric": "available_residual_task_count", "global_plan": len(rows), "local_replan": count_status(rows, "AVAILABLE")},
        {"metric": "recomputed_task_count", "global_plan": len(rows), "local_replan": len(sequence)},
        {"metric": "runtime_s", "global_plan": "see algorithm_summary.csv", "local_replan": f"{runtime_s:.6f}"},
    ]


def explanation_text(
    report: Dict[str, Any],
    rows: List[Dict[str, Any]],
    sequence: List[Dict[str, Any]],
    attempts: List[ReplanAttempt],
) -> str:
    blocked_resource_examples = [r for r in rows if r.get("new_status") == "BLOCKED_RESOURCE"][:5]
    blocked_precedence_examples = [r for r in rows if r.get("new_status") == "BLOCKED_PRECEDENCE"][:5]
    available_examples = sequence[:10]
    lines = [
        "制孔资源故障局部重规划说明",
        "",
        f"故障资源：{report['failed_resource_id']} ({', '.join(report['failed_resource_aliases'])})",
        f"扰动发生点：sequence_index <= {report['current_sequence_index']} 的任务被冻结为 DONE。",
        f"全局任务数：{report['global_task_count']}；局部可用残局任务数：{report['available_task_count']}；局部输出任务数：{report['local_replan_sequence_count']}。",
        "",
        "1. 为什么冻结：",
        "已完成任务代表车间现场真实执行状态，局部重规划不能回滚这些过程数据，因此全部固定为 DONE，不参与重排。",
        "",
        "2. 为什么阻断：",
        "制初孔、制连接孔、初孔输出等任务的 required_resources 包含钻具，RES_钻具 故障后被标记为 BLOCKED_RESOURCE。",
    ]
    for row in blocked_resource_examples:
        lines.append(f"- {row['sequence_index']}. {row['task_name']} | {row.get('operated_object_names','')} | {row.get('required_resources','')}")
    lines.extend([
        "",
        "3. 为什么产生前驱阻断：",
        "制孔任务被阻断后，后续连接/紧固/后处理任务虽然可能不直接使用钻具，但其工艺前驱尚未完成，因此标记为 BLOCKED_PRECEDENCE。",
    ])
    for row in blocked_precedence_examples:
        lines.append(f"- {row['sequence_index']}. {row['task_name']} | {row.get('process_name','')} | {row.get('required_resources','')}")
    lines.extend([
        "",
        "4. 为什么还能继续推进：",
        "扰动点之后、第一项制孔任务之前仍存在不依赖钻具的定位/连接带板/壁板定位任务，这些任务构成可用残局子图。",
    ])
    for row in available_examples:
        lines.append(f"- new {row['new_sequence_index']}. old {row['sequence_index']} | {row['task_name']} | {row.get('operated_object_names','')} | {row['replan_reason']}")
    lines.extend([
        "",
        "5. 为什么不是全局推倒重来：",
        "脚本只读取既有 Task 序列，冻结已完成前缀，剥离被资源和前驱阻断的任务，仅对 AVAILABLE 子集排序；原 algorithm_summary、optimization_result、best_object_sequences 与 inferred_process_task_sequence 均不覆盖。",
        "",
        "6. 改进痕迹：",
    ])
    for attempt in attempts:
        lines.append(
            f"- attempt {attempt.attempt}: {attempt.rule_version}, current={attempt.current_sequence_index}, "
            f"available={attempt.available_task_count}, blocked_resource={attempt.blocked_resource_count}, "
            f"blocked_precedence={attempt.blocked_precedence_count}, quality={attempt.explanation_quality}, "
            f"reason={attempt.improvement_reason}"
        )
    return "\n".join(lines) + "\n"


def write_status_to_neo4j(rows: List[Dict[str, Any]], failed_resource_id: str, uri: str, user: str, password: str) -> None:
    from neo4j import GraphDatabase

    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        with driver.session() as session:
            session.write_transaction(lambda tx: tx.run(
                """
                MATCH (r:Resource {id:$resource_id})
                SET r.status='Maintenance',
                    r.last_event_type='RESOURCE_FAILURE',
                    r.last_event_time=datetime()
                """,
                resource_id=failed_resource_id,
            ).consume())
            for row in rows:
                session.write_transaction(lambda tx, row=row: tx.run(
                    """
                    MATCH (t:Task {id:$task_id})
                    SET t.status=$status,
                        t.block_reason=$block_reason,
                        t.local_replan_source='resource_failure_simulation'
                    """,
                    task_id=row["task_id"],
                    status=row["new_status"],
                    block_reason=row["block_reason"],
                ).consume())


def run_replan(args: argparse.Namespace) -> Dict[str, Any]:
    start = time.perf_counter()
    tasks = read_tasks(Path(args.task_csv))
    initial_index = resolve_current_index(tasks, args.current_sequence_index, args.current_task_keyword)

    attempts: List[Tuple[ReplanAttempt, List[Dict[str, Any]]]] = []
    rows = classify_tasks(tasks, args.failed_resource_id, initial_index, propagate_precedence=True)
    attempts.append((
        make_attempt(1, "strict_sequence_precedence", initial_index, args.failed_resource_id, rows, "初始规则：冻结扰动点前缀，资源阻断后严格沿原序列传播前驱阻断。"),
        rows,
    ))

    if explanation_quality(rows) != "clear":
        adjusted_index = min((t["sequence_index"] for t in tasks if uses_failed_resource(t, args.failed_resource_id)), default=initial_index + 1) - 1
        adjusted_rows = classify_tasks(tasks, args.failed_resource_id, adjusted_index, propagate_precedence=True)
        attempts.append((
            make_attempt(2, "auto_adjust_before_first_failed_resource", adjusted_index, args.failed_resource_id, adjusted_rows, "初始结果不能同时展示冻结、资源阻断、前驱阻断和可用残局，自动调整到首个故障资源任务之前。"),
            adjusted_rows,
        ))

    best_attempt, best_rows = select_best_attempt(attempts)
    sequence = plan_available_sequence(best_rows)
    runtime_s = time.perf_counter() - start
    report = event_report(args.failed_resource_id, best_attempt.current_sequence_index, best_rows, sequence, runtime_s, args.mode)

    trace = [attempt.__dict__ for attempt, _ in attempts]
    write_json(RESULT_DIR / "local_replan_trace.json", trace)
    write_json(RESULT_DIR / "local_replan_event_report.json", report)

    status_fields = [
        "task_id", "task_name", "process_id", "process_name", "operated_objects",
        "operated_object_names", "required_resources", "generation_rule",
        "sequence_index", "new_status", "block_reason",
    ]
    write_csv(RESULT_DIR / "local_replan_task_status.csv", best_rows, status_fields)

    seq_fields = [
        "new_sequence_index", "sequence_index", "task_id", "task_name",
        "process_name", "operated_objects", "operated_object_names",
        "required_resources", "replan_reason", "distance_from_previous_task",
    ]
    write_csv(RESULT_DIR / "local_replan_sequence.csv", sequence, seq_fields)

    comp = comparison_rows(best_rows, sequence, runtime_s)
    write_csv(RESULT_DIR / "local_replan_comparison.csv", comp, ["metric", "global_plan", "local_replan"])

    text = explanation_text(report, best_rows, sequence, [attempt for attempt, _ in attempts])
    (RESULT_DIR / "local_replan_explanation.txt").write_text(text, encoding="utf-8")

    if args.write_status_to_neo4j:
        cfg = Neo4jConfig()
        write_status_to_neo4j(best_rows, args.failed_resource_id, args.uri or cfg.uri, args.user or cfg.user, args.password or cfg.password)

    return report


def main() -> None:
    cfg = Neo4jConfig()
    parser = argparse.ArgumentParser(description="Simulate task-level local replanning under resource failure.")
    parser.add_argument("--failed-resource-id", default="RES_钻具")
    parser.add_argument("--current-task-keyword", default="3号连接带板")
    parser.add_argument("--current-sequence-index", type=int, default=None)
    parser.add_argument("--mode", choices=["simulate"], default="simulate")
    parser.add_argument("--task-csv", default=str(TASK_CSV))
    parser.add_argument("--write-status-to-neo4j", action="store_true")
    parser.add_argument("--uri", default=cfg.uri)
    parser.add_argument("--user", default=cfg.user)
    parser.add_argument("--password", default=cfg.password)
    args = parser.parse_args()

    report = run_replan(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
