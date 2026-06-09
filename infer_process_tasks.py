from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from kg_config import Neo4jConfig
from load_planning_problem import RESULT_DIR, load_problem


PROCESS_ORDER = [
    "定位前缘组件", "定位前缘舱肋", "手工定位角材", "定位两侧壁板",
    "制初孔", "制连接孔", "安装紧固件", "初孔输出", "吊装移站", "下架",
    "补铆安装紧固件", "安装部分支架", "涂胶密封", "清洗排故",
]


def safe_token(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]+", "_", text).strip("_")[:120] or "unknown"


def process_rank(process_name: str) -> int:
    for idx, name in enumerate(PROCESS_ORDER):
        if process_name == name:
            return idx
    for idx, name in sorted(enumerate(PROCESS_ORDER), key=lambda item: len(item[1]), reverse=True):
        if name in process_name:
            return idx
    for idx, name in enumerate(PROCESS_ORDER):
        if name in process_name:
            return idx
    return len(PROCESS_ORDER)


def has_process(problem: Dict[str, Any], oid: str, process_keyword: str) -> Dict[str, Any] | None:
    for proc in problem.get("processes_by_object", {}).get(oid, []):
        if process_keyword in proc["id"] or process_keyword in proc["name"]:
            return proc
    return None


def pair_key(a: str, b: str) -> str:
    return "||".join(sorted((a, b)))


def task_row(task_id: str, task_name: str, proc: Dict[str, Any], objects: Sequence[str], rule: str) -> Dict[str, Any]:
    return {
        "task_id": task_id,
        "task_name": task_name,
        "process_id": proc.get("id") or proc.get("process_id") or "",
        "process_name": proc.get("name") or proc.get("process_name") or "",
        "operated_objects": ";".join(objects),
        "required_resources": ";".join(proc.get("resource_names") or proc.get("resource_ids") or []),
        "generation_rule": rule,
        "sequence_index": 0,
    }


def task_center(objects: Sequence[str], centers: Dict[str, List[float]]) -> List[float] | None:
    values = [centers[oid] for oid in objects if oid in centers and len(centers[oid]) == 3]
    if not values:
        return None
    return [round(sum(point[i] for point in values) / len(values), 9) for i in range(3)]


def object_name_map(problem: Dict[str, Any]) -> Dict[str, str]:
    names = {}
    for obj in problem.get("objects", []):
        label = obj.get("display_name") or obj.get("name") or obj.get("id") or ""
        part_number = obj.get("part_number") or obj.get("name") or ""
        names[obj["id"]] = f"{part_number} {label}".strip()
    for proc in problem.get("product_processes", []):
        product_id = proc.get("product_id")
        if product_id:
            names[product_id] = f"{proc.get('product_name') or product_id}"
    return names


def explain_rule(rule: str) -> str:
    explanations = {
        "object_requireProcess_positioning": "对象存在 requireProcess 定位类工艺，因此生成定位任务。",
        "object_requireProcess_connection_belt_positioning": "连接带板属于支撑件，随角材/支撑件定位阶段定位，因此生成定位连接带板任务。",
        "object_requireProcess_bracket_install": "对象存在 PROC_安装部分支架，因此生成支架安装任务。",
        "connect_and_both_require_initial_hole": "两端对象存在 connect，且两端都 requireProcess 制初孔，因此生成制初孔任务。",
        "connect_and_both_require_connection_hole": "两端对象存在 connect，且两端都 requireProcess 制连接孔，因此生成制连接孔任务。",
        "connect_and_both_require_fastener_install": "两端对象存在 connect，且两端都 requireProcess 安装紧固件，因此生成紧固任务。",
        "product_final_process": "最终 Product 节点 requireProcess 后处理工艺，因此生成产品后处理任务。",
    }
    return explanations.get(rule, rule)


def task_explainability_rows(problem: Dict[str, Any], tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    names = object_name_map(problem)
    rows = []
    for task in tasks:
        object_ids = [oid for oid in task.get("operated_objects", "").split(";") if oid]
        rows.append({
            **task,
            "operated_object_names": ";".join(names.get(oid, oid) for oid in object_ids),
            "task_obb_center": ";".join(str(v) for v in task.get("task_obb_center", [])),
            "inference_explanation": explain_rule(task.get("generation_rule", "")),
        })
    return rows


def infer_tasks(problem: Dict[str, Any], object_sequence: Sequence[str]) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    object_pos = {oid: i for i, oid in enumerate(object_sequence)}
    objects_by_id = {obj["id"]: obj for obj in problem.get("objects", [])}

    for oid in object_sequence:
        for proc in problem.get("processes_by_object", {}).get(oid, []):
            pname = proc["name"]
            if "定位" in pname:
                obj = objects_by_id.get(oid, {})
                if obj.get("category") == "连接带板":
                    task_name = f"定位连接带板_{oid}"
                    tid = f"Task_定位连接带板_{safe_token(oid)}"
                    rule = "object_requireProcess_connection_belt_positioning"
                else:
                    task_name = f"{pname}_{oid}"
                    tid = f"Task_{safe_token(pname)}_{safe_token(oid)}"
                    rule = "object_requireProcess_positioning"
                tasks.append(task_row(tid, task_name, proc, [oid], rule))
            if "安装部分支架" in pname:
                tid = f"Task_安装部分支架_{safe_token(oid)}"
                tasks.append(task_row(tid, f"安装部分支架_{oid}", proc, [oid], "object_requireProcess_bracket_install"))

    connect_pairs = []
    seen = set()
    for edge in problem.get("connect_edges", []):
        a, b = edge["a"], edge["b"]
        if a in object_pos and b in object_pos:
            key = pair_key(a, b)
            if key not in seen:
                seen.add(key)
                connect_pairs.append((a, b))
    connect_pairs.sort(key=lambda p: (min(object_pos[p[0]], object_pos[p[1]]), max(object_pos[p[0]], object_pos[p[1]])))

    for a, b in connect_pairs:
        for keyword, prefix, rule in [
            ("制初孔", "Task_制初孔", "connect_and_both_require_initial_hole"),
            ("制连接孔", "Task_制连接孔", "connect_and_both_require_connection_hole"),
            ("安装紧固件", "Task_安装紧固件", "connect_and_both_require_fastener_install"),
        ]:
            proc_a = has_process(problem, a, keyword)
            proc_b = has_process(problem, b, keyword)
            if proc_a and proc_b:
                tid = f"{prefix}_{safe_token(a)}_{safe_token(b)}"
                tasks.append(task_row(tid, f"{prefix.replace('Task_', '')}_{a}_{b}", proc_a, [a, b], rule))

    final_processes = sorted(problem.get("product_processes", []), key=lambda p: process_rank(p.get("process_name") or ""))
    for proc in final_processes:
        pname = proc.get("process_name") or ""
        product_id = proc.get("product_id") or "Product"
        if "吊装移站" in pname:
            name = f"吊装移站_{product_id}"
        elif "下架" in pname:
            name = f"下架_{product_id}"
        elif "补铆" in pname:
            name = f"补铆安装紧固件_{product_id}"
        elif "涂胶" in pname or "密封" in pname:
            name = f"涂胶密封_{product_id}"
        elif "清洗" in pname or "排故" in pname:
            name = f"清洗排故_{product_id}"
        else:
            name = f"{pname}_{product_id}"
        tasks.append(task_row(f"Task_{safe_token(name)}", name, {
            "id": proc.get("process_id", ""),
            "name": pname,
            "resource_ids": proc.get("resource_ids", []),
            "resource_names": proc.get("resource_names", []),
        }, [product_id], "product_final_process"))

    tasks.sort(key=lambda row: (process_rank(row["process_name"]), min(object_pos.get(o, 10**6) for o in row["operated_objects"].split(";"))))
    names = object_name_map(problem)
    for idx, row in enumerate(tasks, start=1):
        row["sequence_index"] = idx
        object_ids = [oid for oid in row["operated_objects"].split(";") if oid]
        row["operated_object_names"] = ";".join(names.get(oid, oid) for oid in object_ids)
        row["task_obb_center"] = task_center(object_ids, problem.get("object_centers", {})) or []
    return tasks


def write_tasks_csv(tasks: List[Dict[str, Any]], path: Path) -> None:
    fields = ["task_id", "task_name", "process_id", "process_name", "operated_objects", "operated_object_names", "required_resources", "generation_rule", "sequence_index", "task_obb_center"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for row in tasks:
            writer.writerow({key: row.get(key, "") for key in fields})


def write_task_explainability(problem: Dict[str, Any], tasks: List[Dict[str, Any]], csv_path: Path, txt_path: Path, json_path: Path) -> None:
    rows = task_explainability_rows(problem, tasks)
    fields = [
        "sequence_index", "task_id", "task_name", "process_id", "process_name",
        "operated_objects", "operated_object_names", "task_obb_center", "required_resources",
        "generation_rule", "inference_explanation",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})
    lines = []
    for row in rows:
        lines.append(
            f"{row['sequence_index']:03d}. {row['task_name']} | 零件: {row['operated_object_names']} | 推理: {row['inference_explanation']}"
        )
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def write_tasks_txt(tasks: List[Dict[str, Any]], path: Path) -> None:
    lines = [f"{t['sequence_index']:03d}. {t['task_name']} | {t['process_name']} | {t['operated_objects']}" for t in tasks]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_tasks_to_neo4j(tasks: List[Dict[str, Any]], uri: str, user: str, password: str) -> None:
    from neo4j import GraphDatabase

    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        with driver.session() as session:
            session.write_transaction(lambda tx: tx.run("""
            MATCH (t:Task {source:'inferred_from_object_sequence'})
            DETACH DELETE t
            """).consume())
            for task in tasks:
                session.write_transaction(lambda tx, task=task: tx.run("""
                MERGE (t:Task {id:$id})
                SET t.name=$name, t.process_id=$process_id, t.process_name=$process_name,
                    t.sequence_index=$sequence_index, t.generation_rule=$generation_rule,
                    t.task_obb_center=$task_obb_center,
                    t.status='PENDING', t.progress=0.0,
                    t.source='inferred_from_object_sequence'
                WITH t
                MATCH (p:Process {id:$process_id})
                MERGE (p)-[:hasIndividual]->(t)
                """, id=task["task_id"], name=task["task_name"], process_id=task["process_id"],
                process_name=task["process_name"], sequence_index=task["sequence_index"],
                generation_rule=task["generation_rule"], task_obb_center=task.get("task_obb_center", [])).consume())
                for oid in task["operated_objects"].split(";"):
                    session.write_transaction(lambda tx, task=task, oid=oid: tx.run("""
                    MATCH (t:Task {id:$task_id})
                    MATCH (o {id:$oid})
                    MERGE (t)-[:operatesOn]->(o)
                    """, task_id=task["task_id"], oid=oid).consume())
            for a, b in zip(tasks, tasks[1:]):
                session.write_transaction(lambda tx, a=a, b=b: tx.run("""
                MATCH (ta:Task {id:$a})
                MATCH (tb:Task {id:$b})
                MERGE (ta)-[:precedes_task {source:'sequence_index'}]->(tb)
                """, a=a["task_id"], b=b["task_id"]).consume())
            by_proc = {}
            for task in tasks:
                by_proc.setdefault(task["process_id"], []).append(task)
            parallels = session.read_transaction(lambda tx: [dict(r) for r in tx.run("""
            MATCH (a:Process)-[:parallel_def]-(b:Process)
            WHERE a.id < b.id
            RETURN a.id AS a, b.id AS b
            """)])
            for rel in parallels:
                for ta in by_proc.get(rel["a"], []):
                    for tb in by_proc.get(rel["b"], []):
                        session.write_transaction(lambda tx, ta=ta, tb=tb: tx.run("""
                        MATCH (a:Task {id:$a})
                        MATCH (b:Task {id:$b})
                        MERGE (a)-[:parallel_task {source:'parallel_def'}]->(b)
                        """, a=ta["task_id"], b=tb["task_id"]).consume())


def load_best_sequence(algorithm: str) -> List[str]:
    data = json.loads((RESULT_DIR / "optimization_result.json").read_text(encoding="utf-8"))
    return data["best_sequences"][algorithm]["sequence"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", default=str(RESULT_DIR / "problem_snapshot.json"))
    parser.add_argument("--algorithm", default="KG-IGA")
    parser.add_argument("--write-tasks-to-neo4j", action="store_true")
    cfg = Neo4jConfig()
    parser.add_argument("--uri", default=cfg.uri)
    parser.add_argument("--user", default=cfg.user)
    parser.add_argument("--password", default=cfg.password)
    args = parser.parse_args()

    problem = load_problem(Path(args.snapshot))
    sequence = load_best_sequence(args.algorithm)
    tasks = infer_tasks(problem, sequence)
    write_tasks_csv(tasks, RESULT_DIR / "inferred_process_task_sequence.csv")
    write_tasks_txt(tasks, RESULT_DIR / "inferred_process_task_sequence.txt")
    write_task_explainability(
        problem,
        tasks,
        RESULT_DIR / "inferred_process_task_sequence_with_names.csv",
        RESULT_DIR / "task_inference_explanation.txt",
        RESULT_DIR / "task_inference_explanation.json",
    )
    if args.write_tasks_to_neo4j:
        write_tasks_to_neo4j(tasks, args.uri, args.user, args.password)
    print(f"inferred tasks: {len(tasks)}")


if __name__ == "__main__":
    main()
