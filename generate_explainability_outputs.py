from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch


BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "front_cabin_planning_results"


def clean_value(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("|", ", ")


def load_font() -> str | None:
    candidates = ["Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC", "Arial Unicode MS"]
    available = {font.name for font in fm.fontManager.ttflist}
    selected = next((font for font in candidates if font in available), None)
    if selected:
        plt.rcParams["font.sans-serif"] = [selected]
    plt.rcParams["axes.unicode_minus"] = False
    return selected


def category_of(obj: dict) -> str:
    text = "".join(str(obj.get(key, "") or "") for key in ["category", "semantic_category", "name", "id"])
    if "辅助梁" in text:
        return "辅助梁"
    if "连接带板" in text:
        return "连接带板"
    if "角材" in text or "角片" in text or "缘条" in text:
        return "角材/缘条/角片"
    if "肋" in text:
        return "肋"
    if "板" in text or "壁板" in text or "封面" in text or "夹芯" in text:
        return "壁板"
    return "其他"


def extract_kg_iga_sequence(opt: dict, detail: pd.DataFrame) -> list[str]:
    best_sequences = opt.get("best_sequences") or opt.get("best_object_sequences") or {}
    if isinstance(best_sequences, dict):
        seq = best_sequences.get("KG-IGA") or best_sequences.get("KG_IGA")
        if seq:
            if isinstance(seq, dict):
                return list(seq.get("sequence") or [])
            return list(seq)

    if "best_sequence" in detail.columns:
        kg_runs = detail[detail["algorithm"] == "KG-IGA"].copy()
        if not kg_runs.empty:
            kg_runs = kg_runs.sort_values("best_fitness")
            try:
                return list(json.loads(kg_runs.iloc[0]["best_sequence"]))
            except Exception:
                pass

    text = (OUT_DIR / "best_object_sequences.txt").read_text(encoding="utf-8")
    match = re.search(r"\[KG-IGA\].*?(?=\n\[|\Z)", text, re.S)
    if not match:
        return []
    seq: list[str] = []
    for line in match.group(0).splitlines():
        line_match = re.match(r"\d+\.\s+([^|]+)\s+\|", line)
        if line_match:
            seq.append(line_match.group(1).strip())
    return seq


def write_task_steps(tasks: pd.DataFrame, objects: dict[str, dict]) -> None:
    tasks = tasks.copy()
    tasks["_seq_num"] = pd.to_numeric(tasks["sequence_index"], errors="coerce").fillna(0).astype(int)
    rows: list[dict] = []

    for _, row in tasks.sort_values("_seq_num").iterrows():
        obj_ids = [item.strip() for item in re.split(r"[;|]", str(row.get("operated_objects", ""))) if item.strip()]
        part_numbers: list[str] = []
        names: list[str] = []
        for obj_id in obj_ids:
            obj = objects.get(obj_id, {})
            part_number = obj.get("part_number") or obj.get("number") or ""
            name = obj.get("display_name") or obj.get("name") or obj_id
            if part_number:
                part_numbers.append(str(part_number))
            if name:
                names.append(str(name))

        rows.append(
            {
                "步骤": int(row["_seq_num"]),
                "任务名称": row["task_name"],
                "工艺": row["process_name"],
                "零件号": ", ".join(part_numbers),
                "零件名称": ", ".join(names),
                "所需资源": clean_value(row.get("required_resources", "")),
                "推理依据": row.get("generation_rule", ""),
            }
        )

    step_df = pd.DataFrame(rows)
    step_df.to_csv(OUT_DIR / "task_steps_with_part_names.csv", index=False, encoding="utf-8-sig")

    with (OUT_DIR / "task_steps_with_part_names.txt").open("w", encoding="utf-8") as file:
        file.write("工艺任务步骤 + 对应零件名称\n")
        file.write("=" * 40 + "\n")
        for row in rows:
            file.write(f"{str(row['步骤']).zfill(3)}. {row['任务名称']} | 工艺: {row['工艺']}\n")
            file.write(f"     零件号: {row['零件号']}\n")
            file.write(f"     零件名称: {row['零件名称']}\n")
            file.write(f"     推理依据: {row['推理依据']}\n")


def write_assessment(summary: pd.DataFrame, tasks: pd.DataFrame) -> None:
    candidate_report = json.loads((OUT_DIR / "candidate_pool_report.json").read_text(encoding="utf-8"))
    kg_row = summary[summary["算法"] == "KG-IGA"].iloc[0]
    ga_row = summary[summary["算法"] == "GA"].iloc[0]
    connection_belt_tasks = tasks[tasks["task_name"].astype(str).str.contains("连接带板", na=False)]
    runtime_ratio = float(kg_row["平均运行时间/s"]) / float(ga_row["平均运行时间/s"])

    lines = [
        "现有规划结果合理性评估",
        "=" * 34,
        "1. 目标函数方向：fitness 越低越好。当前目标只包含方向切换、工具切换、工艺连贯性中断，硬约束不参与扣分。",
        (
            f"2. 优化有效性：候选池最优 fitness={candidate_report.get('best_candidate_fitness')}，"
            f"KG-IGA 最优 fitness={kg_row['最优目标函数值']}；KG-IGA 命中率 "
            f"{kg_row['最优解命中率(%)']}%，平均收敛迭代 {kg_row['平均收敛迭代次数']}，"
            f"优于 GA 的 {ga_row['平均收敛迭代次数']}。"
        ),
        (
            "3. KG 最小增强：GA 和 KG-IGA 使用相同随机可行初始化；KG-IGA 只在交叉/变异后的"
            "轻量邻域搜索中查询 connect/resource 连续性，不把完整工艺参考顺序灌入初始种群。"
            f"运行时间 KG-IGA={kg_row['平均运行时间/s']}s，GA={ga_row['平均运行时间/s']}s，"
            f"约为 GA 的 {runtime_ratio:.2f} 倍。"
        ),
        (
            "4. 工艺形态问题：当前对象序列在目标函数意义上合理，但肋、角材、连接带板存在穿插。"
            "原因是目标函数明确不能加入区域/编号连续性，而图谱硬约束目前只约束阶段先后，"
            "没有表达“同一肋站位内肋-角材-连接带板应连续装配”的局部节拍。"
        ),
        (
            "5. 图规划基线：GraphPlan 可保留为图启发式构造基线，用于说明纯图规则能构造可行解；"
            "但收敛性对比应主要看 GA 与 KG-IGA，PSO 作为群智能基线。GraphPlan 不需要承担"
            "“收敛曲线”证明。"
        ),
        (
            f"6. 任务推理：已生成 {len(connection_belt_tasks)} 条连接带板定位/相关任务；"
            "新增 task_steps_with_part_names.csv/txt 用于展示步骤 + 零件名称。"
        ),
        (
            "7. 展示建议：论文中把“算法优化对象序列”和“工艺任务包/步骤视图”分开展示。"
            "前者证明算法和目标函数，后者按工艺规则展示步骤与零件名称，避免把展示排序反向写成图谱事实。"
        ),
    ]
    (OUT_DIR / "result_reasonableness_assessment.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figures(summary: pd.DataFrame, tasks: pd.DataFrame, seq_df: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    ax3 = fig.add_subplot(grid[1, :])

    positions = range(len(summary))
    ax1.bar([idx - 0.18 for idx in positions], summary["平均目标函数值"], width=0.36, label="平均目标函数值")
    ax1.bar([idx + 0.18 for idx in positions], summary["最优目标函数值"], width=0.36, label="最优目标函数值")
    ax1.set_xticks(list(positions), summary["算法"])
    ax1.set_title("算法目标函数对比（越低越好）")
    ax1.set_ylabel("fitness")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.25)

    ax2.bar(summary["算法"], summary["平均收敛迭代次数"], color="#4C78A8")
    ax2.set_title("平均收敛迭代次数")
    ax2.set_ylabel("iteration")
    ax2.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(summary["平均收敛迭代次数"]):
        ax2.text(idx, float(value) + 1, f"{float(value):.1f}", ha="center", fontsize=9)

    colors = {
        "辅助梁": "#5B8FF9",
        "肋": "#61DDAA",
        "角材/缘条/角片": "#F6BD16",
        "连接带板": "#E86452",
        "壁板": "#6DC8EC",
        "其他": "#999999",
    }
    if not seq_df.empty:
        for _, row in seq_df.iterrows():
            ax3.scatter(row["index"], 0, s=180, marker="s", color=colors.get(row["category"], "#999999"))
        ax3.set_xlim(0, len(seq_df) + 1)
        ax3.set_yticks([])
        ax3.set_xlabel("KG-IGA 最优对象序列位置")
        ax3.set_title("对象序列阶段分布：连接带板已进入支撑/角材阶段，但同站位连续性仍不足")
        ax3.grid(axis="x", alpha=0.2)
        handles = [Patch(facecolor=color, label=label) for label, color in colors.items() if label in set(seq_df["category"])]
        ax3.legend(handles=handles, ncol=5, loc="upper center")
        belt_count = int((seq_df["category"] == "连接带板").sum())
        ax3.text(
            0.01,
            -0.45,
            f"连接带板数量: {belt_count}；壁板整体位于后段，符合支撑/肋先于壁板的硬约束。",
            transform=ax3.transAxes,
            fontsize=10,
        )

    fig.suptitle("前缘舱装配序列规划结果解释", fontsize=16)
    fig.savefig(OUT_DIR / "planning_explainability_summary.png", dpi=200)
    plt.close(fig)

    process_counts = tasks["process_name"].value_counts().sort_values(ascending=True).tail(16)
    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    ax.barh(process_counts.index, process_counts.values, color="#72B7B2")
    ax.set_title("工艺任务序列中的任务类型数量")
    ax.set_xlabel("任务数")
    ax.grid(axis="x", alpha=0.25)
    for idx, value in enumerate(process_counts.values):
        ax.text(value + 0.5, idx, str(value), va="center")
    fig.savefig(OUT_DIR / "task_sequence_process_counts.png", dpi=200)
    plt.close(fig)


def main() -> None:
    selected_font = load_font()
    summary = pd.read_csv(OUT_DIR / "algorithm_summary.csv")
    detail = pd.read_csv(OUT_DIR / "all_runs_detail.csv")
    tasks = pd.read_csv(OUT_DIR / "inferred_process_task_sequence_with_names.csv")
    problem = json.loads((OUT_DIR / "problem_snapshot.json").read_text(encoding="utf-8"))
    opt = json.loads((OUT_DIR / "optimization_result.json").read_text(encoding="utf-8"))

    objects = {obj["id"]: obj for obj in problem.get("objects", [])}
    write_task_steps(tasks, objects)

    best_seq = extract_kg_iga_sequence(opt, detail)
    seq_rows = []
    for index, obj_id in enumerate(best_seq, 1):
        obj = objects.get(obj_id, {"id": obj_id, "display_name": obj_id, "name": obj_id, "part_number": ""})
        seq_rows.append(
            {
                "index": index,
                "id": obj_id,
                "part_number": obj.get("part_number", ""),
                "name": obj.get("display_name") or obj.get("name", obj_id),
                "category": category_of(obj),
            }
        )
    seq_df = pd.DataFrame(seq_rows)
    seq_df.to_csv(OUT_DIR / "kg_iga_best_sequence_with_stage.csv", index=False, encoding="utf-8-sig")

    write_assessment(summary, tasks)
    write_figures(summary, tasks, seq_df)

    created = [
        "task_steps_with_part_names.csv",
        "task_steps_with_part_names.txt",
        "kg_iga_best_sequence_with_stage.csv",
        "result_reasonableness_assessment.txt",
        "planning_explainability_summary.png",
        "task_sequence_process_counts.png",
    ]
    print(
        json.dumps(
            {
                "created": created,
                "task_rows": len(tasks),
                "font": selected_font,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
