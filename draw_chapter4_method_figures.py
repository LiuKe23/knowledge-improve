from __future__ import annotations

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon


RESULT_DIR = Path(r"F:\proV1.8\front_cabin_planning_results")
BODY_FONT = 10.5
CAPTION_FONT = 10.5
LINE_COLOR = "#333333"


def setup_font() -> None:
    candidates = [
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simsun.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            fm.fontManager.addfont(path)
            plt.rcParams["font.family"] = fm.FontProperties(fname=path).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


def add_box(ax, center, size, text, kind="rect", fontsize=BODY_FONT, lw=1.45):
    x, y = center
    w, h = size
    if kind == "round":
        patch = FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.16",
            linewidth=lw,
            edgecolor=LINE_COLOR,
            facecolor="white",
        )
    elif kind == "io":
        s = 0.16 * w
        patch = Polygon(
            [
                (x - w / 2 + s, y + h / 2),
                (x + w / 2, y + h / 2),
                (x + w / 2 - s, y - h / 2),
                (x - w / 2, y - h / 2),
            ],
            closed=True,
            linewidth=lw,
            edgecolor=LINE_COLOR,
            facecolor="white",
        )
    elif kind == "diamond":
        patch = Polygon(
            [(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)],
            closed=True,
            linewidth=lw,
            edgecolor=LINE_COLOR,
            facecolor="white",
        )
    else:
        patch = FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="square,pad=0.02",
            linewidth=lw,
            edgecolor=LINE_COLOR,
            facecolor="white",
        )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, color=LINE_COLOR, linespacing=1.25)


def add_arrow(ax, start, end, text=None, text_offset=(0, 0), rad=0.0, lw=1.3):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=lw,
        color=LINE_COLOR,
        shrinkA=3,
        shrinkB=3,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(patch)
    if text:
        ax.text(
            (start[0] + end[0]) / 2 + text_offset[0],
            (start[1] + end[1]) / 2 + text_offset[1],
            text,
            ha="center",
            va="center",
            fontsize=BODY_FONT,
            color=LINE_COLOR,
        )


def finish(fig, ax, caption: str, output_name: str) -> Path:
    ax.text(5, 0.45, caption, ha="center", va="center", fontsize=CAPTION_FONT, color=LINE_COLOR)
    ax.axis("off")
    fig.tight_layout(pad=0.25)
    output = RESULT_DIR / output_name
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def draw_overall_framework() -> Path:
    fig, ax = plt.subplots(figsize=(8.4, 8.4), dpi=240)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 11)
    ax.axis("off")

    # Main two-stage spine.
    add_box(ax, (5, 10.25), (3.2, 0.62), "开始", "round")
    add_box(ax, (5, 9.28), (4.1, 0.75), "输入 Neo4j 装配知识图谱", "io")

    add_box(ax, (2.5, 8.15), (3.3, 0.82), "装配对象读取\nPart / SubAssembly")
    add_box(ax, (5, 8.15), (3.3, 0.82), "硬约束构建\nsupport / hasComponent")
    add_box(ax, (7.5, 8.15), (3.3, 0.82), "目标关系读取\nconnect / mayInterfere\nOBB / Resource", fontsize=9.6)

    add_box(ax, (5, 6.95), (4.2, 0.82), "可行候选序列生成\n随机拓扑排序")
    add_box(ax, (5, 5.8), (4.2, 0.82), "KG-IGA 装配对象序列优化")
    add_box(ax, (5, 4.65), (4.2, 0.75), "输出最优装配对象序列", "io")

    add_box(ax, (2.55, 3.5), (3.4, 0.82), "工艺任务推理\nrequireProcess / connect")
    add_box(ax, (5, 3.5), (3.4, 0.82), "任务依赖生成\nprecedes_task / parallel_task")
    add_box(ax, (7.45, 3.5), (3.4, 0.82), "资源需求继承\nrequiresResource")

    add_box(ax, (5, 2.25), (4.2, 0.82), "Task 实例写回图谱\n输出任务序列与可解释结果")
    add_box(ax, (5, 1.16), (3.2, 0.62), "结束", "round")

    add_arrow(ax, (5, 9.93), (5, 9.68))
    add_arrow(ax, (5, 8.9), (2.5, 8.58), rad=0.08)
    add_arrow(ax, (5, 8.9), (5, 8.58))
    add_arrow(ax, (5, 8.9), (7.5, 8.58), rad=-0.08)
    add_arrow(ax, (2.5, 7.72), (4.1, 7.27), rad=-0.05)
    add_arrow(ax, (5, 7.72), (5, 7.39))
    add_arrow(ax, (7.5, 7.72), (5.9, 7.27), rad=0.05)
    add_arrow(ax, (5, 6.53), (5, 6.22))
    add_arrow(ax, (5, 5.38), (5, 5.03))
    add_arrow(ax, (5, 4.27), (2.55, 3.93), rad=0.05)
    add_arrow(ax, (5, 4.27), (5, 3.93))
    add_arrow(ax, (5, 4.27), (7.45, 3.93), rad=-0.05)
    add_arrow(ax, (2.55, 3.08), (4.05, 2.62), rad=-0.05)
    add_arrow(ax, (5, 3.08), (5, 2.67))
    add_arrow(ax, (7.45, 3.08), (5.95, 2.62), rad=0.05)
    add_arrow(ax, (5, 1.83), (5, 1.5))

    ax.text(1.0, 6.37, "第一阶段：\n装配对象序列优化", fontsize=BODY_FONT, ha="center", va="center")
    ax.text(1.0, 2.95, "第二阶段：\n工艺任务推理生成", fontsize=BODY_FONT, ha="center", va="center")
    return finish(fig, ax, "图 4-1  两阶段规划总体框架", "fig4_1_two_stage_framework.png")


def draw_kg_iga_flow() -> Path:
    fig, ax = plt.subplots(figsize=(7.6, 9.2), dpi=240)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")

    left_x, right_x = 3.2, 7.0
    add_box(ax, (left_x, 11.25), (2.3, 0.58), "开始", "round")
    add_box(ax, (left_x, 10.35), (3.7, 0.75), "输入规划问题\n对象、硬约束和目标关系", "io")
    add_box(ax, (left_x, 9.18), (3.7, 0.78), "随机可行拓扑初始化\n与 GA 保持一致")
    add_box(ax, (left_x, 8.05), (3.7, 0.78), "计算适应度\n方向 / 工具 / KG 连贯性")
    add_box(ax, (left_x, 6.92), (3.7, 0.78), "选择、交叉、变异\n生成新个体")
    add_box(ax, (left_x, 5.79), (3.7, 0.78), "硬约束修复\n保证拓扑可行")

    add_box(ax, (right_x, 5.79), (3.65, 0.78), "轻量知识局部搜索\nconnect + OBB")
    add_box(ax, (right_x, 6.92), (3.65, 0.78), "完整适应度接受准则\n保留更优邻域解")
    add_box(ax, (right_x, 8.05), (3.65, 0.78), "精英保留\n更新当前最优序列")
    add_box(ax, (right_x, 9.35), (3.25, 1.16), "是否达到\n最大迭代次数？", "diamond", fontsize=10)
    add_box(ax, (right_x, 10.78), (3.55, 0.72), "输出最优装配对象序列", "io")
    add_box(ax, (right_x, 11.6), (2.2, 0.55), "结束", "round")

    add_arrow(ax, (left_x, 10.96), (left_x, 10.74))
    add_arrow(ax, (left_x, 9.96), (left_x, 9.58))
    add_arrow(ax, (left_x, 8.78), (left_x, 8.45))
    add_arrow(ax, (left_x, 7.65), (left_x, 7.32))
    add_arrow(ax, (left_x, 6.52), (left_x, 6.19))
    add_arrow(ax, (left_x + 1.85, 5.79), (right_x - 1.82, 5.79))
    add_arrow(ax, (right_x, 6.19), (right_x, 6.52))
    add_arrow(ax, (right_x, 7.32), (right_x, 7.65))
    add_arrow(ax, (right_x, 8.45), (right_x, 8.77))
    add_arrow(ax, (right_x, 9.93), (right_x, 10.42), "是", text_offset=(0.32, 0.02))
    add_arrow(ax, (right_x, 11.14), (right_x, 11.32))
    add_arrow(ax, (right_x - 1.62, 9.35), (left_x + 1.85, 8.05), "否", text_offset=(-0.12, 0.26), rad=0.14)

    add_box(ax, (5.1, 3.85), (7.0, 0.86), "知识增强边界：KG 不提供完整初始顺序，\n仅在局部搜索中按需使用 connect 与 OBB 信息", fontsize=10)
    return finish(fig, ax, "图 4-2  KG-IGA 装配对象序列优化流程", "fig4_2_kg_iga_flow.png")


def draw_task_inference_rules() -> Path:
    fig, ax = plt.subplots(figsize=(8.8, 8.6), dpi=240)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 11)
    ax.axis("off")

    add_box(ax, (5, 10.25), (4.0, 0.72), "输入最优装配对象序列", "io")
    add_box(ax, (2.3, 8.95), (3.4, 0.86), "R1 对象工艺实例化\nObject + requireProcess")
    add_box(ax, (5, 8.95), (3.4, 0.86), "R2 连接类任务生成\nconnect + 双端工艺需求")
    add_box(ax, (7.7, 8.95), (3.4, 0.86), "R3 资源需求继承\nProcess + requiresResource")

    add_box(ax, (2.3, 7.52), (3.4, 0.9), "定位 / 装配准备任务\n单对象 operatesOn")
    add_box(ax, (5, 7.52), (3.4, 0.9), "制孔 / 紧固任务\n双对象 operatesOn")
    add_box(ax, (7.7, 7.52), (3.4, 0.9), "任务所需工具、工装\n和设备资源")

    add_box(ax, (3.6, 6.05), (3.6, 0.86), "R4 任务先后关系\nprecedes_def -> precedes_task")
    add_box(ax, (6.4, 6.05), (3.6, 0.86), "R5 任务并行关系\nparallel_def -> parallel_task")

    add_box(ax, (5, 4.65), (4.15, 0.82), "R6 产品级任务生成\nProduct final process")
    add_box(ax, (5, 3.45), (4.3, 0.86), "生成 Task 实例\nid / name / objects / resources / rule")
    add_box(ax, (5, 2.22), (4.3, 0.82), "写回任务图谱\nhasIndividual / operatesOn / precedes_task")
    add_box(ax, (5, 1.05), (4.15, 0.72), "输出工艺任务序列与推理依据", "io")

    add_arrow(ax, (5, 9.89), (2.3, 9.38), rad=0.05)
    add_arrow(ax, (5, 9.89), (5, 9.38))
    add_arrow(ax, (5, 9.89), (7.7, 9.38), rad=-0.05)
    add_arrow(ax, (2.3, 8.52), (2.3, 7.97))
    add_arrow(ax, (5, 8.52), (5, 7.97))
    add_arrow(ax, (7.7, 8.52), (7.7, 7.97))
    add_arrow(ax, (2.3, 7.06), (3.6, 6.48), rad=-0.05)
    add_arrow(ax, (5, 7.06), (3.9, 6.48), rad=0.05)
    add_arrow(ax, (5, 7.06), (6.1, 6.48), rad=-0.05)
    add_arrow(ax, (7.7, 7.06), (6.4, 6.48), rad=0.05)
    add_arrow(ax, (3.6, 5.62), (4.65, 5.05), rad=-0.03)
    add_arrow(ax, (6.4, 5.62), (5.35, 5.05), rad=0.03)
    add_arrow(ax, (5, 4.24), (5, 3.88))
    add_arrow(ax, (5, 3.01), (5, 2.63))
    add_arrow(ax, (5, 1.81), (5, 1.42))

    ax.text(
        5,
        10.85,
        "连接类任务必须同时满足 connect 与双端 requireProcess，避免仅凭连接关系过度生成任务",
        ha="center",
        va="center",
        fontsize=10,
        color=LINE_COLOR,
    )
    return finish(fig, ax, "图 4-5  工艺任务推理规则示意图", "fig4_5_task_inference_rules.png")


def main() -> None:
    setup_font()
    outputs = [
        draw_overall_framework(),
        draw_kg_iga_flow(),
        draw_task_inference_rules(),
    ]
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
