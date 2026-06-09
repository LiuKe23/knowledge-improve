from __future__ import annotations

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon


RESULT_DIR = Path(r"F:\proV1.8\front_cabin_planning_results")
OUTPUT = RESULT_DIR / "local_replan_flowchart_paper_style.png"


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


BODY_FONT = 10.5
CAPTION_FONT = 10.5


def box(ax, center, size, text, kind="rect", fontsize=BODY_FONT):
    x, y = center
    w, h = size
    if kind == "start":
        patch = FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.18",
            linewidth=1.6,
            edgecolor="#333333",
            facecolor="white",
        )
        ax.add_patch(patch)
    elif kind == "io":
        slant = 0.18 * w
        patch = Polygon(
            [
                (x - w / 2 + slant, y + h / 2),
                (x + w / 2, y + h / 2),
                (x + w / 2 - slant, y - h / 2),
                (x - w / 2, y - h / 2),
            ],
            closed=True,
            linewidth=1.6,
            edgecolor="#333333",
            facecolor="white",
        )
        ax.add_patch(patch)
    elif kind == "diamond":
        patch = Polygon(
            [(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)],
            closed=True,
            linewidth=1.6,
            edgecolor="#333333",
            facecolor="white",
        )
        ax.add_patch(patch)
    else:
        patch = FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="square,pad=0.02",
            linewidth=1.6,
            edgecolor="#333333",
            facecolor="white",
        )
        ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, color="#333333", linespacing=1.3)


def arrow(ax, start, end, text=None, text_offset=(0, 0), connectionstyle="arc3,rad=0.0"):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.5,
        color="#333333",
        shrinkA=4,
        shrinkB=4,
        connectionstyle=connectionstyle,
    )
    ax.add_patch(patch)
    if text:
        mx = (start[0] + end[0]) / 2 + text_offset[0]
        my = (start[1] + end[1]) / 2 + text_offset[1]
        ax.text(mx, my, text, fontsize=BODY_FONT, ha="center", va="center", color="#333333")


def main() -> None:
    setup_font()
    fig, ax = plt.subplots(figsize=(7.4, 9.2), dpi=220)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis("off")

    left_x = 3.0
    right_x = 7.2
    bw, bh = 3.25, 0.82

    box(ax, (left_x, 13.2), (2.0, 0.58), "开始", "start")
    box(ax, (left_x, 12.25), (3.4, 0.76), "输入任务知识图谱\n和资源扰动事件", "io")
    box(ax, (left_x, 11.05), (3.45, bh), "冻结已完成任务\nTask.status = DONE")
    box(ax, (left_x, 9.85), (3.45, bh), "标记故障资源任务\nBLOCKED_RESOURCE")
    box(ax, (left_x, 8.65), (3.45, bh), "沿 precedes_task\n传播前驱阻断")
    box(ax, (left_x, 7.45), (3.45, bh), "提取 AVAILABLE 任务\n构建可用残局子图")
    box(ax, (left_x, 6.25), (3.45, bh), "筛选未满足前驱\n数量为 0 的任务")
    box(ax, (left_x, 5.05), (3.45, bh), "存储满足条件的任务\n至候选任务集合")
    box(ax, (left_x, 3.85), (3.45, bh), "计算候选任务\n局部执行优先级")

    box(ax, (right_x, 4.75), (3.25, bh), "选择优先级最高的任务")
    box(ax, (right_x, 6.0), (3.25, bh), "反馈已规划任务名称、\n操作对象和所需资源")
    box(ax, (right_x, 7.55), (3.35, 1.35), "可用残局子图中\n未规划任务\n是否只剩 1 个？", "diamond", fontsize=10)
    box(ax, (right_x, 9.25), (3.4, 0.72), "输出局部替代任务序列", "io")
    box(ax, (right_x, 10.35), (2.0, 0.58), "结束", "start")

    # Left vertical flow.
    arrow(ax, (left_x, 12.9), (left_x, 12.64))
    arrow(ax, (left_x, 11.86), (left_x, 11.47))
    arrow(ax, (left_x, 10.63), (left_x, 10.27))
    arrow(ax, (left_x, 9.43), (left_x, 9.07))
    arrow(ax, (left_x, 8.23), (left_x, 7.87))
    arrow(ax, (left_x, 7.03), (left_x, 6.67))
    arrow(ax, (left_x, 5.83), (left_x, 5.47))
    arrow(ax, (left_x, 4.63), (left_x, 4.27))

    # Right branch.
    arrow(ax, (left_x + 1.75, 3.85), (right_x - 1.65, 4.75))
    arrow(ax, (right_x, 5.17), (right_x, 5.58))
    arrow(ax, (right_x, 6.42), (right_x, 6.86))
    arrow(ax, (right_x, 8.23), (right_x, 8.86), "是", text_offset=(0.33, 0.02))
    arrow(ax, (right_x, 9.61), (right_x, 10.03))

    # No loop from diamond back to left filter.
    arrow(
        ax,
        (right_x - 1.68, 7.55),
        (left_x + 1.75, 6.25),
        "否",
        text_offset=(0.0, 0.25),
        connectionstyle="arc3,rad=0.18",
    )

    ax.text(
        5,
        0.65,
        "图 4-x  资源扰动下的任务级局部重规划流程",
        ha="center",
        va="center",
        fontsize=CAPTION_FONT,
        color="#333333",
    )
    fig.tight_layout(pad=0.3)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
