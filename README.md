# Front Cabin Knowledge Graph and Assembly Planning

本仓库整理自本地项目 `F:\proV1.8`，用于前缘舱装配知识图谱构建、装配序列规划、算法对比和论文图表生成。

## 内容

- `*_kg.py`：从 CATIA/PDF/工艺规则构建 Neo4j 知识图谱。
- `load_planning_problem.py`：从 Neo4j 或快照读取装配规划问题。
- `candidate_sequence_generator.py`：生成可行候选装配序列。
- `optimization_algorithms.py`：GraphPlan、PSO、GA、KG-IGA、RL、DRL 等算法对比。
- `plan_front_cabin_sequence.py`：端到端规划入口。
- `infer_process_tasks.py`：由对象序列推断工艺任务序列。
- `local_replanner.py`：局部重规划实验。
- `draw_*.py`、`generate_explainability_outputs.py`：图表和可解释性输出。
- `front_cabin_kg_results/`：已生成的知识图谱构建报告。
- `front_cabin_planning_results/`：已生成的规划结果、收敛曲线、算法汇总、审计文件和论文图。

## 未纳入仓库的原始数据

原始数据不推送到 GitHub。完整重建知识图谱时，需要在本地准备以下类型文件：

- CATIA 装配模型：`.CATProduct`
- RPL/MPL 工艺或物料 PDF：`.pdf`
- CATIA Contact/干涉接触导出：`.xml`
- 可选的 CATIA 临时图片、数据库导出、几何分析中间文件

默认本地路径来自 `kg_config.py`：

```text
F:\00041\5536C10000G26_B\5536C10000G26.CATProduct
F:\00041\5536C10000G26_B\RPL5536C10000G26.pdf
F:\00041\5536C10000G26_B\MPL5536C10000G26.pdf
F:\proV1.8\1\CATTemp\tempFile.xml
```

也可以通过环境变量覆盖：

```powershell
$env:FRONT_CABIN_CATPRODUCT="D:\data\5536C10000G26.CATProduct"
$env:FRONT_CABIN_RPL_PDF="D:\data\RPL5536C10000G26.pdf"
$env:FRONT_CABIN_MPL_PDF="D:\data\MPL5536C10000G26.pdf"
$env:FRONT_CABIN_PROJECT_DIR="F:\proV1.8"
$env:FRONT_CABIN_KG_RESULT_DIR="F:\proV1.8\front_cabin_kg_results"
$env:FRONT_CABIN_RESULT_DIR="F:\proV1.8\front_cabin_planning_results"
```

## 环境

建议 Python 3.10。安装依赖：

```powershell
pip install -r requirements.txt
```

Neo4j 默认连接：

```text
bolt://localhost:7693
user: neo4j
password: 123456
```

可通过环境变量覆盖：

```powershell
$env:NEO4J_URI="bolt://localhost:7693"
$env:NEO4J_USER="neo4j"
$env:NEO4J_PASSWORD="123456"
```

## 直接复现实验图和结果

仓库已包含 `front_cabin_planning_results/problem_snapshot*.json`，因此可不连接 Neo4j，直接从快照复现实验。

复现 KG-IGA、RL、DRL 快速对比：

```powershell
python optimization_algorithms.py --snapshot front_cabin_planning_results\problem_snapshot.json --runs 3 --population-size 20 --iterations 20 --rl-vs-kgiga-only
```

复现完整算法对比：

```powershell
python optimization_algorithms.py --snapshot front_cabin_planning_results\problem_snapshot.json --runs 20 --population-size 40 --iterations 80
```

主要输出：

```text
front_cabin_planning_results\algorithm_summary*.csv
front_cabin_planning_results\all_runs_detail*.csv
front_cabin_planning_results\optimization_result*.json
front_cabin_planning_results\convergence_curve*.png
front_cabin_planning_results\best_object_sequences*.txt
```

## 完整重建流程

完整重建需要本地原始数据、CATIA COM 环境和 Neo4j：

```powershell
python run_build_kg.py --clear-graph
python plan_front_cabin_sequence.py --problem-snapshot front_cabin_planning_results\problem_snapshot.json
```

如果要从 Neo4j 重新抓取规划问题：

```powershell
python load_planning_problem.py --output front_cabin_planning_results\problem_snapshot.json
python plan_front_cabin_sequence.py --problem-snapshot front_cabin_planning_results\problem_snapshot.json
```

## 注意

- `1/` 目录是本地 CATIA/Contact 中间数据目录，已被 `.gitignore` 排除。
- 仓库中的结果文件保留历史实验输出，部分 JSON/日志中可能记录本地证据文件路径，但不包含原始 CATIA/PDF 数据本体。
- `fitness` 越低代表结果越优。
