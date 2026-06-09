from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple


@dataclass(frozen=True)
class Neo4jConfig:
    uri: str = "bolt://localhost:7693"
    user: str = "neo4j"
    password: str = "123456"


@dataclass(frozen=True)
class KGConfig:
    project_dir: Path = Path(r"F:\proV1.8")
    result_dir: Path = Path(r"F:\proV1.8\front_cabin_kg_results")
    catproduct_path: Path = Path(r"F:\00041\5536C10000G26_B\5536C10000G26.CATProduct")
    rpl_pdf_path: Path = Path(r"F:\00041\5536C10000G26_B\RPL5536C10000G26.pdf")
    mpl_pdf_path: Path = Path(r"F:\00041\5536C10000G26_B\MPL5536C10000G26.pdf")
    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)
    allowed_labels: Set[str] = field(default_factory=lambda: {
        "Product", "SubAssembly", "Part", "Feature", "Process", "Resource", "Task",
    })
    allowed_relationships: Set[str] = field(default_factory=lambda: {
        "hasComponent", "connect", "mayInterfere", "constrainedBy", "requireProcess",
        "requiresResource", "precedes_def", "parallel_def", "hasIndividual",
        "precedes_task", "parallel_task", "operatesOn",
    })
    allowed_inputs: Tuple[Path, Path, Path] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "allowed_inputs",
            (self.catproduct_path.resolve(), self.rpl_pdf_path.resolve(), self.mpl_pdf_path.resolve()),
        )


PROCESS_TEXT = """
9.3 垂直安定面前缘舱组件装配站位

9.3.1 站位介绍

前缘舱包括辅助梁、约 7 个前缘舱肋和左右壁板。辅助梁采用钣金或机加成型，前缘舱肋由铝合金厚板整体机加制成。前缘舱壁板采用玻璃纤维增强复合材料面板的蜂窝夹芯结构，芯材高度约 15~25mm，其中左侧壁板可分块拆卸，用以维护和更换前缘舱内安装的高频天线耦合器、系统管路等。

9.3.2 装配流程

1）定位前缘组件。定位方式采用前缘辅助梁上定位孔和腹板面定位。
2）定位前缘舱肋。采用腹板面及定位孔定位。并在定位器上设前缘舱肋夹紧器，用于在定位时夹紧前缘舱肋。
3）手工定位角材。
4）定位两侧壁板。采用边缘定位挡块定位，并钻壁板与肋、前缘辅助梁上的初孔。
5）制角材与辅助梁连接孔、前缘舱壁板连接孔、支撑件与肋连接孔并安装紧固件，其余部分初孔输出。
6）利用吊具将前缘舱组件移至站位3。
7）下架使用补铆托架安装紧固件及部分支架、涂胶密封，清洗排故。
""".strip()


PROCESS_DEFS: List[Dict[str, str]] = [
    {"id": "PROC_定位前缘组件", "name": "定位前缘组件", "process_type": "定位"},
    {"id": "PROC_定位前缘舱肋", "name": "定位前缘舱肋", "process_type": "定位"},
    {"id": "PROC_手工定位角材", "name": "手工定位角材", "process_type": "定位"},
    {"id": "PROC_定位两侧壁板", "name": "定位两侧壁板", "process_type": "定位"},
    {"id": "PROC_制初孔", "name": "制初孔", "process_type": "制孔"},
    {"id": "PROC_制连接孔", "name": "制连接孔", "process_type": "制孔"},
    {"id": "PROC_安装紧固件", "name": "安装紧固件", "process_type": "连接"},
    {"id": "PROC_初孔输出", "name": "初孔输出", "process_type": "制孔"},
    {"id": "PROC_吊装移站", "name": "吊装移站", "process_type": "搬运"},
    {"id": "PROC_下架", "name": "下架", "process_type": "下架"},
    {"id": "PROC_补铆安装紧固件", "name": "补铆安装紧固件", "process_type": "连接"},
    {"id": "PROC_安装部分支架", "name": "安装部分支架", "process_type": "连接"},
    {"id": "PROC_涂胶密封", "name": "涂胶密封", "process_type": "密封"},
    {"id": "PROC_清洗排故", "name": "清洗排故", "process_type": "清洗"},
]

RESOURCE_DEFS: List[Dict[str, str]] = [
    {"id": "RES_定位器", "name": "定位器", "resource_type": "工装"},
    {"id": "RES_前缘舱肋夹紧器", "name": "前缘舱肋夹紧器", "resource_type": "工装"},
    {"id": "RES_边缘定位挡块", "name": "边缘定位挡块", "resource_type": "工装"},
    {"id": "RES_钻具", "name": "钻具", "resource_type": "工具"},
    {"id": "RES_铆接工具", "name": "铆接工具", "resource_type": "工具"},
    {"id": "RES_吊具", "name": "吊具", "resource_type": "设备"},
    {"id": "RES_补铆托架", "name": "补铆托架", "resource_type": "工装"},
    {"id": "RES_密封胶枪", "name": "密封胶枪", "resource_type": "工具"},
    {"id": "RES_清洗工具", "name": "清洗工具", "resource_type": "工具"},
]

PRECEDES_DEF = [
    ("PROC_定位前缘组件", "PROC_定位前缘舱肋"),
    ("PROC_定位前缘舱肋", "PROC_手工定位角材"),
    ("PROC_手工定位角材", "PROC_定位两侧壁板"),
    ("PROC_定位两侧壁板", "PROC_制初孔"),
    ("PROC_制初孔", "PROC_制连接孔"),
    ("PROC_制连接孔", "PROC_安装紧固件"),
    ("PROC_安装紧固件", "PROC_初孔输出"),
    ("PROC_初孔输出", "PROC_吊装移站"),
    ("PROC_吊装移站", "PROC_下架"),
    ("PROC_下架", "PROC_补铆安装紧固件"),
    ("PROC_下架", "PROC_安装部分支架"),
    ("PROC_补铆安装紧固件", "PROC_涂胶密封"),
    ("PROC_安装部分支架", "PROC_涂胶密封"),
    ("PROC_涂胶密封", "PROC_清洗排故"),
]

PARALLEL_DEF = [("PROC_补铆安装紧固件", "PROC_安装部分支架")]

REQUIRES_RESOURCE = [
    ("PROC_定位前缘组件", "RES_定位器"),
    ("PROC_定位前缘舱肋", "RES_定位器"),
    ("PROC_定位前缘舱肋", "RES_前缘舱肋夹紧器"),
    ("PROC_手工定位角材", "RES_定位器"),
    ("PROC_定位两侧壁板", "RES_边缘定位挡块"),
    ("PROC_制初孔", "RES_钻具"),
    ("PROC_制连接孔", "RES_钻具"),
    ("PROC_安装紧固件", "RES_铆接工具"),
    ("PROC_初孔输出", "RES_钻具"),
    ("PROC_吊装移站", "RES_吊具"),
    ("PROC_下架", "RES_吊具"),
    ("PROC_补铆安装紧固件", "RES_补铆托架"),
    ("PROC_补铆安装紧固件", "RES_铆接工具"),
    ("PROC_安装部分支架", "RES_铆接工具"),
    ("PROC_涂胶密封", "RES_密封胶枪"),
    ("PROC_清洗排故", "RES_清洗工具"),
]

PROCESS_FEATURE_CONSTRAINTS = [
    ("PROC_定位前缘组件", "Feature_前缘辅助梁定位孔", "前缘辅助梁定位孔"),
    ("PROC_定位前缘组件", "Feature_腹板面", "腹板面"),
    ("PROC_定位前缘舱肋", "Feature_腹板面", "腹板面"),
    ("PROC_定位前缘舱肋", "Feature_前缘舱肋定位孔", "定位孔"),
    ("PROC_定位两侧壁板", "Feature_边缘定位挡块", "边缘定位挡块"),
    ("PROC_定位两侧壁板", "Feature_壁板边缘定位面", "壁板边缘定位面"),
]
