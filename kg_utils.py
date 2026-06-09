from __future__ import annotations

import csv
import json
import logging
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from kg_config import KGConfig


CONFIG = KGConfig()


def setup_logging(name: str) -> logging.Logger:
    CONFIG.result_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    file_handler = logging.FileHandler(CONFIG.result_dir / f"{name}.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    return logger


def ensure_allowed_input(path: Path) -> Path:
    resolved = path.resolve()
    if resolved not in CONFIG.allowed_inputs:
        raise ValueError(f"Input path is not whitelisted: {resolved}")
    return resolved


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def clean_com_value(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() == "none":
        return ""
    if "<bound method" in text or "<COMObject" in text:
        return ""
    return text


def json_dumps_safe(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return "{}"


def stable_id(prefix: str, *parts: Any) -> str:
    raw = "|".join(clean_com_value(p) for p in parts if clean_com_value(p))
    safe = re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]+", "_", raw).strip("_")
    if not safe:
        safe = "unknown"
    return f"{prefix}_{safe[:180]}"


@contextmanager
def neo4j_session() -> Iterator[Any]:
    try:
        from neo4j import GraphDatabase
    except Exception as exc:
        raise RuntimeError("neo4j Python driver is required. Install package 'neo4j' in the dgl environment.") from exc
    driver = GraphDatabase.driver(CONFIG.neo4j.uri, auth=(CONFIG.neo4j.user, CONFIG.neo4j.password))
    try:
        with driver.session() as session:
            yield session
    finally:
        driver.close()


def run_write(session: Any, query: str, **params: Any) -> Any:
    return session.write_transaction(lambda tx: tx.run(query, **params).consume())


def run_read(session: Any, query: str, **params: Any) -> List[Dict[str, Any]]:
    result = session.read_transaction(lambda tx: [dict(record) for record in tx.run(query, **params)])
    return result


def validate_label(label: str) -> None:
    if label not in CONFIG.allowed_labels:
        raise ValueError(f"Disallowed label: {label}")


def validate_rel(rel_type: str) -> None:
    if rel_type not in CONFIG.allowed_relationships:
        raise ValueError(f"Disallowed relationship type: {rel_type}")


def merge_node(session: Any, label: str, node_id: str, props: Dict[str, Any]) -> None:
    validate_label(label)
    clean_props = {k: v for k, v in props.items() if v is not None}
    clean_props["id"] = node_id
    run_write(session, f"MERGE (n:{label} {{id: $id}}) SET n += $props", id=node_id, props=clean_props)


def merge_relationship(
    session: Any,
    start_id: str,
    end_id: str,
    rel_type: str,
    props: Optional[Dict[str, Any]] = None,
    undirected: bool = False,
) -> None:
    validate_rel(rel_type)
    props = {k: v for k, v in (props or {}).items() if v is not None}
    if undirected and start_id > end_id:
        start_id, end_id = end_id, start_id
    run_write(
        session,
        f"""
        MATCH (a {{id: $start_id}})
        MATCH (b {{id: $end_id}})
        MERGE (a)-[r:{rel_type}]->(b)
        SET r += $props
        """,
        start_id=start_id,
        end_id=end_id,
        props=props,
    )


def read_result_json(name: str, default: Any) -> Any:
    path = CONFIG.result_dir / name
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(name: str) -> List[Dict[str, str]]:
    path = CONFIG.result_dir / name
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        return list(csv.DictReader(fp))
