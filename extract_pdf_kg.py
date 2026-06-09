from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from kg_config import KGConfig
from kg_utils import ensure_allowed_input, merge_node, neo4j_session, run_read, run_write, setup_logging, stable_id, write_csv, write_json


CONFIG = KGConfig()
LOGGER = setup_logging("extract_pdf_kg")

PART_NO_RE = re.compile(r"\b(?:5536C|MS|NAS|BAC|ABS|HL|CF[A-Z]{2})[0-9A-Z\-]+(?:G\d+)?\b")
STANDARD_KEYWORDS = (
    "铆钉", "螺栓", "螺钉", "螺母", "垫圈", "托板螺母", "紧固件",
    "rivet", "bolt", "screw", "nut", "washer", "fastener",
    "NAS", "MS", "BAC", "ABS", "HL", "CFBL",
)
STRUCTURAL_KEYWORDS = (
    "辅助梁", "前缘舱肋", "肋", "肋角材", "角材", "角片", "缘条", "壁板", "蒙皮",
    "封面板", "夹芯板", "连接带板", "支撑件", "支架",
    "beam", "rib", "angle", "stringer", "skin", "panel", "support", "bracket",
)
RESOURCE_WORDS = ("定位器", "夹紧器", "定位挡块", "钻具", "铆接工具", "吊具", "补铆托架", "密封胶枪", "清洗工具", "工装", "工具")
PROCESS_WORDS = ("定位", "制孔", "钻", "安装", "紧固件", "吊装", "下架", "涂胶", "密封", "清洗", "排故", "铆")


def extract_text_with_pymupdf(path: Path) -> Tuple[str, Dict[str, Any]]:
    import fitz
    texts: List[str] = []
    meta = {"pages": 0, "empty_pages": 0}
    doc = fitz.open(str(path))
    meta["pages"] = doc.page_count
    for page in doc:
        text = page.get_text("text") or ""
        if not text.strip():
            meta["empty_pages"] += 1
        texts.append(text)
    return "\n".join(texts), meta


def extract_tables_with_pdfplumber(path: Path) -> Tuple[List[List[str]], List[str]]:
    warnings: List[str] = []
    rows: List[List[str]] = []
    try:
        import pdfplumber
    except Exception:
        warnings.append("pdfplumber not available; table extraction skipped.")
        return rows, warnings
    try:
        with pdfplumber.open(str(path)) as pdf:
            for page_index, page in enumerate(pdf.pages, start=1):
                try:
                    tables = page.extract_tables() or []
                    for table in tables:
                        for row in table:
                            rows.append([(cell or "").strip() for cell in row])
                except Exception as exc:
                    warnings.append(f"table extraction failed on page {page_index}: {exc}")
    except Exception as exc:
        warnings.append(f"pdfplumber failed to open {path.name}: {exc}")
    if not rows:
        warnings.append(f"No explicit tables extracted from {path.name}.")
    return rows, warnings


def is_standard_part_text(text: str) -> bool:
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in STANDARD_KEYWORDS)


def is_structural_text(text: str) -> bool:
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in STRUCTURAL_KEYWORDS)


def classify_object(text: str) -> str:
    lowered = text.lower()
    # Priority matters: 肋角材 should be angle, not rib.
    if "角材" in text or "角片" in text or "angle" in lowered:
        return "角材"
    if "缘条" in text or "stringer" in lowered:
        return "缘条"
    if "封面板" in text or "夹芯板" in text or "壁板" in text or "蒙皮" in text or "skin" in lowered or "panel" in lowered:
        return "壁板"
    if "连接带板" in text:
        return "连接带板"
    if "辅助梁" in text or "beam" in lowered:
        return "辅助梁"
    if "前缘舱肋" in text or "肋" in text or "rib" in lowered:
        return "前缘舱肋"
    if "支撑件" in text or "支架" in text or "support" in lowered or "bracket" in lowered:
        return "支撑件/支架"
    if is_standard_part_text(text):
        return "标准件/紧固件"
    return ""


def likely_name_cells(row: List[str], part_no: str) -> str:
    cells = []
    for cell in row:
        cell = cell.strip()
        if not cell or part_no in cell or re.fullmatch(r"\d+", cell):
            continue
        if PART_NO_RE.fullmatch(cell):
            continue
        cells.append(cell)
    return " ".join(cells[:3]).strip()


def parse_parts(text: str, table_rows: List[List[str]], source_name: str) -> List[Dict[str, str]]:
    parts: Dict[str, Dict[str, str]] = {}
    for row in table_rows:
        joined = " ".join(row)
        matches = PART_NO_RE.findall(joined)
        if not matches:
            continue
        part_no = matches[0]
        quantity = ""
        for cell in row:
            if re.fullmatch(r"\d+", cell.strip()):
                quantity = cell.strip()
                break
        name = likely_name_cells(row, part_no)
        evidence = joined[:500]
        parts.setdefault(part_no, {
            "part_number": part_no,
            "name_cn": name,
            "display_name": name or part_no,
            "quantity": quantity,
            "material": "",
            "object_category": classify_object(joined),
            "is_standard_part": str(is_standard_part_text(joined)),
            "is_structural_part": str(is_structural_text(joined)),
            "source_pdf": source_name,
            "evidence": evidence,
        })
    for line in text.splitlines():
        matches = PART_NO_RE.findall(line)
        for part_no in matches:
            parts.setdefault(part_no, {
                "part_number": part_no,
                "name_cn": "",
                "display_name": part_no,
                "quantity": "",
                "material": "",
                "object_category": classify_object(line),
                "is_standard_part": str(is_standard_part_text(line)),
                "is_structural_part": str(is_structural_text(line)),
                "source_pdf": source_name,
                "evidence": line[:500],
            })
    return list(parts.values())


def parse_mentions(text: str, words: Tuple[str, ...], source_name: str, kind: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    seen = set()
    for line in text.splitlines():
        compact = line.strip()
        if not compact:
            continue
        for word in words:
            if word in compact and (source_name, word, compact[:80]) not in seen:
                seen.add((source_name, word, compact[:80]))
                rows.append({"source_pdf": source_name, "mention": word, "kind": kind, "evidence": compact[:500]})
    return rows


def update_pdf_data_in_neo4j(parts: List[Dict[str, str]]) -> Dict[str, int]:
    stats = {
        "pdf_matched_to_catproduct_count": 0,
        "pdf_unmatched_structural_part_count": 0,
        "pdf_standard_part_count": 0,
        "pdf_unmatched_ignored_count": 0,
    }
    with neo4j_session() as session:
        for part in parts:
            part_no = part["part_number"]
            is_standard = part["is_standard_part"] == "True" or is_standard_part_text(f"{part_no} {part.get('name_cn', '')}")
            if is_standard:
                stats["pdf_standard_part_count"] += 1
            matched = run_read(session, """
            MATCH (n)
            WHERE (n:Product OR n:SubAssembly OR n:Part) AND n.part_number = $part_number
            RETURN n.id AS id LIMIT 1
            """, part_number=part_no)
            props = {
                "display_name": part.get("display_name") or part_no,
                "name_cn": part.get("name_cn", ""),
                "object_category": part.get("object_category", "") or classify_object(f"{part.get('display_name', '')} {part.get('name_cn', '')}"),
                "pdf_quantity": part.get("quantity", ""),
                "pdf_evidence": part.get("evidence", ""),
                "is_standard_part": True if is_standard else None,
                "is_assembly_object": True if not is_standard else None,
            }
            if matched:
                run_write(session, """
                MATCH (n {id: $id})
                SET n.display_name = CASE WHEN $display_name <> '' THEN $display_name ELSE coalesce(n.display_name, n.name) END,
                    n.name_cn = CASE WHEN $name_cn <> '' THEN $name_cn ELSE coalesce(n.name_cn, '') END,
                    n.object_category = CASE WHEN $object_category <> '' THEN $object_category ELSE coalesce(n.object_category, '') END,
                    n.pdf_quantity = $pdf_quantity,
                    n.pdf_evidence = $pdf_evidence
                FOREACH (_ IN CASE WHEN $is_standard_part = true THEN [1] ELSE [] END |
                    SET n.is_standard_part = true REMOVE n.is_assembly_object)
                FOREACH (_ IN CASE WHEN $is_assembly_object = true THEN [1] ELSE [] END |
                    SET n.is_assembly_object = true)
                """, id=matched[0]["id"], **props)
                stats["pdf_matched_to_catproduct_count"] += 1
                continue
            if is_standard:
                # Standard parts not present in CATProduct are inventory/consumption evidence, not assembly objects.
                stats["pdf_unmatched_ignored_count"] += 1
                continue
            if part["is_structural_part"] == "True" and part.get("object_category"):
                node_id = stable_id("PDFPart", part_no)
                merge_node(session, "Part", node_id, {
                    "id": node_id,
                    "name": part_no,
                    "part_number": part_no,
                    "display_name": part.get("display_name") or part_no,
                    "name_cn": part.get("name_cn", ""),
                    "object_category": part.get("object_category", ""),
                    "node_type": "Part",
                    "source": "PDF_unmatched_structural_part",
                    "source_document": part.get("source_pdf", ""),
                    "is_assembly_object": True,
                    "pdf_quantity": part.get("quantity", ""),
                    "pdf_evidence": part.get("evidence", ""),
                })
                stats["pdf_unmatched_structural_part_count"] += 1
            else:
                stats["pdf_unmatched_ignored_count"] += 1
    return stats


def main() -> None:
    report: Dict[str, Any] = {"inputs": [], "warnings": []}
    all_parts: List[Dict[str, str]] = []
    all_process_mentions: List[Dict[str, str]] = []
    all_resource_mentions: List[Dict[str, str]] = []
    for pdf_path in (CONFIG.rpl_pdf_path, CONFIG.mpl_pdf_path):
        path = ensure_allowed_input(pdf_path)
        LOGGER.info("Extracting allowed PDF as supplementary source: %s", path)
        source_name = path.name
        try:
            text, meta = extract_text_with_pymupdf(path)
        except Exception as exc:
            report["warnings"].append(f"{source_name}: PyMuPDF extraction failed: {exc}")
            text, meta = "", {"pages": 0, "empty_pages": 0}
        tables, warnings = extract_tables_with_pdfplumber(path)
        report["warnings"].extend([f"{source_name}: {warning}" for warning in warnings])
        if meta.get("pages") and meta.get("empty_pages") == meta.get("pages"):
            report["warnings"].append(f"{source_name}: appears to be scanned or textless; OCR was not used.")
        all_parts.extend(parse_parts(text, tables, source_name))
        all_process_mentions.extend(parse_mentions(text, PROCESS_WORDS, source_name, "process"))
        all_resource_mentions.extend(parse_mentions(text, RESOURCE_WORDS, source_name, "resource"))
        report["inputs"].append({"path": str(path), **meta, "table_rows": len(tables), "text_chars": len(text)})
    neo4j_stats = update_pdf_data_in_neo4j(all_parts)
    report.update({
        "parts": len(all_parts),
        "connections_candidates": 0,
        "process_mentions": len(all_process_mentions),
        "resource_mentions": len(all_resource_mentions),
        "neo4j": neo4j_stats,
        "pdf_matched_to_catproduct_count": neo4j_stats["pdf_matched_to_catproduct_count"],
        "pdf_unmatched_structural_part_count": neo4j_stats["pdf_unmatched_structural_part_count"],
        "pdf_standard_part_count": neo4j_stats["pdf_standard_part_count"],
        "connect_note": "PDF no longer creates connect relationships; connect is reserved for shared standard part evidence.",
    })
    write_csv(CONFIG.result_dir / "pdf_parts.csv", all_parts, [
        "part_number", "display_name", "name_cn", "quantity", "material", "object_category",
        "is_standard_part", "is_structural_part", "source_pdf", "evidence",
    ])
    write_csv(CONFIG.result_dir / "pdf_connections.csv", [], ["object_a", "object_b", "connect_type", "confidence", "source_pdf", "evidence"])
    write_csv(CONFIG.result_dir / "pdf_process_mentions.csv", all_process_mentions, ["source_pdf", "mention", "kind", "evidence"])
    write_csv(CONFIG.result_dir / "pdf_resource_mentions.csv", all_resource_mentions, ["source_pdf", "mention", "kind", "evidence"])
    write_json(CONFIG.result_dir / "pdf_extract_report.json", report)
    LOGGER.info("PDF supplementary extraction complete: %s", report)


if __name__ == "__main__":
    main()
