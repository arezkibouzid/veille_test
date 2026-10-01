"""Migration unique du dataset historique CSV vers la base SQLite opérationnelle.

- compléter la table articles avec toutes les métadonnées du dashboard ;
- importer les 1 402 piliers humains historiques dans validations ;
- construire les tables relationnelles utilisées par les pages Quarto.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from .db import connect_db, database_path, ensure_schema
    from .data_rules import infer_private_partners, normalize_lab, split_values
except ImportError:  # exécution directe du script
    from db import connect_db, database_path, ensure_schema
    from data_rules import infer_private_partners, normalize_lab, split_values

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "dashboard" / "data" / "processed" / "publications_dashboard.csv"


def text(row: pd.Series, *names: str) -> str | None:
    for name in names:
        if name not in row.index:
            continue
        value = row.get(name)
        if pd.isna(value):
            continue
        value = str(value).strip()
        if value and value.lower() not in {"nan", "none", "<na>"}:
            return value
    return None


def integer(row: pd.Series, *names: str) -> int | None:
    value = text(row, *names)
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def number(row: pd.Series, *names: str) -> float | None:
    value = text(row, *names)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def status_for(row: pd.Series) -> str:
    raw = (text(row, "classification_status", "pillar_status", "status") or "").lower()
    if raw in {"validated", "to_review", "new"}:
        return raw
    return "validated" if text(row, "manual_pillar") else "to_review"


def replace_relations(connection, hal_id: str, row: pd.Series) -> None:
    relation_specs = [
        ("publication_authors", "author", split_values(text(row, "authors") or "", "authors")),
        ("publication_labs", "lab", split_values(text(row, "labs") or "", "labs")),
        (
            "publication_institutions",
            "institution",
            split_values(text(row, "institutions") or "", "institutions"),
        ),
        (
            "publication_keywords",
            "keyword",
            split_values(text(row, "keywords") or "", "keywords"),
        ),
        (
            "publication_domains",
            "domain",
            split_values(text(row, "domains") or "", "domains"),
        ),
    ]

    for table, value_column, values in relation_specs:
        connection.execute(f"DELETE FROM {table} WHERE hal_id = ?", (hal_id,))
        connection.executemany(
            f"INSERT OR IGNORE INTO {table} (hal_id, {value_column}, position) VALUES (?, ?, ?)",
            [(hal_id, value, index) for index, value in enumerate(values, start=1)],
        )

    connection.execute("DELETE FROM publication_labs_cluster WHERE hal_id = ?", (hal_id,))
    normalized_labs = []
    for lab in split_values(text(row, "labs") or "", "labs"):
        normalized = normalize_lab(lab)
        if normalized and normalized not in normalized_labs:
            normalized_labs.append(normalized)
    connection.executemany(
        "INSERT OR IGNORE INTO publication_labs_cluster (hal_id, lab) VALUES (?, ?)",
        [(hal_id, lab) for lab in normalized_labs],
    )

    connection.execute("DELETE FROM publication_partners WHERE hal_id = ?", (hal_id,))
    partners = infer_private_partners(text(row, "institutions") or "")
    connection.executemany(
        "INSERT OR IGNORE INTO publication_partners (hal_id, partner, position) VALUES (?, ?, ?)",
        [(hal_id, partner, index) for index, partner in enumerate(partners, start=1)],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Dataset introuvable : {args.input}")

    df = pd.read_csv(args.input, low_memory=False, dtype={"halId_s": "string"})

    if "halId_s" not in df.columns:
        raise ValueError("publications_dashboard.csv doit contenir halId_s")

    imported = 0
    historical_validations = 0

    with connect_db() as con:
        ensure_schema(con)

        for _, row in df.iterrows():
            hal_id = text(row, "halId_s", "hal_id", "publication_id")
            if not hal_id:
                continue

            desired_status = status_for(row)

            con.execute(
                """
                INSERT INTO articles (
                    hal_id, docid, title, authors, author_ids, labs, institutions,
                    doc_type, year, journal, conference, keywords, abstract, url,
                    doi, domains, language, author_affil_map, manual_pillar,
                    openalex_id, openalex_match_method, openalex_match_status,
                    openalex_match_score, citations, openalex_updated_at,
                    predicted_pillar, pillar_confidence, predicted_axis,
                    axis_similarity, model_version, status, updated_at
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP
                )
                ON CONFLICT(hal_id) DO UPDATE SET
                    docid = excluded.docid,
                    title = excluded.title,
                    authors = excluded.authors,
                    author_ids = excluded.author_ids,
                    labs = excluded.labs,
                    institutions = excluded.institutions,
                    doc_type = excluded.doc_type,
                    year = excluded.year,
                    journal = excluded.journal,
                    conference = excluded.conference,
                    keywords = excluded.keywords,
                    abstract = excluded.abstract,
                    url = excluded.url,
                    doi = excluded.doi,
                    domains = excluded.domains,
                    language = excluded.language,
                    author_affil_map = excluded.author_affil_map,
                    manual_pillar = COALESCE(excluded.manual_pillar, articles.manual_pillar),
                    openalex_id = COALESCE(excluded.openalex_id, articles.openalex_id),
                    openalex_match_method = COALESCE(excluded.openalex_match_method, articles.openalex_match_method),
                    openalex_match_status = COALESCE(excluded.openalex_match_status, articles.openalex_match_status),
                    openalex_match_score = COALESCE(excluded.openalex_match_score, articles.openalex_match_score),
                    citations = COALESCE(excluded.citations, articles.citations),
                    openalex_updated_at = COALESCE(excluded.openalex_updated_at, articles.openalex_updated_at),
                    predicted_pillar = COALESCE(excluded.predicted_pillar, articles.predicted_pillar),
                    pillar_confidence = COALESCE(excluded.pillar_confidence, articles.pillar_confidence),
                    predicted_axis = COALESCE(excluded.predicted_axis, articles.predicted_axis),
                    axis_similarity = COALESCE(excluded.axis_similarity, articles.axis_similarity),
                    model_version = COALESCE(excluded.model_version, articles.model_version),
                    status = CASE
                        WHEN articles.status = 'validated' THEN 'validated'
                        ELSE excluded.status
                    END,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    hal_id,
                    text(row, "docid"),
                    text(row, "title"),
                    text(row, "authors"),
                    text(row, "author_ids"),
                    text(row, "labs"),
                    text(row, "institutions"),
                    text(row, "doc_type"),
                    integer(row, "year"),
                    text(row, "journal"),
                    text(row, "conference"),
                    text(row, "keywords"),
                    text(row, "abstract"),
                    text(row, "url"),
                    text(row, "doi"),
                    text(row, "domains"),
                    text(row, "language"),
                    text(row, "author_affil_map"),
                    text(row, "manual_pillar"),
                    text(row, "openalex_id"),
                    text(row, "match_method", "openalex_match_method"),
                    text(row, "match_status", "openalex_match_status"),
                    number(row, "match_score", "openalex_match_score"),
                    integer(row, "citations"),
                    text(row, "refreshed_at", "openalex_updated_at"),
                    text(row, "predicted_pillar"),
                    number(row, "pillar_confidence"),
                    text(row, "predicted_axis"),
                    number(row, "axis_similarity"),
                    text(row, "model_version"),
                    desired_status,
                ),
            )

            manual_pillar = text(row, "manual_pillar")
            if manual_pillar:
                existing = con.execute(
                    "SELECT validation_source FROM validations WHERE hal_id = ?",
                    (hal_id,),
                ).fetchone()

                if existing is None:
                    con.execute(
                        """
                        INSERT INTO validations (
                            hal_id, validated_pillar, validated_axis,
                            reviewer, validated_at, validation_source
                        ) VALUES (?, ?, NULL, NULL, NULL, 'historical_import')
                        """,
                        (hal_id, manual_pillar),
                    )
                    historical_validations += 1

                con.execute(
                    "UPDATE articles SET status = 'validated' WHERE hal_id = ?",
                    (hal_id,),
                )

            replace_relations(con, hal_id, row)
            imported += 1

        total = con.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        validations = con.execute("SELECT COUNT(*) FROM validations").fetchone()[0]
        pending = con.execute("SELECT COUNT(*) FROM articles WHERE status='to_review'").fetchone()[0]
        validated = con.execute("SELECT COUNT(*) FROM articles WHERE status='validated'").fetchone()[0]

    print(f"Base : {database_path()}")
    print(f"Articles parcourus : {imported}")
    print(f"Articles en DB : {total}")
    print(f"Validés : {validated}")
    print(f"À valider : {pending}")
    print(f"Validations en DB : {validations}")
    print(f"Labels historiques ajoutés cette exécution : {historical_validations}")


if __name__ == "__main__":
    main()
