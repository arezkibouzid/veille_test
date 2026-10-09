"""Extraction HAL vers SQLite"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests

from dashboard.scripts.hal_resources import HAL_RESOURCE_FIELDS, save_documents, enrich_tei, enrich_doi_resources

try:
    from .db import connect_db, database_path, ensure_schema
    from .data_rules import infer_private_partners, normalize_lab, split_values
except ImportError:  # exécution directe du script
    from db import connect_db, database_path, ensure_schema
    from data_rules import infer_private_partners, normalize_lab, split_values


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

HAL_BASE_URL = "https://api.archives-ouvertes.fr/search/"
HAL_COLLECTION = "SEQUOIA"
ROWS_PER_PAGE = 1000

# Si ce script est placé dans dashboard/scripts/,
# ROOT pointe vers la racine du projet.
ROOT = Path(__file__).resolve().parents[2]

HAL_FIELDS = [
    "docid",
    "halId_s",
    "title_s",
    "authFullName_s",
    "authIdHal_s",
    "authIdHasStructure_fs",
    "labStructName_s",
    "instStructName_s",
    "docType_s",
    "publicationDateY_i",
    "producedDateY_i",
    "journalTitle_s",
    "conferenceTitle_s",
    "keyword_s",
    "abstract_s",
    "uri_s",
    "doiId_s",
    "domainAllCode_s",
    "language_s",
]
HAL_FIELDS += HAL_RESOURCE_FIELDS

RETRY_STATUS = {429, 500, 502, 503, 504}


DOC_TYPE_LABELS = {
    "ART": "Article",
    "COMM": "Communication",
    "POSTER": "Poster",
    "COUV": "Chapitre d'ouvrage",
    "OUV": "Ouvrage",
    "THESE": "Thèse",
    "HDR": "HDR",
    "REPORT": "Rapport",
    "UNDEFINED": "Non défini",
    "PROCEEDINGS": "Actes",
    "OTHER": "Autre",
    "SOFTWARE": "Logiciel",
    "PATENT": "Brevet",
    "PRESCONF": "Présentation",
    "MEM": "Mémoire",
    "LECTURE": "Cours",
}


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def _join(values, sep: str = " | ") -> str:
    if values is None:
        return ""

    if isinstance(values, list):
        return sep.join(
            str(value)
            for value in values
            if value is not None
        )

    return str(values)


def _author_affil_map(document: dict) -> str:
    """Construit une version simplifiée auteur -> affiliation principale."""
    entries = document.get(
        "authIdHasStructure_fs"
    ) or []

    if not entries:
        return ""

    affiliations = []
    affiliation_index = {}
    authors = {}
    seen_authors = set()

    for entry in entries:
        try:
            left, right = entry.split(
                "_JoinSep_",
                1,
            )

            author = (
                left.split("_FacetSep_")[-1]
                .strip()
            )

            structure = (
                right.split("_FacetSep_")[-1]
                .strip()
            )

        except ValueError:
            continue

        if not author or not structure:
            continue

        if author in seen_authors:
            continue

        seen_authors.add(author)

        if structure not in affiliation_index:
            affiliation_index[structure] = (
                len(affiliations) + 1
            )
            affiliations.append(structure)

        authors[author] = [
            affiliation_index[structure]
        ]

    if not authors:
        return ""

    return json.dumps(
        {
            "authors": [
                [author, indices]
                for author, indices
                in authors.items()
            ],
            "affiliations": affiliations,
        },
        ensure_ascii=False,
    )


def _get_with_retry(
    url: str,
    params: dict,
    timeout: int = 60,
    max_retries: int = 4,
    base_delay: float = 3.0,
):
    """GET HTTP avec retry exponentiel sur erreurs réseau ou serveur transitoires."""

    for attempt in range(
        max_retries + 1
    ):
        try:
            response = requests.get(
                url,
                params=params,
                timeout=timeout,
            )

            response.raise_for_status()

            return response

        except (
            requests.exceptions.ConnectTimeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.HTTPError,
        ) as exc:
            status = getattr(exc.response, "status_code", None)
            if isinstance(exc, requests.exceptions.HTTPError) and status not in RETRY_STATUS:
                raise

            if attempt == max_retries:
                raise

            delay = (
                base_delay
                * (2 ** attempt)
            )

            print(
                f"Erreur HAL transitoire ({status or type(exc).__name__}). "
                f"Nouvel essai dans {delay:.0f}s..."
            )

            time.sleep(delay)


# --------------------------------------------------
# HAL EXTRACTION
# 

def fetch_hal() -> list[dict]:
    """Télécharge toutes les notices de la collection SEQUOIA."""

    url = (
        HAL_BASE_URL
        + HAL_COLLECTION
        + "/"
    )

    cursor = "*"
    documents = []

    while True:
        response = _get_with_retry(
            url,
            {
                "q": "*:*",
                "wt": "json",
                "fl": ",".join(
                    HAL_FIELDS
                ),
                "rows": ROWS_PER_PAGE,
                "sort": "docid asc",
                "cursorMark": cursor,
            },
        )

        body = response.json()

        page = (
            body
            .get("response", {})
            .get("docs", [])
        )

        documents.extend(page)

        total = (
            body
            .get("response", {})
            .get("numFound", 0)
        )

        print(
            f"{len(documents):,} / "
            f"{total:,} notices récupérées"
        )

        next_cursor = body.get(
            "nextCursorMark"
        )

        if (
            not next_cursor
            or next_cursor == cursor
        ):
            break

        cursor = next_cursor

        time.sleep(0.2)

    if len(documents) != total:
        # Une collection tronquée ferait croire à des retraits de notices.
        raise RuntimeError(
            f"Réponse HAL incomplète : {len(documents)} / {total} notices"
        )

    return documents


def load_notices(path: Path) -> list[dict]:
    """Lit des notices HAL depuis un fichier JSONL (une notice par ligne)."""

    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def to_dataframe(
    documents: list[dict],
) -> pd.DataFrame:
    """Transforme les notices HAL en DataFrame propre."""

    rows = []

    for document in documents:
        year = (
            document.get(
                "publicationDateY_i"
            )
            or document.get(
                "producedDateY_i"
            )
        )

        rows.append(
            {
                "hal_metadata_json": json.dumps(document, ensure_ascii=False),
                "docid":
                    document.get("docid"),

                "halId_s":
                    document.get("halId_s"),

                "title":
                    _join(
                        document.get("title_s"),
                        " / ",
                    ),

                "authors":
                    _join(
                        document.get(
                            "authFullName_s"
                        ),
                        ", ",
                    ),

                "author_ids":
                    _join(
                        document.get(
                            "authIdHal_s"
                        )
                    ),

                "labs":
                    _join(
                        document.get(
                            "labStructName_s"
                        )
                    ),

                "institutions":
                    _join(
                        document.get(
                            "instStructName_s"
                        )
                    ),

                "doc_type":
                    DOC_TYPE_LABELS.get(
                        document.get(
                            "docType_s",
                            "",
                        ),
                        document.get(
                            "docType_s",
                            "",
                        ),
                    ),

                "year":
                    year,

                "journal":
                    document.get(
                        "journalTitle_s",
                        "",
                    ),

                "conference":
                    document.get(
                        "conferenceTitle_s",
                        "",
                    ),

                "keywords":
                    _join(
                        document.get(
                            "keyword_s"
                        ),
                        "; ",
                    ),

                "abstract":
                    _join(
                        document.get(
                            "abstract_s"
                        ),
                        " ",
                    ),

                "url":
                    document.get(
                        "uri_s",
                        "",
                    ),

                "doi":
                    document.get(
                        "doiId_s",
                        "",
                    ),

                "domains":
                    _join(
                        document.get(
                            "domainAllCode_s"
                        ),
                        "; ",
                    ),

                "language":
                    _join(
                        document.get(
                            "language_s"
                        )
                    ),

                "author_affil_map":
                    _author_affil_map(
                        document
                    ),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["halId_s"])

    dataframe = pd.DataFrame(rows)

    dataframe = (
        dataframe
        .dropna(
            subset=["halId_s"]
        )
        .drop_duplicates(
            subset=["halId_s"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    return dataframe


# --------------------------------------------------
# MAIN
# --------------------------------------------------

# --------------------------------------------------
# SQLITE UPSERT
# --------------------------------------------------

def _nullable_int(value):
    try:
        if pd.isna(value):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _clean(value):
    if pd.isna(value):
        return None
    value = str(value).strip()
    return value or None


def replace_relations(connection, hal_id: str, row: pd.Series) -> None:
    specs = [
        ("publication_authors", "author", split_values(row.get("authors", ""), "authors")),
        ("publication_labs", "lab", split_values(row.get("labs", ""), "labs")),
        ("publication_institutions", "institution", split_values(row.get("institutions", ""), "institutions")),
        ("publication_keywords", "keyword", split_values(row.get("keywords", ""), "keywords")),
        ("publication_domains", "domain", split_values(row.get("domains", ""), "domains")),
    ]

    for table, column, values in specs:
        connection.execute(f"DELETE FROM {table} WHERE hal_id = ?", (hal_id,))
        connection.executemany(
            f"INSERT OR IGNORE INTO {table} (hal_id, {column}, position) VALUES (?, ?, ?)",
            [(hal_id, value, pos) for pos, value in enumerate(values, start=1)],
        )

    connection.execute("DELETE FROM publication_labs_cluster WHERE hal_id = ?", (hal_id,))
    normalized = []
    for lab in split_values(row.get("labs", ""), "labs"):
        short = normalize_lab(lab)
        if short and short not in normalized:
            normalized.append(short)
    connection.executemany(
        "INSERT OR IGNORE INTO publication_labs_cluster (hal_id, lab) VALUES (?, ?)",
        [(hal_id, lab) for lab in normalized],
    )

    connection.execute("DELETE FROM publication_partners WHERE hal_id = ?", (hal_id,))
    partners = infer_private_partners(row.get("institutions", ""))
    connection.executemany(
        "INSERT OR IGNORE INTO publication_partners (hal_id, partner, position) VALUES (?, ?, ?)",
        [(hal_id, partner, pos) for pos, partner in enumerate(partners, start=1)],
    )


def upsert_hal(dataframe: pd.DataFrame, *, collection_complete=False) -> tuple[int, int, int]:
    inserted = 0
    updated = 0
    metadata_documents = []

    with connect_db() as connection:
        ensure_schema(connection)
        existing_ids = {
            row[0]
            for row in connection.execute("SELECT hal_id FROM articles").fetchall()
        }

        for _, row in dataframe.iterrows():
            hal_id = _clean(row.get("halId_s"))
            if not hal_id:
                continue

            is_new = hal_id not in existing_ids

            connection.execute(
                """
                INSERT INTO articles (
                    hal_id, docid, title, authors, author_ids, labs, institutions,
                    doc_type, year, journal, conference, keywords, abstract,
                    url, doi, domains, language, author_affil_map,
                    status, created_at, updated_at
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    'new', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
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
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    hal_id,
                    _clean(row.get("docid")),
                    _clean(row.get("title")),
                    _clean(row.get("authors")),
                    _clean(row.get("author_ids")),
                    _clean(row.get("labs")),
                    _clean(row.get("institutions")),
                    _clean(row.get("doc_type")),
                    _nullable_int(row.get("year")),
                    _clean(row.get("journal")),
                    _clean(row.get("conference")),
                    _clean(row.get("keywords")),
                    _clean(row.get("abstract")),
                    _clean(row.get("url")),
                    _clean(row.get("doi")),
                    _clean(row.get("domains")),
                    _clean(row.get("language")),
                    _clean(row.get("author_affil_map")),
                ),
            )

            replace_relations(connection, hal_id, row)
            raw_metadata = row.get('hal_metadata_json')
            if isinstance(raw_metadata, str) and raw_metadata:
                metadata_documents.append(json.loads(raw_metadata))

            if is_new:
                inserted += 1
                existing_ids.add(hal_id)
            else:
                updated += 1

        if metadata_documents:
            save_documents(connection, metadata_documents)
            if collection_complete:
                connection.execute('UPDATE hal_resource_notices SET in_collection=0')
                connection.executemany('UPDATE hal_resource_notices SET in_collection=1 WHERE hal_id=?',
                    [(d['halId_s'],) for d in metadata_documents if 'SEQUOIA' in d.get('collCode_s', [])])
        total = connection.execute("SELECT COUNT(*) FROM articles").fetchone()[0]

    return inserted, updated, total


def refresh_hal(source: Path | None = None) -> dict:
    """Met à jour SQLite depuis l'API HAL, ou depuis un fichier JSONL déjà synchronisé."""
    documents = enrich_tei(load_notices(source) if source else fetch_hal())
    resource_failures = enrich_doi_resources(documents)
    dataframe = to_dataframe(documents)
    inserted, updated, total = upsert_hal(dataframe, collection_complete=True)
    return {"received": len(dataframe), "inserted": inserted,
            "updated": updated, "total": total, "resource_metadata_failures": resource_failures}


def main() -> None:
    parser = argparse.ArgumentParser(description="HAL vers SQLite")
    parser.add_argument(
        "--from-file",
        type=Path,
        help="Fichier JSONL de notices au lieu de l'API HAL (rejeu hors ligne)",
    )
    print(refresh_hal(parser.parse_args().from_file))
    print(f"Base SQLite : {database_path()}")


if __name__ == "__main__":
    main()
