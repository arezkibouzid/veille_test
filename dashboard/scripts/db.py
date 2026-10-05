"""Accès SQLite centralisé pour SequoIA.

La base opérationnelle est SQLite sur EBS en production.
Définir SQLITE_DB_PATH pour surcharger le chemin par défaut.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = ROOT / "dashboard" / "data" / "validation" / "sequoia_v2.db"


def database_path() -> Path:
    raw = os.getenv("SQLITE_DB_PATH", "").strip()
    return Path(raw).expanduser() if raw else DEFAULT_DB_PATH


@contextmanager
def connect_db(*, readonly: bool = False, path: Path | None = None) -> Iterator[sqlite3.Connection]:
    path = (path or database_path()).expanduser().resolve()

    if readonly:
        if not path.exists():
            raise FileNotFoundError(f"Base SQLite introuvable : {path}")

        connection = sqlite3.connect(
            f"{path.as_uri()}?mode=ro",
            uri=True,
            timeout=30,
        )
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            path,
            timeout=30,
        )

    connection.row_factory = sqlite3.Row

    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 30000")

    try:
        yield connection

        if not readonly:
            connection.commit()

    except Exception:
        if not readonly:
            connection.rollback()
        raise

    finally:
        connection.close()


def table_columns(
    connection: sqlite3.Connection,
    table: str,
) -> set[str]:
    return {
        str(row[1])
        for row in connection.execute(
            f"PRAGMA table_info({table})"
        ).fetchall()
    }


def add_column_if_missing(
    connection: sqlite3.Connection,
    table: str,
    column: str,
    ddl: str,
) -> None:
    if column not in table_columns(connection, table):
        connection.execute(
            f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"
        )


def ensure_schema(connection: sqlite3.Connection) -> None:
    """Crée ou complète le schéma sans supprimer les données existantes."""

    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS articles (
            hal_id TEXT PRIMARY KEY,
            title TEXT,
            authors TEXT,
            abstract TEXT,
            year INTEGER,
            doi TEXT,
            url TEXT,

            openalex_id TEXT,
            citations INTEGER,
            openalex_updated_at TEXT,

            predicted_pillar TEXT,
            pillar_confidence REAL,
            predicted_axis TEXT,
            axis_similarity REAL,
            model_version TEXT,

            status TEXT NOT NULL DEFAULT 'new',

            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );


        CREATE TABLE IF NOT EXISTS validations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            hal_id TEXT NOT NULL,

            validated_pillar TEXT,
            validated_axis TEXT,

            reviewer TEXT,
            validated_at TEXT,

            validation_source TEXT NOT NULL DEFAULT 'dashboard',

            FOREIGN KEY (hal_id)
                REFERENCES articles(hal_id)
                ON DELETE CASCADE
        );


        CREATE UNIQUE INDEX IF NOT EXISTS idx_validations_hal_id
            ON validations(hal_id);


        CREATE TABLE IF NOT EXISTS article_locks (
            hal_id TEXT PRIMARY KEY,

            locked_by TEXT NOT NULL,
            locked_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,

            FOREIGN KEY (hal_id)
                REFERENCES articles(hal_id)
                ON DELETE CASCADE
        );


        CREATE TABLE IF NOT EXISTS model_versions (
            version TEXT PRIMARY KEY,

            status TEXT,
            dataset_version TEXT,
            metrics_json TEXT,

            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS pipeline_jobs (
            id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            status TEXT NOT NULL,
            step TEXT,
            result_json TEXT,
            error TEXT,
            requested_by TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            finished_at TEXT
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_pipeline_single_running
            ON pipeline_jobs(status) WHERE status = 'running';

        CREATE TABLE IF NOT EXISTS validation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hal_id TEXT NOT NULL,
            validated_pillar TEXT,
            validated_axis TEXT,
            reviewer TEXT,
            validated_at TEXT,
            validation_source TEXT,
            action TEXT NOT NULL,
            recorded_at TEXT DEFAULT CURRENT_TIMESTAMP
        );


        CREATE TABLE IF NOT EXISTS publication_authors (
            hal_id TEXT NOT NULL,
            author TEXT NOT NULL,
            position INTEGER,

            PRIMARY KEY (hal_id, author),

            FOREIGN KEY (hal_id)
                REFERENCES articles(hal_id)
                ON DELETE CASCADE
        );


        CREATE TABLE IF NOT EXISTS publication_labs (
            hal_id TEXT NOT NULL,
            lab TEXT NOT NULL,
            position INTEGER,

            PRIMARY KEY (hal_id, lab),

            FOREIGN KEY (hal_id)
                REFERENCES articles(hal_id)
                ON DELETE CASCADE
        );


        CREATE TABLE IF NOT EXISTS publication_labs_cluster (
            hal_id TEXT NOT NULL,
            lab TEXT NOT NULL,

            PRIMARY KEY (hal_id, lab),

            FOREIGN KEY (hal_id)
                REFERENCES articles(hal_id)
                ON DELETE CASCADE
        );


        CREATE TABLE IF NOT EXISTS publication_institutions (
            hal_id TEXT NOT NULL,
            institution TEXT NOT NULL,
            position INTEGER,

            PRIMARY KEY (hal_id, institution),

            FOREIGN KEY (hal_id)
                REFERENCES articles(hal_id)
                ON DELETE CASCADE
        );


        CREATE TABLE IF NOT EXISTS publication_partners (
            hal_id TEXT NOT NULL,
            partner TEXT NOT NULL,
            position INTEGER,

            PRIMARY KEY (hal_id, partner),

            FOREIGN KEY (hal_id)
                REFERENCES articles(hal_id)
                ON DELETE CASCADE
        );


        CREATE TABLE IF NOT EXISTS publication_keywords (
            hal_id TEXT NOT NULL,
            keyword TEXT NOT NULL,
            position INTEGER,

            PRIMARY KEY (hal_id, keyword),

            FOREIGN KEY (hal_id)
                REFERENCES articles(hal_id)
                ON DELETE CASCADE
        );


        CREATE TABLE IF NOT EXISTS publication_domains (
            hal_id TEXT NOT NULL,
            domain TEXT NOT NULL,
            position INTEGER,

            PRIMARY KEY (hal_id, domain),

            FOREIGN KEY (hal_id)
                REFERENCES articles(hal_id)
                ON DELETE CASCADE
        );


        CREATE INDEX IF NOT EXISTS idx_articles_status
            ON articles(status);

        CREATE INDEX IF NOT EXISTS idx_articles_year
            ON articles(year);

        CREATE INDEX IF NOT EXISTS idx_articles_openalex
            ON articles(openalex_id);

        CREATE INDEX IF NOT EXISTS idx_author_hal
            ON publication_authors(hal_id);

        CREATE INDEX IF NOT EXISTS idx_lab_hal
            ON publication_labs_cluster(hal_id);

        CREATE INDEX IF NOT EXISTS idx_partner_hal
            ON publication_partners(hal_id);
        """
    )

    # Compatibilité avec une base SQLite V2 déjà existante.
    # Ces colonnes peuvent ne pas être présentes dans une ancienne table articles.
    extra_columns = {
        "docid": "TEXT",
        "author_ids": "TEXT",
        "labs": "TEXT",
        "institutions": "TEXT",
        "doc_type": "TEXT",
        "journal": "TEXT",
        "conference": "TEXT",
        "keywords": "TEXT",
        "domains": "TEXT",
        "language": "TEXT",
        "author_affil_map": "TEXT",
        "manual_pillar": "TEXT",

        "openalex_title": "TEXT",
        "openalex_doi": "TEXT",
        "openalex_year": "INTEGER",
        "openalex_match_method": "TEXT",
        "openalex_match_status": "TEXT",
        "openalex_match_score": "REAL",
        "openalex_title_score": "REAL",
        "openalex_author_score": "REAL",
        "openalex_year_score": "REAL",
        "openalex_error_reason": "TEXT",
    }

    for column, ddl in extra_columns.items():
        add_column_if_missing(
            connection,
            "articles",
            column,
            ddl,
        )

    for column, ddl in {
        'artifact_path': 'TEXT',
        'training_hal_ids_json': 'TEXT',
    }.items():
        add_column_if_missing(connection, 'model_versions', column, ddl)

    connection.executescript("""
        CREATE TRIGGER IF NOT EXISTS archive_validation_update
        BEFORE UPDATE ON validations
        BEGIN
            INSERT INTO validation_history (
                hal_id, validated_pillar, validated_axis, reviewer,
                validated_at, validation_source, action
            ) VALUES (OLD.hal_id, OLD.validated_pillar, OLD.validated_axis,
                OLD.reviewer, OLD.validated_at, OLD.validation_source, 'corrected');
        END;
        CREATE TRIGGER IF NOT EXISTS archive_validation_delete
        BEFORE DELETE ON validations
        BEGIN
            INSERT INTO validation_history (
                hal_id, validated_pillar, validated_axis, reviewer,
                validated_at, validation_source, action
            ) VALUES (OLD.hal_id, OLD.validated_pillar, OLD.validated_axis,
                OLD.reviewer, OLD.validated_at, OLD.validation_source, 'deleted');
        END;
    """)

    add_column_if_missing(
        connection,
        "validations",
        "validation_source",
        "TEXT NOT NULL DEFAULT 'dashboard'",
    )

    # One-way compatibility migration inside SQLite; never replace a human label.
    connection.execute("""
        INSERT OR IGNORE INTO validations(hal_id,validated_pillar,validation_source)
        SELECT hal_id, manual_pillar, 'historical_import' FROM articles
        WHERE status='validated' AND COALESCE(TRIM(manual_pillar),'') != ''
          AND NOT EXISTS (SELECT 1 FROM validations WHERE hal_id=articles.hal_id)
    """)

    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_validations_source
        ON validations(validation_source)
        """
    )


if __name__ == "__main__":
    with connect_db() as con:
        ensure_schema(con)

    print(f"Schéma SQLite prêt : {database_path()}")
