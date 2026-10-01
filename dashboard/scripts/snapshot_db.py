"""Crée un snapshot cohérent de la base SQLite SequoIA.

Variables d'environnement :
- SQLITE_DB_PATH : chemin de la base source
- SQLITE_SNAPSHOT_PATH : chemin du snapshot de sortie (optionnel)
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SOURCE = (
    ROOT
    / "dashboard"
    / "data"
    / "validation"
    / "sequoia_v2.db"
)

DEFAULT_SNAPSHOT = (
    ROOT
    / "dashboard"
    / "data"
    / "snapshots"
    / "sequoia_v2_snapshot.db"
)


def source_path() -> Path:
    raw = os.getenv("SQLITE_DB_PATH", "").strip()
    return Path(raw).expanduser() if raw else DEFAULT_SOURCE


def snapshot_path() -> Path:
    raw = os.getenv("SQLITE_SNAPSHOT_PATH", "").strip()
    return Path(raw).expanduser() if raw else DEFAULT_SNAPSHOT


def create_snapshot(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(
            f"Base SQLite source introuvable : {source}"
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if destination.exists():
        destination.unlink()

    source_connection = sqlite3.connect(
        f"file:{source}?mode=ro",
        uri=True,
        timeout=30,
    )

    destination_connection = sqlite3.connect(
        destination,
        timeout=30,
    )

    try:
        source_connection.execute(
            "PRAGMA busy_timeout = 30000"
        )

        source_connection.backup(
            destination_connection
        )

        result = destination_connection.execute(
            "PRAGMA integrity_check"
        ).fetchone()

        if not result or result[0] != "ok":
            raise RuntimeError(
                f"Échec integrity_check : {result}"
            )

        destination_connection.commit()

    finally:
        destination_connection.close()
        source_connection.close()


def main() -> None:
    source = source_path()
    destination = snapshot_path()

    create_snapshot(
        source,
        destination,
    )

    size_mb = destination.stat().st_size / (1024 * 1024)

    print(f"Source   : {source}")
    print(f"Snapshot : {destination}")
    print(f"Taille   : {size_mb:.2f} MB")
    print("Integrity check : ok")


if __name__ == "__main__":
    main()