"""Import des labels humains historiques vers SQLite.

Les métadonnées des publications proviennent désormais de HAL (refresh_hal) ;
ce script ne lit du CSV historique que le couple (halId_s, manual_label) et ne
remplace jamais une validation existante.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config.config import PILLAR_MAP
from dashboard.scripts.db import connect_db, database_path, ensure_schema

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "manual_labels_with_hal_metadata.csv"


def import_manual_labels(path: Path = DEFAULT_INPUT) -> dict:
    labels = pd.read_csv(path, usecols=["halId_s", "manual_label"], dtype="string").dropna()
    labels["pillar"] = labels["manual_label"].map(lambda value: PILLAR_MAP.get(value.strip().lower()))
    unknown = sorted(labels.loc[labels["pillar"].isna(), "manual_label"].unique())
    if unknown:
        raise ValueError(f"Piliers inconnus dans {path.name} : {unknown}")
    if labels["halId_s"].duplicated().any():
        raise ValueError(f"Identifiants HAL dupliqués dans {path.name}")

    imported, missing = 0, []
    with connect_db() as con:
        ensure_schema(con)
        for hal_id, pillar in zip(labels["halId_s"], labels["pillar"]):
            if con.execute("SELECT 1 FROM articles WHERE hal_id=?", (hal_id,)).fetchone() is None:
                missing.append(hal_id)
                continue
            imported += con.execute(
                "UPDATE articles SET manual_pillar=?, status='validated', updated_at=CURRENT_TIMESTAMP "
                "WHERE hal_id=? AND NOT EXISTS (SELECT 1 FROM validations WHERE hal_id=articles.hal_id)",
                (pillar, hal_id)).rowcount
        # ensure_schema crée les validations 'historical_import' des articles ainsi marqués.
        ensure_schema(con)
    return {"labels": len(labels), "imported": imported,
            "already_validated": len(labels) - imported - len(missing), "missing_from_hal": missing}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Importe les labels humains historiques dans SQLite")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    print(import_manual_labels(parser.parse_args().input))
    print(f"Base SQLite : {database_path()}")
