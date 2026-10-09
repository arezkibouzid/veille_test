"""Ingestion HAL sans réseau : chargement SQLite, labels historiques, holdout, déclencheur."""
import csv
import json

import pytest

from dashboard.scripts import refresh_hal, trigger_pipeline
from dashboard.scripts.db import connect_db
from dashboard.scripts.import_manual_labels import import_manual_labels
from src.preprocess import in_holdout, load_and_preprocess_data

PILLARS = ["Core AI", "AI, cybersecurity and defense", "AI, environment and ocean", "No class"]


def notice(number, title=None):
    return {
        "docid": str(number), "halId_s": f"hal-{number:08d}", "title_s": [title or f"Title {number}"],
        "abstract_s": [f"Abstract {number}"], "keyword_s": ["ai", "ocean"], "authFullName_s": ["A. Author"],
        "docType_s": "ART", "publicationDateY_i": 2025, "collCode_s": ["SEQUOIA"],
    }


def test_notices_and_labels_feed_the_training_dataset(tmp_path, monkeypatch):
    monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "sequoia.db"))
    store = tmp_path / "notices.jsonl"
    store.write_text("\n".join(json.dumps(notice(n)) for n in range(1, 41)), encoding="utf-8")

    result = refresh_hal.refresh_hal(store)
    assert (result["inserted"], result["total"]) == (40, 40)

    labels = tmp_path / "labels.csv"
    rows = [(f"hal-{n:08d}", PILLARS[n % 4]) for n in range(1, 37)] + [("hal-99999999", "Core AI")]
    with labels.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerows([("halId_s", "manual_label"), *rows])
    assert import_manual_labels(labels) == {
        "labels": 37, "imported": 36, "already_validated": 0, "missing_from_hal": ["hal-99999999"]}
    assert import_manual_labels(labels)["already_validated"] == 36

    # Une nouvelle extraction HAL met à jour les métadonnées sans toucher aux décisions humaines.
    store.write_text("\n".join(json.dumps(notice(n, title=f"Renamed {n}")) for n in range(1, 42)), encoding="utf-8")
    result = refresh_hal.refresh_hal(store)
    assert (result["inserted"], result["updated"]) == (1, 40)
    with connect_db(readonly=True) as con:
        statuses = dict(con.execute("SELECT status, COUNT(*) FROM articles GROUP BY status").fetchall())
        sources = [row[0] for row in con.execute("SELECT DISTINCT validation_source FROM validations")]
        title = con.execute("SELECT title FROM articles WHERE hal_id='hal-00000001'").fetchone()[0]
    assert statuses == {"new": 5, "validated": 36}
    assert sources == ["historical_import"]
    assert title == "Renamed 1"

    df, train_df, test_df, _ = load_and_preprocess_data()
    assert len(df) == 36 and len(train_df) + len(test_df) == 36
    assert set(test_df["halId_s"]) == {hal_id for hal_id in df["halId_s"] if in_holdout(hal_id)}
    assert df["manual_label"].str.contains("Cybersecurity and Defense").any()


def test_trigger_waits_for_the_job_and_reports_it(monkeypatch):
    answers = [
        {"job_id": "abc"},
        {"status": "running", "step": "Extraction HAL"},
        {"status": "succeeded", "step": "Terminé",
         "result": {"hal": {"received": 41, "inserted": 1, "updated": 40, "total": 41}, "predicted": 1}},
    ]
    calls = []
    monkeypatch.setattr(trigger_pipeline, "call", lambda method, path: calls.append((method, path)) or answers.pop(0))
    monkeypatch.setattr(trigger_pipeline.time, "sleep", lambda _: None)
    monkeypatch.setattr("sys.argv", ["trigger_pipeline", "predict"])

    trigger_pipeline.main()
    assert calls == [("POST", "/api/pipelines/predict"), ("GET", "/api/pipelines/jobs/abc"),
                     ("GET", "/api/pipelines/jobs/abc")]

    failed = {"status": "failed", "step": "Échec", "error": "FileNotFoundError", "result": None}
    answers.extend([{"job_id": "abc"}, failed])
    with pytest.raises(SystemExit):
        trigger_pipeline.main()
    assert "FileNotFoundError" in trigger_pipeline.markdown_summary("predict", failed)
