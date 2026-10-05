"""Single background pipeline, coordinated across API processes by a file lock."""
from __future__ import annotations

import fcntl
import json
import os
import threading
import uuid

from dashboard.scripts.db import connect_db, database_path


class PipelineBusy(RuntimeError):
    pass


def enabled() -> bool:
    return os.getenv('SEQUOIA_ENABLE_PIPELINE_JOBS', '0') == '1'


def update_job(job_id, *, step=None, status=None, result=None, error=None):
    with connect_db() as con:
        con.execute(
            'UPDATE pipeline_jobs SET step=COALESCE(?,step), status=COALESCE(?,status), '
            'result_json=COALESCE(?,result_json), error=?, '
            'finished_at=CASE WHEN ? IS NOT NULL THEN CURRENT_TIMESTAMP ELSE finished_at END WHERE id=?',
            (step, status, json.dumps(result) if result is not None else None, error, status, job_id))


def job_payload(row):
    item = dict(row)
    item['result'] = json.loads(item.pop('result_json') or 'null')
    return item


def get_job(job_id):
    latest_job()  # Detect a stopped worker while a browser is polling.
    with connect_db(readonly=True) as con:
        row = con.execute('SELECT * FROM pipeline_jobs WHERE id=?', (job_id,)).fetchone()
    return job_payload(row) if row else None


def latest_job():
    lock_path = database_path().resolve().with_suffix('.pipeline.lock')
    if lock_path.exists():
        with lock_path.open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                with connect_db() as con:
                    con.execute("UPDATE pipeline_jobs SET status='failed', error='Processus interrompu', "
                                "finished_at=CURRENT_TIMESTAMP WHERE status='running'")
    with connect_db(readonly=True) as con:
        row = con.execute('SELECT * FROM pipeline_jobs ORDER BY rowid DESC LIMIT 1').fetchone()
    return job_payload(row) if row else None


def predict_pipeline(progress):
    from dashboard.scripts.model_store import active_bundle
    from dashboard.scripts.refresh_hal import refresh_hal
    from dashboard.scripts.enrich_openalex_db import enrich_articles
    from dashboard.scripts.predict_new_articles import predict_new_articles
    # Check local model availability before making external requests.
    active_bundle()
    progress('Extraction HAL')
    hal_result = refresh_hal()
    with connect_db(readonly=True) as con:
        ids = [row[0] for row in con.execute(
            "SELECT hal_id FROM articles WHERE status='new' AND NOT EXISTS "
            '(SELECT 1 FROM validations WHERE hal_id=articles.hal_id) ORDER BY hal_id')]
    progress('Enrichissement OpenAlex')
    oa_result = enrich_articles(hal_ids=ids, retry_unresolved=True)
    progress('Prédiction')
    result = predict_new_articles(hal_ids=ids, progress=progress)
    return {'hal': hal_result, 'openalex': oa_result, **result}


def retrain_pipeline(progress, job_id):
    from config.config import ARTIFACT_DIR
    from dashboard.scripts.snapshot_db import create_snapshot
    from main import main
    progress('Snapshot des validations SQLite')
    snapshot = ARTIFACT_DIR / 'datasets' / f'{job_id}.db'
    create_snapshot(database_path(), snapshot)
    progress('Entraînement et évaluation')
    return main(snapshot)


def run_job(job_id, kind, lock):
    try:
        progress = lambda step: update_job(job_id, step=step)
        if kind == 'citations':
            from dashboard.scripts.enrich_openalex_db import refresh_citations
            progress('Rafraîchissement des citations OpenAlex')
            result = refresh_citations(progress=progress)
        else:
            result = predict_pipeline(progress) if kind == 'predict' else retrain_pipeline(progress, job_id)
        update_job(job_id, status='succeeded', step='Terminé', result=result)
    except Exception as exc:
        # Requests errors may contain keys in their URLs; do not expose them.
        safe_message = str(exc) if isinstance(exc, (ValueError, FileNotFoundError)) else type(exc).__name__
        update_job(job_id, status='failed', step='Échec — relance possible', error=safe_message)
    finally:
        lock.close()


def start_job(kind, reviewer):
    if kind not in {'predict', 'retrain', 'citations'}:
        raise ValueError('Pipeline inconnu')
    lock_path = database_path().resolve().with_suffix('.pipeline.lock')
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open('a')
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PipelineBusy('Un traitement est déjà en cours.') from exc
        job_id = uuid.uuid4().hex
        with connect_db() as con:
            con.execute('BEGIN IMMEDIATE')
            # A free process lock proves that any recorded running job was interrupted.
            con.execute("UPDATE pipeline_jobs SET status='failed', error='Processus interrompu', "
                        "finished_at=CURRENT_TIMESTAMP WHERE status='running'")
            con.execute("INSERT INTO pipeline_jobs(id,kind,status,step,requested_by) "
                        "VALUES (?,?,'running','Démarrage',?)", (job_id, kind, reviewer))
        thread = threading.Thread(target=run_job, args=(job_id, kind, lock), daemon=True)
        thread.start()
        return job_id
    except Exception:
        lock.close()
        raise
