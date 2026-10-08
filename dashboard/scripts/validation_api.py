"""API SequoIA — version SQLite/EBS.

"""

from __future__ import annotations

import math
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    from .db import connect_db, database_path, ensure_schema
except ImportError:  # exécution directe du script
    from db import connect_db, database_path, ensure_schema

ROOT = Path(__file__).resolve().parents[2]
SITE_DIR = ROOT / "dashboard" / "_site"

PILLARS = [
    "Core AI",
    "AI, Cybersecurity and Defense",
    "AI, Environment and Ocean",
    "No class",
]


def load_axes_by_pillar() -> dict[str, list[str]]:
    from config.config import SEQUOIA_SAXES
    return {pillar: list(SEQUOIA_SAXES.get(pillar, {})) for pillar in PILLARS}


AXES_BY_PILLAR = load_axes_by_pillar()


app = FastAPI(title="SequoIA Validation API", version="5.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

security = HTTPBasic()


class ValidationPayload(BaseModel):
    validated_pillar: str
    validated_axis: str | None = None
    reviewer: str | None = None


class ReopenPayload(BaseModel):
    reviewer: str | None = None


def require_admin(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    expected_user = os.getenv("VALIDATION_USER", "").strip()
    expected_password = os.getenv("VALIDATION_PASSWORD", "")

    if not expected_user or not expected_password:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Authentification de validation non configurée. "
                "Définis VALIDATION_USER et VALIDATION_PASSWORD."
            ),
        )

    user_ok = secrets.compare_digest(credentials.username, expected_user)
    password_ok = secrets.compare_digest(credentials.password, expected_password)

    if not (user_ok and password_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Identifiants invalides.",
            headers={"WWW-Authenticate": "Basic"},
        )

    return credentials.username


def row_to_dict(row) -> dict | None:
    return dict(row) if row is not None else None


def rows_to_dicts(rows) -> list[dict]:
    from config.config import PILLAR_MAP
    items = [dict(row) for row in rows]
    for item in items:
        for key in ('pillar', 'final_pillar', 'predicted_pillar', 'validated_pillar'):
            if item.get(key):
                item[key] = PILLAR_MAP.get(str(item[key]).strip().lower(), item[key])
    return items


def normalize_axis(pillar: str, axis: str | None) -> str | None:
    if pillar not in PILLARS:
        raise HTTPException(status_code=400, detail="Pilier invalide.")

    if pillar == "No class":
        return None

    value = (axis or "").strip()
    if not value:
        raise HTTPException(status_code=400, detail="Un axe est requis pour ce pilier.")

    allowed = AXES_BY_PILLAR.get(pillar, [])
    if allowed and value not in allowed:
        raise HTTPException(status_code=400, detail="Axe invalide pour ce pilier.")

    return value


def public_article_sql() -> str:
    return """
        SELECT
            a.hal_id AS halId_s,
            a.hal_id AS publication_id,
            a.docid,
            a.title,
            a.authors,
            a.author_ids,
            a.labs,
            a.institutions,
            a.doc_type,
            a.year,
            a.publication_date,
            a.publication_date_precision,
            a.journal,
            a.conference,
            a.keywords,
            a.abstract,
            a.url,
            a.doi,
            a.domains,
            a.language,
            a.author_affil_map,
            a.openalex_id,
            a.openalex_match_method AS match_method,
            a.openalex_match_status AS match_status,
            a.openalex_match_score AS match_score,
            CASE WHEN a.openalex_match_status='matched' THEN a.citations END AS citations,
            CASE WHEN a.openalex_match_status='matched' THEN a.openalex_updated_at END AS refreshed_at,
            a.predicted_pillar,
            a.pillar_confidence,
            a.predicted_axis,
            a.axis_similarity,
            a.model_version,
            COALESCE(v.validated_pillar, a.predicted_pillar) AS pillar,
            CASE
                WHEN COALESCE(v.validated_pillar, a.predicted_pillar) = 'No class'
                    THEN 'No class'
                WHEN v.validated_axis IS NOT NULL AND TRIM(v.validated_axis) != '' THEN v.validated_axis
                WHEN COALESCE(a.axis_pillar,a.predicted_pillar) = COALESCE(v.validated_pillar,a.predicted_pillar)
                    THEN COALESCE(a.predicted_axis,'')
                ELSE ''
            END AS axis,
            COALESCE(v.validated_pillar, a.predicted_pillar) AS final_pillar,
            CASE
                WHEN v.validation_source = 'historical_import' THEN 'human_historical'
                WHEN v.hal_id IS NOT NULL THEN 'human_validated'
                ELSE 'model'
            END AS pillar_source,
            'validated' AS pillar_status,
            'validated' AS classification_status,
            0 AS needs_review,
            CASE
                WHEN v.validated_axis IS NOT NULL THEN 'human_validated'
                ELSE 'model'
            END AS axis_source,
            CASE
                WHEN v.validated_axis IS NOT NULL THEN 'validated'
                ELSE 'proposed'
            END AS axis_status,
            CASE WHEN a.manual_pillar IS NULL OR a.manual_pillar = '' THEN 1 ELSE 0 END AS is_new_article,
            v.reviewer AS validation_reviewer,
            v.validated_at,
            COALESCE(NULLIF(a.journal, ''), NULLIF(a.conference, ''), '') AS venue
        FROM articles a
        LEFT JOIN validations v ON v.hal_id = a.hal_id
        WHERE a.status = 'validated'
    """


@app.on_event("startup")
def startup() -> None:
    with connect_db() as con:
        ensure_schema(con)


@app.get("/api/health")
def health():
    try:
        with connect_db(readonly=True) as con:
            database_ok = con.execute("SELECT 1").fetchone()[0] == 1
            articles = con.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
            pending = con.execute(
                "SELECT COUNT(*) FROM articles WHERE status = 'to_review'"
            ).fetchone()[0]
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Connexion SQLite impossible : {exc}",
        ) from exc

    return {
        "status": "ok",
        "storage": "sqlite-ebs",
        "database_connected": database_ok,
        "database_path": str(database_path()),
        "articles": articles,
        "to_review": pending,
    }


@app.get("/api/taxonomy")
def taxonomy():
    return {"pillars": PILLARS, "axes_by_pillar": AXES_BY_PILLAR}


@app.get("/api/validation/stats")
def validation_stats(_admin: str = Depends(require_admin)):
    with connect_db(readonly=True) as con:
        to_review = con.execute(
            "SELECT COUNT(*) FROM articles WHERE status = 'to_review'"
        ).fetchone()[0]

        validated_workflow = con.execute(
            """
            SELECT COUNT(*)
            FROM validations v JOIN articles a ON a.hal_id=v.hal_id
            WHERE validation_source != 'historical_import' AND a.status='validated'
            """
        ).fetchone()[0]

    return {
        "total": to_review + validated_workflow,
        "to_review": to_review,
        "validated": validated_workflow,
    }


@app.get("/api/articles/to-review")
def articles_to_review(
    limit: int = Query(default=25, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    search: str | None = None,
    order: Literal["confidence_asc", "confidence_desc", "year_desc"] = "confidence_asc",
    _admin: str = Depends(require_admin),
):
    order_sql = {
        "confidence_asc": "pillar_confidence ASC, hal_id ASC",
        "confidence_desc": "pillar_confidence DESC, hal_id ASC",
        "year_desc": "year DESC, pillar_confidence ASC, hal_id ASC",
    }[order]

    where = ["status = 'to_review'"]
    params: list[object] = []

    if search and search.strip():
        term = f"%{search.strip()}%"
        where.append(
            "(title LIKE ? OR abstract LIKE ? OR authors LIKE ? OR hal_id LIKE ?)"
        )
        params.extend([term, term, term, term])

    where_sql = " AND ".join(where)

    with connect_db(readonly=True) as con:
        total = con.execute(
            f"SELECT COUNT(*) FROM articles WHERE {where_sql}",
            tuple(params),
        ).fetchone()[0]

        rows = con.execute(
            f"""
            SELECT
                hal_id AS halId_s,
                title,
                abstract,
                authors,
                year,
                url,
                doi,
                predicted_pillar,
                pillar_confidence,
                predicted_axis,
                axis_similarity,
                status AS validation_status,
                EXISTS(SELECT 1 FROM validations v WHERE v.hal_id=articles.hal_id) AS has_validation
            FROM articles
            WHERE {where_sql}
            ORDER BY {order_sql}
            LIMIT ? OFFSET ?
            """,
            tuple([*params, limit, offset]),
        ).fetchall()

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "items": rows_to_dicts(rows),
    }


@app.get("/api/articles/{hal_id}")
def article_detail(hal_id: str, _admin: str = Depends(require_admin)):
    with connect_db(readonly=True) as con:
        row = con.execute(
            """
            SELECT
                a.*,
                v.validated_pillar,
                v.validated_axis,
                v.reviewer,
                v.validated_at,
                v.validation_source
            FROM articles a
            LEFT JOIN validations v ON v.hal_id = a.hal_id
            WHERE a.hal_id = ?
            """,
            (hal_id,),
        ).fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Article introuvable.")
    return row_to_dict(row)


@app.post("/api/articles/{hal_id}/validate")
def validate_article(
    hal_id: str,
    payload: ValidationPayload,
    _admin: str = Depends(require_admin),
):
    pillar = payload.validated_pillar.strip()
    axis = normalize_axis(pillar, payload.validated_axis)
    reviewer = payload.reviewer.strip() if payload.reviewer else _admin
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with connect_db() as con:
        con.execute("BEGIN IMMEDIATE")
        existing = con.execute(
            "SELECT status FROM articles WHERE hal_id = ?",
            (hal_id,),
        ).fetchone()
        if existing is None:
            raise HTTPException(status_code=404, detail="Article introuvable.")

        if existing["status"] != "to_review":
            raise HTTPException(status_code=409, detail="Article déjà validé ou non prêt pour validation.")
        if con.execute("SELECT 1 FROM validations WHERE hal_id=?", (hal_id,)).fetchone():
            raise HTTPException(status_code=409, detail="Validation existante conservée ; utiliser une correction explicite.")

        con.execute(
            """
            INSERT INTO validations (
                hal_id, validated_pillar, validated_axis,
                reviewer, validated_at, validation_source
            ) VALUES (?, ?, ?, ?, ?, 'dashboard')
            """,
            (hal_id, pillar, axis, reviewer, now),
        )

        con.execute(
            """
            UPDATE articles
            SET status = 'validated', updated_at = CURRENT_TIMESTAMP
            WHERE hal_id = ?
            """,
            (hal_id,),
        )

        row = con.execute(
            """
            SELECT
                a.hal_id AS halId_s,
                a.status AS validation_status,
                v.validated_pillar,
                v.validated_axis,
                v.reviewer,
                v.validated_at
            FROM articles a
            JOIN validations v ON v.hal_id = a.hal_id
            WHERE a.hal_id = ?
            """,
            (hal_id,),
        ).fetchone()

    return {"ok": True, "article": row_to_dict(row)}


@app.post("/api/articles/{hal_id}/reopen")
def reopen_article(
    hal_id: str,
    payload: ReopenPayload | None = None,
    _admin: str = Depends(require_admin),
):
    with connect_db() as con:
        con.execute("BEGIN IMMEDIATE")
        validation = con.execute(
            "SELECT validation_source FROM validations WHERE hal_id = ?",
            (hal_id,),
        ).fetchone()

        if validation is None:
            raise HTTPException(status_code=404, detail="Validation introuvable.")

        if validation["validation_source"] == "historical_import":
            raise HTTPException(
                status_code=409,
                detail="Une annotation historique ne peut pas être rouverte depuis cette page.",
            )

        # Reopening changes queue membership, never deletes the human label.
        con.execute("INSERT INTO validation_history(hal_id,validated_pillar,validated_axis,reviewer,validated_at,validation_source,action) "
                    "SELECT hal_id,validated_pillar,validated_axis,reviewer,validated_at,validation_source,'reopened' "
                    "FROM validations WHERE hal_id=?", (hal_id,))
        con.execute(
            "UPDATE articles SET status='to_review', updated_at=CURRENT_TIMESTAMP WHERE hal_id=?",
            (hal_id,),
        )

    return {"ok": True, "halId_s": hal_id, "validation_status": "to_review"}


@app.post("/api/articles/{hal_id}/correct")
def correct_article(hal_id: str, payload: ValidationPayload,
                    _admin: str = Depends(require_admin)):
    pillar = payload.validated_pillar.strip()
    axis = normalize_axis(pillar, payload.validated_axis)
    with connect_db() as con:
        con.execute("BEGIN IMMEDIATE")
        current = con.execute("SELECT a.status FROM articles a JOIN validations v ON v.hal_id=a.hal_id WHERE a.hal_id=?", (hal_id,)).fetchone()
        if current is None or current["status"] != "to_review":
            raise HTTPException(status_code=409, detail="Rouvrir l’article avant de corriger sa validation.")
        con.execute("UPDATE validations SET validated_pillar=?, validated_axis=?, reviewer=?, validated_at=?, validation_source='dashboard' WHERE hal_id=?",
                    (pillar, axis, payload.reviewer or _admin, datetime.now(timezone.utc).isoformat(timespec="seconds"), hal_id))
        con.execute("UPDATE articles SET status='validated', updated_at=CURRENT_TIMESTAMP WHERE hal_id=?", (hal_id,))
    return {"ok": True, "halId_s": hal_id}


@app.get("/api/pipelines/latest")
def pipeline_latest(_admin: str = Depends(require_admin)):
    from dashboard.scripts.pipeline_jobs import latest_job, enabled
    return {"enabled": enabled(), "job": latest_job()}


@app.get("/api/pipelines/jobs/{job_id}")
def pipeline_status(job_id: str, _admin: str = Depends(require_admin)):
    from dashboard.scripts.pipeline_jobs import get_job
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Traitement introuvable.")
    return job


def public_citation_job(job):
    if not job or job['kind'] != 'citations':
        return None
    return {key: job[key] for key in ('id', 'kind', 'status', 'step', 'result', 'error')}


@app.get("/api/citations/latest")
def citation_latest():
    from dashboard.scripts.pipeline_jobs import latest_job, enabled
    job = latest_job()
    return {'enabled': enabled(), 'busy': bool(job and job['status'] == 'running'),
            'job': public_citation_job(job)}


@app.get("/api/citations/jobs/{job_id}")
def citation_job(job_id: str):
    from dashboard.scripts.pipeline_jobs import get_job
    job = public_citation_job(get_job(job_id))
    if job is None:
        raise HTTPException(status_code=404, detail='Traitement de citations introuvable.')
    return job


@app.post("/api/pipelines/citations", status_code=202)
def citation_start():
    return pipeline_start('citations', 'dashboard-public')


@app.post("/api/pipelines/{kind}", status_code=202)
def pipeline_start(kind: Literal["predict", "retrain", "report"], _admin: str = Depends(require_admin)):
    from dashboard.scripts.pipeline_jobs import enabled, start_job, PipelineBusy
    if not enabled():
        raise HTTPException(status_code=403, detail="Traitements désactivés : configurer SEQUOIA_ENABLE_PIPELINE_JOBS=1 dans l’environnement de test.")
    try:
        job_id = start_job(kind, _admin)
    except PipelineBusy as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"job_id": job_id}


@app.get("/api/reports")
def list_reports(_admin: str = Depends(require_admin)):
    from dashboard.scripts.veille_report import report_summary
    from dashboard.scripts.pipeline_jobs import enabled
    with connect_db(readonly=True) as con:
        rows = con.execute('SELECT id,generated_at,recent_since,article_count,recent_count '
                           'FROM veille_reports ORDER BY rowid DESC').fetchall()
    return {'enabled': enabled(), 'items': [report_summary(row) for row in rows]}


@app.get("/api/reports/{report_id}/{format}")
def report_download(report_id: str, format: Literal['html', 'pdf'],
                    _admin: str = Depends(require_admin)):
    from dashboard.scripts.veille_report import reports_directory
    with connect_db(readonly=True) as con:
        report = con.execute('SELECT id,generated_at FROM veille_reports WHERE id=?', (report_id,)).fetchone()
    if report is None:
        raise HTTPException(status_code=404, detail='Rapport introuvable.')
    # IDs are generated UUIDs, never user-supplied paths.
    if len(report['id']) != 32 or any(char not in '0123456789abcdef' for char in report['id']):
        raise HTTPException(status_code=404, detail='Rapport introuvable.')
    path = reports_directory() / report['id'] / ('report.' + format)
    if not path.is_file():
        raise HTTPException(status_code=404, detail='Fichier de rapport absent ; relancer la génération.')
    headers = {'Cache-Control': 'private, no-cache'}
    if format == 'pdf':
        return FileResponse(path, media_type='application/pdf', headers=headers,
                            filename='sequoia-veille-' + report['generated_at'][:10] + '.pdf')
    return FileResponse(path, media_type='text/html', headers=headers)


@app.get("/rapports.html", include_in_schema=False)
def reports_page(_admin: str = Depends(require_admin)):
    page = SITE_DIR / 'rapports.html'
    if not page.is_file():
        raise HTTPException(status_code=404, detail='Lance `quarto render dashboard` pour construire la page.')
    return FileResponse(page)


@app.get("/api/articles/validated/list")
def validated_articles(_admin: str = Depends(require_admin)):
    with connect_db(readonly=True) as con:
        rows = con.execute(
            """
            SELECT
                hal_id AS halId_s,
                validated_pillar,
                validated_axis,
                reviewer,
                validated_at,
                validation_source
            FROM validations
            WHERE validation_source != 'historical_import'
            ORDER BY validated_at DESC
            """
        ).fetchall()
    return {"items": rows_to_dicts(rows)}


@app.get("/api/dashboard/publications")
def dashboard_publications():
    with connect_db(readonly=True) as con:
        rows = con.execute(public_article_sql() + " ORDER BY a.year DESC, a.hal_id").fetchall()
    return {"count": len(rows), "items": rows_to_dicts(rows)}


@app.get("/api/dashboard/resources")
def dashboard_resources():
    from dashboard.scripts.resource_catalog import catalog
    with connect_db(readonly=True) as con:
        con.execute('BEGIN')
        return catalog(con)


@app.get("/api/dashboard/citations-status")
def citations_status():
    with connect_db(readonly=True) as con:
        row = con.execute("SELECT MAX(openalex_updated_at) AS last_updated_at, "
            "MIN(openalex_updated_at) AS oldest_updated_at, COUNT(*) AS eligible, "
            "COUNT(openalex_updated_at) AS refreshed, "
            "SUM(CASE WHEN TRIM(COALESCE(openalex_id,'')) != '' THEN 1 ELSE 0 END) AS with_id "
            "FROM articles WHERE status='validated'").fetchone()
    return dict(row)


def relation_payload(table: str, value_column: str) -> dict:
    allowed = {
        "publication_authors": "author",
        "publication_labs_cluster": "lab",
        "publication_partners": "partner",
    }
    if allowed.get(table) != value_column:
        raise RuntimeError("Relation dashboard non autorisée")

    with connect_db(readonly=True) as con:
        rows = con.execute(
            f"""
            SELECT
                r.hal_id AS publication_id,
                r.hal_id AS halId_s,
                r.{value_column} AS {value_column}
            FROM {table} r
            JOIN articles a ON a.hal_id = r.hal_id
            WHERE a.status = 'validated'
            ORDER BY r.{value_column}, r.hal_id
            """
        ).fetchall()
    return {"count": len(rows), "items": rows_to_dicts(rows)}


@app.get("/api/dashboard/labs")
def dashboard_labs():
    return relation_payload("publication_labs_cluster", "lab")


@app.get("/api/dashboard/partners")
def dashboard_partners():
    return relation_payload("publication_partners", "partner")


@app.get("/api/dashboard/authors")
def dashboard_authors():
    return relation_payload("publication_authors", "author")


@app.get("/validation.html", include_in_schema=False)
def validation_page(_admin: str = Depends(require_admin)):
    page = SITE_DIR / "validation.html"
    if not page.exists():
        raise HTTPException(
            status_code=404,
            detail="validation.html introuvable. Lance `quarto render dashboard`.",
        )
    return FileResponse(page)


if SITE_DIR.exists():
    app.mount(
        "/",
        StaticFiles(directory=str(SITE_DIR), html=True),
        name="quarto-site",
    )
