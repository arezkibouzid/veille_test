from __future__ import annotations

import argparse
import os
import re
import sys
import time
import unicodedata
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

try:
    from .db import connect_db, database_path, ensure_schema
    from .report_evidence import citation_value
except ImportError:  # exécution directe du script
    from db import connect_db, database_path, ensure_schema
    from report_evidence import citation_value

OPENALEX_WORKS = "https://api.openalex.org/works"
USER_AGENT = "sequoia-veille/1.0"
TIMEOUT = 30

AUTO_MATCH_MIN_SCORE = 88.0
AUTO_MATCH_MIN_TITLE_SCORE = 90.0
AMBIGUITY_MARGIN = 5.0

MAP_COLUMNS = [
    "publication_id",
    "openalex_id",
    "openalex_title",
    "openalex_doi",
    "openalex_year",
    "match_method",
    "match_status",
    "match_score",
    "title_score",
    "author_score",
    "year_score",
    "error_reason",
]

METRIC_COLUMNS = [
    "publication_id",
    "openalex_id",
    "citations",
    "refreshed_at",
]


class BudgetExhausted(RuntimeError):
    pass


class OpenAlexBadRequest(RuntimeError):
    def __init__(self, message: str, url: str = ""):
        super().__init__(message)
        self.message = message
        self.url = url


def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().split())

def normalize_doi(value) -> str:
    doi = clean_text(value).lower()

    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi)
    doi = re.sub(r"^doi:\s*", "", doi)

    doi = doi.strip()
    doi = doi.rstrip(".,;")
    return doi


def normalize_text(value) -> str:
    text = clean_text(value).casefold()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        c for c in text
        if not unicodedata.combining(c)
    )
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def title_variants(value) -> list[str]:
    """
    HAL contient parfois :
    titre anglais | titre français

    On considère chaque partie comme une variante possible du titre.
    """
    text = clean_text(value)

    if not text:
        return []

    parts = [
        part.strip()
        for part in text.split("|")
        if part.strip()
    ]

    return parts or [text]

def surname(name) -> str:
    text = normalize_text(name)
    return text.split()[-1] if text else ""

def title_similarity(a, b) -> float:
    variants_a = title_variants(a)
    variants_b = title_variants(b)

    best_score = 0.0

    for variant_a in variants_a:
        for variant_b in variants_b:
            norm_a = normalize_text(variant_a)
            norm_b = normalize_text(variant_b)

            if not norm_a or not norm_b:
                continue

            if norm_a == norm_b:
                return 100.0

            score = (
                100.0
                * SequenceMatcher(
                    None,
                    norm_a,
                    norm_b,
                ).ratio()
            )

            best_score = max(
                best_score,
                score,
            )

    return best_score

def doi_candidate_is_plausible(row, work) -> bool:
    """
    Vérifie qu'un résultat trouvé par DOI reste cohérent
    avec la publication HAL.
    """

    title_score = title_similarity(
        row.get("title", ""),
        work.get("display_name", ""),
    )

    try:
        year_hal = int(float(row.get("year")))
    except (TypeError, ValueError):
        year_hal = None

    try:
        year_openalex = int(
            work.get("publication_year")
        )
    except (TypeError, ValueError):
        year_openalex = None

    # Années beaucoup trop éloignées = DOI suspect
    # Écart d'année
    year_gap = None

    if (
        year_hal is not None
        and year_openalex is not None
    ):
        year_gap = abs(
            year_hal - year_openalex
        )

    # Deux ans ou plus + titre non quasi identique
    # => DOI suspect
    if (
        year_gap is not None
        and year_gap >= 2
        and title_score < 90
    ):
        return False

    # Titre trop différent, même si l'année correspond

    if title_score < 60:
        return False

    return True
def year_similarity(a, b) -> float:
    try:
        delta = abs(int(a) - int(b))
    except (TypeError, ValueError):
        return 0.0

    if delta == 0:
        return 100.0
    if delta == 1:
        return 70.0
    return 0.0


def author_similarity(local_authors: list[str], work: dict) -> float | None:
    local = {surname(a) for a in local_authors if surname(a)}
    remote = set()

    for authorship in work.get("authorships") or []:
        author = authorship.get("author") or {}
        s = surname(author.get("display_name", ""))
        if s:
            remote.add(s)

    if not local or not remote:
        return None

    return 100.0 * len(local & remote) / min(len(local), len(remote))


def score_candidate(row: pd.Series, local_authors: list[str], work: dict) -> dict:
    t = title_similarity(row.get("title", ""), work.get("display_name", ""))
    y = year_similarity(row.get("year"), work.get("publication_year"))
    a = author_similarity(local_authors, work)

    if a is None:
        total = 0.90 * t + 0.10 * y
    else:
        total = 0.75 * t + 0.15 * a + 0.10 * y

    return {
        "work": work,
        "score": round(total, 2),
        "title_score": round(t, 2),
        "author_score": None if a is None else round(a, 2),
        "year_score": round(y, 2),
    }


class OpenAlexClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        })

    def _params(self, params=None):
        result = dict(params or {})
        if self.api_key:
            result["api_key"] = self.api_key
        return result

    def get(self, url: str, params=None):
        for attempt in range(4):
            try:
                r = self.session.get(
                    url,
                    params=self._params(params),
                    timeout=TIMEOUT,
                )

                if r.status_code == 404:
                    return r

                if r.status_code == 400:
                    try:
                        payload = r.json()
                        message = (
                            payload.get("message")
                            or payload.get("error")
                            or r.text
                        )
                    except Exception:
                        message = r.text or "Bad Request"

                    raise OpenAlexBadRequest(
                        clean_text(message),
                        url=r.url,
                    )

                if r.status_code == 429:
                    text = r.text.lower()
                    if "budget" in text:
                        raise BudgetExhausted(
                            "quota/budget OpenAlex épuisé"
                        )

                    wait = 2 ** attempt
                    print(f"HTTP 429 — nouvelle tentative dans {wait}s")
                    time.sleep(wait)
                    continue

                r.raise_for_status()
                return r

            except (BudgetExhausted, OpenAlexBadRequest):
                raise

            except requests.RequestException:
                if attempt == 3:
                    raise
                wait = 2 ** attempt
                print(f"Erreur réseau — nouvelle tentative dans {wait}s")
                time.sleep(wait)

        raise RuntimeError("Échec de la requête OpenAlex.")

    def by_doi(self, doi: str) -> dict | None:
        encoded = quote(doi, safe="/()_-.:")
        r = self.get(
            f"{OPENALEX_WORKS}/doi:{encoded}",
            params={
                "select": (
                    "id,doi,display_name,publication_year,"
                    "authorships,cited_by_count"
                )
            },
        )
        return None if r.status_code == 404 else r.json()

    def search_title(self, title: str) -> list[dict]:
        safe_title = clean_text(title)
        safe_title = safe_title.replace("?", " ").replace("*", " ")
        safe_title = re.sub(r"\s+", " ", safe_title).strip()

        r = self.get(
            OPENALEX_WORKS,
            params={
                "search": safe_title[:500],
                "per_page": 5,
                "select": (
                    "id,doi,display_name,publication_year,"
                    "authorships,cited_by_count"
                ),
            },
        )
        return r.json().get("results", [])

    def by_openalex_id(self, openalex_id: str) -> dict | None:
        short_id = clean_text(openalex_id).rstrip("/").split("/")[-1]
        if not short_id:
            return None

        r = self.get(
            f"{OPENALEX_WORKS}/{short_id}",
            params={"select": "id,doi,display_name,publication_year,cited_by_count"},
        )
        return None if r.status_code == 404 else r.json()


def load_authors_from_db(connection) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    rows = connection.execute(
        """
        SELECT hal_id, author
        FROM publication_authors
        ORDER BY hal_id, position, author
        """
    ).fetchall()

    for row in rows:
        result.setdefault(str(row["hal_id"]), []).append(
            clean_text(row["author"])
        )
    return result


def empty_mapping(
    publication_id: str,
    method="",
    status="not_found",
    error_reason="",
) -> dict:
    return {
        "publication_id": publication_id,
        "openalex_id": "",
        "openalex_title": "",
        "openalex_doi": "",
        "openalex_year": "",
        "match_method": method,
        "match_status": status,
        "match_score": "",
        "title_score": "",
        "author_score": "",
        "year_score": "",
        "error_reason": error_reason,
    }


def mapping_from_work(
    publication_id: str,
    work: dict,
    method: str,
    status="matched",
    score="",
    title_score="",
    author_score="",
    year_score="",
) -> dict:
    return {
        "publication_id": publication_id,
        "openalex_id": clean_text(work.get("id", "")),
        "openalex_title": clean_text(work.get("display_name", "")),
        "openalex_doi": normalize_doi(work.get("doi", "")),
        "openalex_year": work.get("publication_year", "") or "",
        "match_method": method,
        "match_status": status,
        "match_score": score,
        "title_score": title_score,
        "author_score": "" if author_score is None else author_score,
        "year_score": year_score,
        "error_reason": "",
    }


def choose_candidate(row, local_authors, candidates) -> dict:
    publication_id = str(row["publication_id"])

    if not candidates:
        return empty_mapping(
            publication_id,
            method="title_search",
            status="not_found",
        )

    ranked = sorted(
        (score_candidate(row, local_authors, work) for work in candidates),
        key=lambda x: x["score"],
        reverse=True,
    )

    best = ranked[0]

    second = (
        ranked[1]["score"]
        if len(ranked) > 1
        else None
    )

    ambiguous = (
        second is not None
        and best["score"] - second < AMBIGUITY_MARGIN
    )

    title_score = best["title_score"]
    author_score = best["author_score"]
    year_score = best["year_score"]

    local_doi = normalize_doi(
        row.get("doi", "")
    )

    candidate_doi = normalize_doi(
        best["work"].get("doi", "")
    )


    # 1. Candidat clairement faux

    clearly_wrong = (
        year_score == 0
        and (
            author_score is None
            or author_score == 0
        )
    )

    if clearly_wrong:
        return empty_mapping(
            publication_id,
            method="title_search",
            status="not_found",
        )


    # DOI HAL et DOI OpenAlex incompatibles

    doi_conflict = (
        local_doi
        and candidate_doi
        and local_doi != candidate_doi
    )

    if doi_conflict:
        return mapping_from_work(
            publication_id,
            best["work"],
            method="title_year_authors",
            status="review",
            score=best["score"],
            title_score=title_score,
            author_score=author_score,
            year_score=year_score,
        )

    # Match très fort

    exact_title_same_year = (
        title_score >= 99
        and year_score == 100
    )

    exact_title_close_year_with_authors = (
        title_score >= 99
        and year_score >= 70
        and author_score is not None
        and author_score >= 50
    )

    # 4. Match standard

    standard_match = (
        best["score"] >= AUTO_MATCH_MIN_SCORE
        and title_score >= AUTO_MATCH_MIN_TITLE_SCORE
        and not ambiguous
        and (
            year_score >= 70
            or (
                author_score is not None
                and author_score >= 50
            )
        )
    )


    matched = (
        exact_title_same_year
        or exact_title_close_year_with_authors
        or standard_match
    )


    return mapping_from_work(
        publication_id,
        best["work"],
        method="title_year_authors",
        status="matched" if matched else "review",
        score=best["score"],
        title_score=title_score,
        author_score=author_score,
        year_score=year_score,
    )



def _mapping_from_article(row) -> dict:
    return {
        "publication_id": str(row["publication_id"]),
        "openalex_id": clean_text(row.get("openalex_id", "")),
        "openalex_title": clean_text(row.get("openalex_title", "")),
        "openalex_doi": clean_text(row.get("openalex_doi", "")),
        "openalex_year": row.get("openalex_year", "") or "",
        "match_method": clean_text(row.get("openalex_match_method", "")),
        "match_status": clean_text(row.get("openalex_match_status", "")),
        "match_score": row.get("openalex_match_score", "") or "",
        "title_score": row.get("openalex_title_score", "") or "",
        "author_score": row.get("openalex_author_score", "") or "",
        "year_score": row.get("openalex_year_score", "") or "",
        "error_reason": clean_text(row.get("openalex_error_reason", "")),
    }


def save_mapping(connection, mapping: dict) -> None:
    connection.execute(
        """
        UPDATE articles
        SET
            openalex_id = ?,
            openalex_title = ?,
            openalex_doi = ?,
            openalex_year = ?,
            openalex_match_method = ?,
            openalex_match_status = ?,
            openalex_match_score = ?,
            openalex_title_score = ?,
            openalex_author_score = ?,
            openalex_year_score = ?,
            openalex_error_reason = ?,
            updated_at = CURRENT_TIMESTAMP
        WHERE hal_id = ?
        """,
        (
            clean_text(mapping.get("openalex_id", "")) or None,
            clean_text(mapping.get("openalex_title", "")) or None,
            clean_text(mapping.get("openalex_doi", "")) or None,
            mapping.get("openalex_year") or None,
            clean_text(mapping.get("match_method", "")) or None,
            clean_text(mapping.get("match_status", "")) or None,
            mapping.get("match_score") if mapping.get("match_score") != "" else None,
            mapping.get("title_score") if mapping.get("title_score") != "" else None,
            mapping.get("author_score") if mapping.get("author_score") != "" else None,
            mapping.get("year_score") if mapping.get("year_score") != "" else None,
            clean_text(mapping.get("error_reason", "")) or None,
            str(mapping["publication_id"]),
        ),
    )


def save_work_metrics(con, hal_id, openalex_id, work):
    """Keep identity evidence alongside counts, with a guarded write after I/O."""
    count = citation_value(work.get('cited_by_count'))
    if count is None:
        raise ValueError('Nombre de citations OpenAlex invalide')
    short_id = str(openalex_id).rstrip('/').split('/')[-1]
    if work.get('id') and str(work['id']).rstrip('/').split('/')[-1] != short_id:
        raise ValueError('Identifiant OpenAlex incohérent')
    return con.execute(
        "UPDATE articles SET citations=?, openalex_updated_at=?, "
        "openalex_title=COALESCE(NULLIF(?,''),openalex_title), "
        "openalex_doi=COALESCE(NULLIF(?,''),openalex_doi), "
        "openalex_year=COALESCE(?,openalex_year), updated_at=CURRENT_TIMESTAMP "
        "WHERE hal_id=? AND openalex_match_status='matched' "
        "AND RTRIM(openalex_id,'/') IN (?,?)",
        (count, datetime.now(timezone.utc).isoformat(timespec='seconds'),
         clean_text(work.get('display_name')), normalize_doi(work.get('doi')),
         work.get('publication_year'), hal_id, short_id, 'https://openalex.org/' + short_id)).rowcount


def enrich_articles(*, hal_ids=None, limit=None, skip_metrics_refresh=False,
                    retry_unresolved=False, recheck_doi_matches=False) -> dict:
    """Les requêtes réseau sont exécutées hors transaction SQLite."""
    client = OpenAlexClient(os.getenv("OPENALEX_API_KEY", "").strip())
    with connect_db(readonly=True) as con:
        publications = pd.read_sql_query(
            "SELECT *, hal_id AS publication_id FROM articles ORDER BY hal_id", con)
        authors = load_authors_from_db(con)
    if hal_ids is not None:
        publications = publications[publications["publication_id"].isin(hal_ids)]
    if limit is not None:
        publications = publications.head(limit)
    processed = 0
    errors = 0
    for _, row in publications.iterrows():
        publication_id = str(row["publication_id"])
        match_status = clean_text(row.get("openalex_match_status"))
        if recheck_doi_matches:
            should_match = clean_text(row.get("openalex_match_method")) in {"doi", "doi_validated"}
        else:
            should_match = match_status != "matched" if retry_unresolved else not match_status
        if not should_match:
            continue
        doi = normalize_doi(row.get("doi"))
        try:
            local_authors = authors.get(publication_id, [])
            work = client.by_doi(doi) if doi else None
            if work is not None and doi_candidate_is_plausible(row, work):
                scores = score_candidate(row, local_authors, work)
                mapping = mapping_from_work(
                    publication_id, work, method="doi_validated", status="matched",
                    score=scores["score"], title_score=scores["title_score"],
                    author_score=scores["author_score"], year_score=scores["year_score"])
            else:
                title = clean_text(row.get("title"))
                candidates = client.search_title(title) if title else []
                mapping = choose_candidate(row, local_authors, candidates)
        except OpenAlexBadRequest as exc:
            mapping = empty_mapping(publication_id, status="error", error_reason=exc.message)
            errors += 1
        except requests.RequestException:
            errors += 1
            continue
        with connect_db() as con:
            save_mapping(con, mapping)
        processed += 1

    if not skip_metrics_refresh:
        with connect_db(readonly=True) as con:
            matched = con.execute(
                "SELECT hal_id, openalex_id FROM articles WHERE openalex_match_status='matched' "
                "AND COALESCE(openalex_id, '') != '' ORDER BY hal_id").fetchall()
        selected_ids = set(publications["publication_id"])
        for row in matched:
            if row["hal_id"] not in selected_ids:
                continue
            try:
                work = client.by_openalex_id(row["openalex_id"])
                if work and work.get('cited_by_count') is not None:
                    with connect_db() as con:
                        save_work_metrics(con, row['hal_id'], row['openalex_id'], work)
            except requests.RequestException:
                errors += 1
    if errors:
        raise RuntimeError(f"OpenAlex : {errors} erreur(s). Les enrichissements réussis sont conservés ; relancer pour reprendre.")
    return {"selected": len(publications), "matching_attempts": processed}


def refresh_citations(*, progress=None) -> dict:
    """Refresh the validated dashboard corpus, resolving only missing IDs."""
    client = OpenAlexClient(os.getenv("OPENALEX_API_KEY", "").strip())
    with connect_db(readonly=True) as con:
        rows = con.execute("SELECT hal_id, openalex_id FROM articles "
                           "WHERE status='validated' AND openalex_match_status='matched' AND TRIM(COALESCE(openalex_id,'')) != '' "
                           "ORDER BY hal_id").fetchall()
        missing_ids = [row[0] for row in con.execute("SELECT hal_id FROM articles WHERE "
                              "status='validated' AND (openalex_match_status IS NOT 'matched' OR TRIM(COALESCE(openalex_id,''))='') ORDER BY hal_id")]
    groups = {}
    for row in rows:
        groups.setdefault(row['openalex_id'].rstrip('/').split('/')[-1], []).append(row['hal_id'])
    updated = not_found = 0
    for index, (openalex_id, ids) in enumerate(groups.items(), 1):
        work = client.by_openalex_id(openalex_id)
        if work is None or work.get('cited_by_count') is None:
            not_found += len(ids)
        else:
            with connect_db() as con:
                for hal_id in ids:
                    if con.execute("SELECT 1 FROM articles WHERE hal_id=? AND status='validated'", (hal_id,)).fetchone():
                        updated += save_work_metrics(con, hal_id, openalex_id, work)
        if progress:
            progress(f'Citations OpenAlex : {index}/{len(groups)} identifiants')
    # Known IDs are refreshed first; matching is reserved for absent IDs.
    if missing_ids:
        if progress:
            progress(f'Recherche OpenAlex : {len(missing_ids)} articles validés sans identifiant')
        enrich_articles(hal_ids=missing_ids, retry_unresolved=True, skip_metrics_refresh=True)
        with connect_db(readonly=True) as con:
            resolved = con.execute("SELECT hal_id, openalex_id FROM articles WHERE status='validated' "
                                   "AND openalex_match_status='matched' AND TRIM(COALESCE(openalex_id,'')) != ''").fetchall()
        wanted = set(missing_ids)
        for row in resolved:
            if row['hal_id'] not in wanted:
                continue
            work = client.by_openalex_id(row['openalex_id'])
            if work is None or work.get('cited_by_count') is None:
                not_found += 1
                continue
            with connect_db() as con:
                if con.execute("SELECT 1 FROM articles WHERE hal_id=? AND status='validated'", (row['hal_id'],)).fetchone():
                    updated += save_work_metrics(con, row['hal_id'], row['openalex_id'], work)
    with connect_db(readonly=True) as con:
        unresolved = con.execute("SELECT COUNT(*) FROM articles WHERE status='validated' "
                                "AND (openalex_match_status IS NOT 'matched' OR TRIM(COALESCE(openalex_id,''))='') ").fetchone()[0]
    return {'updated': updated, 'requested_ids': len(groups), 'not_found': not_found,
            'searched_missing_ids': len(missing_ids), 'unresolved': unresolved}


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrichit directement SQLite avec OpenAlex.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-metrics-refresh", action="store_true")
    parser.add_argument("--retry-unresolved", action="store_true")
    parser.add_argument("--recheck-doi-matches", action="store_true")
    print(enrich_articles(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
