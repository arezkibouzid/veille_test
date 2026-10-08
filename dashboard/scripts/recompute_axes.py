"""Recompute automatic axes, preserving manual classification and pillar predictions."""
from __future__ import annotations

import argparse
import fcntl
import math

from config.config import SEQUOIA_SAXES
from dashboard.scripts.db import connect_db, database_path
from dashboard.scripts.model_store import load_bundle
from src.predict import predict_article


def recompute_axes(*, hal_ids=None, progress=None):
    with connect_db(readonly=True) as con:
        rows = con.execute("SELECT a.*,v.validated_pillar,v.validated_axis FROM articles a "
            "LEFT JOIN validations v USING(hal_id) WHERE a.status='validated' "
            "AND TRIM(COALESCE(v.validated_axis,''))='' ORDER BY a.hal_id").fetchall()
    if hal_ids is not None:
        rows = [r for r in rows if r['hal_id'] in hal_ids]
    if not rows:
        return {'selected': 0, 'updated': 0, 'skipped': 0}
    has_unclassified = any(not (r['validated_pillar'] or r['manual_pillar'] or '').strip() for r in rows)
    version, encoder, classifier, references = load_bundle(include_classifier=has_unclassified)
    updated = 0
    for index, row in enumerate(rows, 1):
        manual = (row['validated_pillar'] or row['manual_pillar'] or '').strip()
        result = predict_article(encoder, classifier, references, row['title'], row['abstract'],
                                 row['keywords'], manual_pillar=manual or None)
        pillar, axis = result['pillar'], result['subaxis']
        score = result['subaxis_similarity']
        if pillar != 'No class' and (axis not in SEQUOIA_SAXES.get(pillar, {})
                                    or score is None or not math.isfinite(float(score))):
            raise ValueError('Axe incompatible avec le pilier retenu')
        # Conditional write protects changes made while inference ran outside the transaction.
        with connect_db() as con:
            con.execute('BEGIN IMMEDIATE')
            assignments = 'predicted_axis=?,axis_similarity=?,axis_pillar=?,axis_model_version=?,updated_at=CURRENT_TIMESTAMP'
            args = [axis, score, pillar, version]
            if not manual:
                assignments += ',predicted_pillar=?,pillar_confidence=?,model_version=?'
                args += [pillar, result['pillar_confidence'], version]
            args += [row['hal_id'], row['manual_pillar'], row['predicted_pillar'], row['title'], row['abstract'],
                     row['keywords'], row['validated_pillar'], row['validated_axis']]
            cursor = con.execute('UPDATE articles SET ' + assignments + " WHERE hal_id=? AND status='validated' "
                "AND manual_pillar IS ? AND predicted_pillar IS ? AND title IS ? AND abstract IS ? AND keywords IS ? "
                "AND (SELECT validated_pillar FROM validations WHERE hal_id=articles.hal_id) IS ? "
                "AND (SELECT validated_axis FROM validations WHERE hal_id=articles.hal_id) IS ?", args)
            updated += cursor.rowcount
        if progress and (index % 25 == 0 or index == len(rows)):
            progress(f'Axes : {index}/{len(rows)} articles traités')
    return {'selected': len(rows), 'updated': updated, 'skipped': len(rows) - updated, 'model_version': version}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Recalcule les axes automatiques des articles classés.')
    parser.parse_args()
    lock_path = database_path().resolve().with_suffix('.pipeline.lock')
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(recompute_axes(progress=lambda message: print(message, flush=True)), flush=True)
