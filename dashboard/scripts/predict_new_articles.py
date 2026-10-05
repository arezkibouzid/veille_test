"""Inference from SQLite, with conditional writes protecting human decisions."""
from __future__ import annotations

import math
from config.config import PILLAR_MAP
from dashboard.scripts.db import connect_db
from dashboard.scripts.model_store import load_bundle
from src.predict import predict_article


def predict_new_articles(*, hal_ids=None, progress=None) -> dict:
    with connect_db(readonly=True) as con:
        rows = con.execute(
            "SELECT a.* FROM articles a WHERE a.status='new' AND NOT EXISTS "
            "(SELECT 1 FROM validations v WHERE v.hal_id=a.hal_id) ORDER BY a.hal_id").fetchall()
    if hal_ids is not None:
        rows = [row for row in rows if row['hal_id'] in hal_ids]
    if not rows:
        return {'predicted': 0, 'skipped': 0}
    version, encoder, classifier, references = load_bundle()
    written = 0
    for index, row in enumerate(rows, 1):
        result = predict_article(encoder, classifier, references,
                                row['title'], row['abstract'], row['keywords'])
        pillar = PILLAR_MAP.get(result['pillar'].strip().lower())
        confidence = float(result['pillar_confidence'])
        similarity = result.get('subaxis_similarity')
        if pillar is None or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError(f"Prédiction invalide pour {row['hal_id']}")
        if similarity is not None and not math.isfinite(float(similarity)):
            raise ValueError(f"Similarité invalide pour {row['hal_id']}")
        with connect_db() as con:
            cursor = con.execute(
                "UPDATE articles SET predicted_pillar=?, pillar_confidence=?, predicted_axis=?, "
                "axis_similarity=?, model_version=?, status='to_review', updated_at=CURRENT_TIMESTAMP "
                "WHERE hal_id=? AND status='new' AND NOT EXISTS "
                "(SELECT 1 FROM validations WHERE hal_id=articles.hal_id)",
                (pillar, confidence, result['subaxis'], similarity, version, row['hal_id']))
            written += cursor.rowcount
        if progress:
            progress(f'Prédiction : {index}/{len(rows)}')
    return {'predicted': written, 'skipped': len(rows) - written, 'model_version': version}


if __name__ == '__main__':
    print(predict_new_articles())
