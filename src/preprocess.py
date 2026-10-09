import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from config.config import CANONICAL_LABELS, PILLAR_MAP, RANDOM_STATE
from dashboard.scripts.db import connect_db


def load_dataset(database=None):
    with connect_db(readonly=True, path=Path(database) if database else None) as con:
        df = pd.read_sql_query(
            "SELECT a.hal_id AS halId_s, a.title AS title_s, a.abstract, a.keywords, "
            "v.validated_pillar AS manual_label, v.validated_axis, v.validation_source "
            "FROM articles a JOIN validations v ON v.hal_id=a.hal_id "
            "WHERE a.status='validated' ORDER BY a.hal_id", con)
    df['manual_label'] = df['manual_label'].map(
        lambda value: PILLAR_MAP.get(str(value).strip().lower()))
    if df['manual_label'].isna().any():
        raise ValueError('Des validations contiennent un pilier inconnu ; corriger les labels en SQLite.')
    text_columns = ['title_s', 'abstract', 'keywords']
    df['text_for_classification'] = (
        df[text_columns].fillna('').astype(str).agg(' '.join, axis=1)
        .str.replace(r'\s+', ' ', regex=True).str.strip())
    if (df['text_for_classification'] == '').any():
        raise ValueError('Des articles validés ne contiennent aucun texte utilisable.')
    fingerprint = hashlib.sha256(
        df.fillna('').to_json(orient='records', force_ascii=False).encode()).hexdigest()
    df.attrs['dataset_version'] = fingerprint
    return df, text_columns


def in_holdout(hal_id):
    """Affectation stable (~20 %) : un article ne change jamais de côté quand le dataset grandit.

    Un tirage aléatoire redistribuait le holdout à chaque nouvelle validation, si bien que
    le modèle actif était réévalué sur des articles vus à l'entraînement.
    """
    digest = hashlib.sha256(f'{RANDOM_STATE}:{hal_id}'.encode()).digest()
    return int.from_bytes(digest[:8], 'big') % 5 == 0


def load_and_preprocess_data(database=None):
    df, text_columns = load_dataset(database)
    holdout = df['halId_s'].map(in_holdout).astype(bool)
    train_df, test_df = df[~holdout], df[holdout]
    if any(set(part['manual_label']) != set(CANONICAL_LABELS) for part in (train_df, test_df)):
        raise ValueError('Dataset insuffisant : les quatre piliers doivent être représentés '
                         "à la fois dans l'entraînement et dans le holdout.")
    return df, train_df, test_df, text_columns


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=Path)
    parser.add_argument('--manifest', type=Path)
    args = parser.parse_args()
    df, train_df, test_df, columns = load_and_preprocess_data(args.database)
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps({
            'dataset_version': df.attrs['dataset_version'], 'rows': len(df),
            'train_ids': train_df['halId_s'].tolist(), 'test_ids': test_df['halId_s'].tolist(),
            'labels': df['manual_label'].value_counts().to_dict(),
        }, indent=2), encoding='utf-8')
    print(f'Total: {len(df)} | Train: {len(train_df)} | Test: {len(test_df)}')
