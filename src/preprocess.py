import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
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


def load_and_preprocess_data(database=None):
    df, text_columns = load_dataset(database)
    counts = df['manual_label'].value_counts()
    test_rows = math.ceil(len(df) * 0.20)
    if (set(counts.index) != set(CANONICAL_LABELS) or counts.min() < 2
            or test_rows < len(counts) or len(df) - test_rows < len(counts)):
        raise ValueError('Dataset insuffisant : les quatre piliers doivent être représentés, '
                         'avec au moins deux validations par pilier et un holdout stratifié possible.')
    train_df, test_df = train_test_split(
        df, test_size=0.20, stratify=df['manual_label'], random_state=RANDOM_STATE)
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
