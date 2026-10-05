import argparse
from pathlib import Path

from src.preprocess import load_and_preprocess_data
from src.encoder import load_encoder, get_or_compute_embeddings
from src.train import train_classifier
from src.build_subaxes import compute_subaxis_references
from src.evaluate_and_deploy import evaluate_and_register


def main(database=None):
    df, train_df, test_df, text_columns = load_and_preprocess_data(database)
    from config.config import CLASSIFIER_PATH
    from dashboard.scripts.db import connect_db
    with connect_db(readonly=True) as con:
        has_active = con.execute("SELECT 1 FROM model_versions WHERE status='active'").fetchone() is not None
    if CLASSIFIER_PATH.exists() or has_active:
        from dashboard.scripts.model_store import active_bundle
        from sentence_transformers import SentenceTransformer
        _, paths = active_bundle()
        encoder = SentenceTransformer(str(paths['encoder']), local_files_only=True)
    else:
        encoder = load_encoder()
    train_embeds, test_embeds, cache_hit, embed_time = get_or_compute_embeddings(encoder, train_df, test_df)
    classifier, mode, fit_time = train_classifier(train_embeds, train_df['manual_label'])
    references = compute_subaxis_references(encoder)
    return evaluate_and_register(
        classifier, encoder, test_embeds, test_df, len(df), text_columns,
        cache_hit, embed_time, fit_time, mode, dataset_version=df.attrs['dataset_version'],
        train_ids=train_df['halId_s'].tolist(), references=references)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=Path)
    print(main(parser.parse_args().database))
