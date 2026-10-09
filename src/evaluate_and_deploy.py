"""Evaluate and save an immutable candidate before activating it in SQLite."""
import json
import uuid
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
from config.config import ARTIFACT_DIR, MODEL_NAME, CANONICAL_LABELS, RANDOM_STATE, REGISTERED_MODEL_NAME
from dashboard.scripts.db import connect_db

# Below this, the unseen part of the holdout is too small to rank two models.
MIN_COMPARISON_ROWS = 30


def evaluate_and_register(classifier, encoder, test_embeddings, test_df, full_df_len,
                          text_columns, cache_hit, embedding_seconds, fit_seconds,
                          training_mode, *, dataset_version, train_ids, references):
    predictions = classifier.predict(test_embeddings)
    metrics = {
        'holdout_accuracy': float(accuracy_score(test_df['manual_label'], predictions)),
        'holdout_macro_f1': float(f1_score(test_df['manual_label'], predictions, average='macro')),
        'embedding_time_sec': embedding_seconds, 'fit_time_sec': fit_seconds,
    }
    print(classification_report(test_df['manual_label'], predictions, zero_division=0))
    with connect_db(readonly=True) as con:
        previous = con.execute(
            "SELECT training_hal_ids_json FROM model_versions WHERE status='active'").fetchone()
    # Evaluate both classifiers on the same holdout rows, leaving out those the
    # active model was trained on: they would inflate its score and block activation.
    previous_score = candidate_score = None
    comparison = None
    from config.config import CLASSIFIER_PATH
    if previous or CLASSIFIER_PATH.exists():
        from dashboard.scripts.model_store import active_bundle
        _, paths = active_bundle()
        active_classifier = joblib.load(paths['classifier'])
        # main uses the active encoder for both classifiers; reuse its holdout
        # embeddings rather than allocating a second 420 MB encoder on the VM.
        seen = set(json.loads((previous['training_hal_ids_json'] if previous else None) or '[]'))
        mask = ~test_df['halId_s'].isin(seen).to_numpy()
        leak_free = bool(seen) and int(mask.sum()) >= MIN_COMPARISON_ROWS
        if not leak_free:
            mask = np.ones(len(test_df), dtype=bool)
        truth = test_df['manual_label'].to_numpy()[mask]
        baseline_predictions = active_classifier.predict(test_embeddings[mask])
        from config.config import PILLAR_MAP
        baseline_predictions = [PILLAR_MAP.get(str(label).strip().lower(), str(label))
                                for label in baseline_predictions]
        scores = [float(f1_score(truth, labels, average='macro', labels=CANONICAL_LABELS, zero_division=0))
                  for labels in (baseline_predictions, predictions[mask])]
        previous_score, candidate_score = scores
        comparison = {'rows': int(mask.sum()), 'excludes_active_training_rows': leak_free,
                      'active_macro_f1': previous_score, 'candidate_macro_f1': candidate_score}
    activate = previous_score is None or candidate_score > previous_score
    version = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:8]
    directory = ARTIFACT_DIR / 'versions' / version
    directory.mkdir(parents=True, exist_ok=False)
    joblib.dump(classifier, directory / 'sgd_classifier.joblib')
    joblib.dump({'references': references}, directory / 'subaxis_references.joblib')
    encoder.save(str(directory / 'mpnet_encoder'))
    metadata = {
        'version': version, 'model_name': MODEL_NAME, 'labels': CANONICAL_LABELS,
        'text_columns': text_columns, 'last_training_mode': training_mode,
        'total_samples': full_df_len, 'dataset_version': dataset_version,
        'train_ids': train_ids, 'test_ids': test_df['halId_s'].tolist(),
        'embedding_cache_hit': bool(cache_hit), 'random_state': RANDOM_STATE,
        'metrics': metrics, 'previous_comparable_macro_f1': previous_score,
        'comparison': comparison,
    }
    (directory / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    (directory / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    with connect_db() as con:
        con.execute('BEGIN IMMEDIATE')
        if activate:
            con.execute("UPDATE model_versions SET status='archived' WHERE status='active'")
        con.execute(
            'INSERT INTO model_versions(version,status,dataset_version,metrics_json,artifact_path,training_hal_ids_json) '
            'VALUES (?,?,?,?,?,?)',
            (version, 'active' if activate else 'candidate', dataset_version,
             json.dumps(metrics), str(directory.relative_to(ARTIFACT_DIR)), json.dumps(train_ids)))
    # SQLite and the local bundle remain sufficient when MLflow is not installed.
    mlflow_status = 'not_installed'
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError:
        pass
    else:
        try:
            mlflow.set_tracking_uri(f'sqlite:///{ARTIFACT_DIR / "mlflow.db"}')
            mlflow.set_experiment('sequoia-mpnet-sgd-continual')
            with mlflow.start_run(run_name=version):
                mlflow.log_params({'dataset_version': dataset_version, 'model_version': version,
                                   'learning_mode': training_mode, 'dataset_rows': full_df_len})
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(classifier, name='model' if activate else 'candidate_model',
                                        registered_model_name=REGISTERED_MODEL_NAME if activate else None)
            mlflow_status = 'logged'
        except Exception as exc:
            mlflow_status = f'logging_failed: {type(exc).__name__}'
    result = {'model_version': version, 'status': 'active' if activate else 'candidate',
              'dataset_version': dataset_version, 'metrics': metrics, 'comparison': comparison,
              'mlflow': mlflow_status}
    (ARTIFACT_DIR / 'latest_training.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result
