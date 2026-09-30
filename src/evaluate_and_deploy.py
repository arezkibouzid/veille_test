import json
import joblib
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from config.config import (
    BEST_METRIC_PATH, CLASSIFIER_PATH, ENCODER_PATH, METADATA_PATH, 
    MODEL_NAME, REGISTERED_MODEL_NAME, CANONICAL_LABELS, RANDOM_STATE
)

def evaluate_and_register(
    classifier, encoder, test_embeddings, test_df, full_df_len, 
    text_columns, cache_hit, embedding_seconds, fit_seconds, training_mode
):
    test_preds = classifier.predict(test_embeddings)
    test_acc = accuracy_score(test_df['manual_label'], test_preds)
    test_macro_f1 = f1_score(test_df['manual_label'], test_preds, average='macro')

    print(f"\n================ EVALUATION REPORT ({training_mode}) ================")
    print(classification_report(test_df['manual_label'], test_preds, zero_division=0))

    previous_best_f1 = 0.0
    if BEST_METRIC_PATH.exists():
        try:
            previous_best_f1 = json.loads(BEST_METRIC_PATH.read_text()).get("best_macro_f1", 0.0)
        except Exception:
            previous_best_f1 = 0.0

    is_new_best = test_macro_f1 > previous_best_f1

    mlflow.set_tracking_uri('sqlite:///mlflow.db')
    mlflow.set_experiment('sequoia-mpnet-sgd-continual')

    with mlflow.start_run(run_name=f"pipeline-{training_mode}"):
        mlflow.log_params({
            "dataset_rows": full_df_len,
            "learning_mode": training_mode,
            "loss": "log_loss",
            "dvc_tracked": True,
            "embedding_cache_hit": int(cache_hit),
            "random_state": RANDOM_STATE
        })

        mlflow.log_metrics({
            "holdout_accuracy": test_acc,
            "holdout_macro_f1": test_macro_f1,
            "previous_best_macro_f1": previous_best_f1,
            "is_new_best": int(is_new_best),
            "embedding_time_sec": embedding_seconds,
            "fit_time_sec": fit_seconds
        })

        if is_new_best:
            print(f"🎯 NEW BEST MODEL! Macro-F1 ({test_macro_f1:.4f} > {previous_best_f1:.4f}). Exporting & Registering...")
            joblib.dump(classifier, CLASSIFIER_PATH)
            encoder.save(str(ENCODER_PATH))

            BEST_METRIC_PATH.write_text(json.dumps({"best_macro_f1": test_macro_f1}, indent=2))
            
            metadata = {
                'model_name': MODEL_NAME,
                'classifier': 'sklearn.linear_model.SGDClassifier',
                'labels': CANONICAL_LABELS,
                'text_columns': text_columns,
                'last_training_mode': training_mode,
                'total_samples': full_df_len,
                'best_holdout_macro_f1': test_macro_f1
            }
            METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding='utf-8')

            mlflow.sklearn.log_model(
                classifier,
                name="model",
                registered_model_name=REGISTERED_MODEL_NAME
            )
        else:
            print(f"🛑 Metric did not improve baseline ({previous_best_f1:.4f}). Disk artifacts untouched.")
            mlflow.sklearn.log_model(classifier, name="candidate_model")
