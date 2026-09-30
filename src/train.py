import time
import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier
from config.config import CLASSIFIER_PATH, CANONICAL_LABELS, RANDOM_STATE

def train_classifier(train_embeddings, train_labels):
    fit_started = time.perf_counter()

    if CLASSIFIER_PATH.exists():
        print("\n🧠 Updating existing model with partial_fit...")
        classifier = joblib.load(CLASSIFIER_PATH)
        classifier.partial_fit(
            train_embeddings,
            train_labels,
            classes=np.array(CANONICAL_LABELS)
        )
        training_mode = "incremental_partial_fit"
    else:
        print("\n🚀 Training initial SGD model...")
        classifier = SGDClassifier(
            loss='log_loss',
            max_iter=2000,
            tol=1e-2,
            class_weight='balanced',
            random_state=RANDOM_STATE
        )
        classifier.fit(train_embeddings, train_labels)
        training_mode = "initial_full_fit"

    fit_seconds = time.perf_counter() - fit_started
    return classifier, training_mode, fit_seconds
