import time
from sklearn.linear_model import SGDClassifier
from config.config import RANDOM_STATE

def train_classifier(train_embeddings, train_labels):
    fit_started = time.perf_counter()

    classifier = SGDClassifier(
        loss='log_loss', max_iter=2000, tol=1e-2,
        class_weight='balanced', random_state=RANDOM_STATE)
    classifier.fit(train_embeddings, train_labels)
    training_mode = "full_fit_sqlite_validations"

    fit_seconds = time.perf_counter() - fit_started
    return classifier, training_mode, fit_seconds
