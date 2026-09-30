import time
import json
import hashlib
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from config.config import ENCODER_PATH, MODEL_NAME, EMBEDDING_CACHE_PATH

def load_encoder():
    if ENCODER_PATH.exists():
        return SentenceTransformer(str(Path.cwd() / ENCODER_PATH), local_files_only=True)
    encoder = SentenceTransformer(MODEL_NAME)
    encoder.save(str(ENCODER_PATH))
    return encoder

def get_or_compute_embeddings(encoder, train_df, test_df):
    embedding_started = time.perf_counter()
    train_texts = train_df['text_for_classification'].tolist()
    test_texts = test_df['text_for_classification'].tolist()

    fingerprint_data = {
        "model_name": MODEL_NAME,
        "normalize_embeddings": True,
        "train_texts": train_texts,
        "test_texts": test_texts,
    }
    current_fingerprint = hashlib.sha256(
        json.dumps(fingerprint_data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    cache_hit = False
    if EMBEDDING_CACHE_PATH.exists():
        with np.load(EMBEDDING_CACHE_PATH, allow_pickle=False) as cache:
            if str(cache.get("fingerprint", "")) == current_fingerprint:
                train_embeddings = cache["train_embeddings"]
                test_embeddings = cache["test_embeddings"]
                cache_hit = True

    if not cache_hit:
        print("🔄 Cache miss. Computing embeddings...")
        train_embeddings = encoder.encode(train_texts, normalize_embeddings=True, show_progress_bar=True)
        test_embeddings = encoder.encode(test_texts, normalize_embeddings=True, show_progress_bar=True)
        np.savez_compressed(
            EMBEDDING_CACHE_PATH,
            train_embeddings=train_embeddings,
            test_embeddings=test_embeddings,
            fingerprint=np.array(current_fingerprint)
        )

    embedding_seconds = time.perf_counter() - embedding_started
    return train_embeddings, test_embeddings, cache_hit, embedding_seconds
