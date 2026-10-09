import time
import os
import hashlib
import numpy as np
from sentence_transformers import SentenceTransformer
from config.config import ENCODER_PATH, MODEL_NAME, EMBEDDING_CACHE_PATH

def load_encoder():
    ENCODER_PATH.parent.mkdir(parents=True, exist_ok=True)
    if ENCODER_PATH.exists():
        return SentenceTransformer(str(ENCODER_PATH), local_files_only=True)
    encoder = SentenceTransformer(MODEL_NAME)
    encoder.save(str(ENCODER_PATH))
    return encoder

def get_or_compute_embeddings(encoder, train_df, test_df):
    embedding_started = time.perf_counter()
    batch_size = int(os.getenv("SEQUOIA_ENCODING_BATCH_SIZE", "8"))
    if batch_size < 1:
        raise ValueError("SEQUOIA_ENCODING_BATCH_SIZE doit être positif.")
    texts = train_df['text_for_classification'].tolist() + test_df['text_for_classification'].tolist()
    keys = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts]

    # Cache par texte : un réentraînement n'encode que les articles nouveaux ou modifiés.
    cached = {}
    if EMBEDDING_CACHE_PATH.exists():
        with np.load(EMBEDDING_CACHE_PATH, allow_pickle=False) as cache:
            if "keys" in cache and str(cache["model_name"]) == MODEL_NAME:
                cached = dict(zip(cache["keys"].tolist(), cache["embeddings"]))

    missing = {key: text for key, text in zip(keys, texts) if key not in cached}
    if missing:
        print(f"🔄 Computing {len(missing)}/{len(texts)} embeddings...")
        vectors = encoder.encode(list(missing.values()), normalize_embeddings=True,
                                 show_progress_bar=True, batch_size=batch_size)
        cached.update(zip(missing, vectors))
        EMBEDDING_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        kept = sorted(set(keys))
        np.savez_compressed(
            EMBEDDING_CACHE_PATH,
            keys=np.array(kept),
            embeddings=np.stack([cached[key] for key in kept]),
            model_name=np.array(MODEL_NAME),
        )

    embeddings = np.stack([cached[key] for key in keys])
    embedding_seconds = time.perf_counter() - embedding_started
    return embeddings[:len(train_df)], embeddings[len(train_df):], not missing, embedding_seconds
