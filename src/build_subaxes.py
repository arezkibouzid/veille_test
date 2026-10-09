import json
import joblib
import hashlib
from config.config import SUBAXIS_PATH, MODEL_NAME, SEQUOIA_SAXES

def compute_subaxis_references(encoder):
    subaxis_fingerprint_data = {
        "model_name": MODEL_NAME,
        "sequoia_saxes": SEQUOIA_SAXES,
    }
    current_subaxis_fingerprint = hashlib.sha256(
        json.dumps(subaxis_fingerprint_data, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()

    if SUBAXIS_PATH.exists():
        try:
            cached_data = joblib.load(SUBAXIS_PATH)
            if isinstance(cached_data, dict) and cached_data.get("_fingerprint") == current_subaxis_fingerprint:
                print(f"✅ Loaded valid cached sub-axis references")
                return cached_data["references"]
        except Exception as e:
            print(f"⚠️ Failed to read cache: {e}")

    print("🔄 Recomputing sub-axis embeddings...")
    subaxis_references = {}
    for pillar, subaxes in SEQUOIA_SAXES.items():
        subaxis_names = list(subaxes)
        subaxis_texts = [" ".join(keywords) for keywords in subaxes.values()]

        subaxis_references[pillar] = {
            'names': subaxis_names,
            'embeddings': encoder.encode(subaxis_texts, normalize_embeddings=True, show_progress_bar=False),
        }

    SUBAXIS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"_fingerprint": current_subaxis_fingerprint, "references": subaxis_references}, SUBAXIS_PATH)
    return subaxis_references
