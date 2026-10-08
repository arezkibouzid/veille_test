import numpy as np
from config.config import PILLAR_MAP, SEQUOIA_SAXES

def predict_article(encoder, classifier, subaxis_references, title='', abstract='', keywords='', *, manual_pillar=None):
    supplied = str(manual_pillar or '').strip()
    confidence = None
    if supplied:
        pillar = PILLAR_MAP.get(supplied.lower())
        if pillar is None:
            raise ValueError('Pilier manuel inconnu')
        if pillar == 'No class':
            return {'pillar': pillar, 'pillar_confidence': None, 'subaxis': None, 'subaxis_similarity': None}
    text = ' '.join(str(value or '') for value in [title, abstract, keywords]).strip()
    article_embedding = encoder.encode([text], normalize_embeddings=True)

    if not supplied:
        pillar_probabilities = classifier.predict_proba(article_embedding)[0]
        pillar_index = int(pillar_probabilities.argmax())
        pillar = PILLAR_MAP.get(str(classifier.classes_[pillar_index]).strip().lower())
        if pillar is None:
            raise ValueError('Pilier prédit inconnu')
        confidence = float(pillar_probabilities[pillar_index])

    references_dict = subaxis_references.get("references", subaxis_references) if isinstance(subaxis_references, dict) else subaxis_references
    normalized_pillar = pillar.strip().lower()
    saxes_key = PILLAR_MAP.get(normalized_pillar, pillar)

    if pillar == 'No class':
        return {
            'pillar': pillar,
            'pillar_confidence': confidence,
            'subaxis': None,
            'subaxis_similarity': None
        }

    if saxes_key not in references_dict:
        raise ValueError(f'Reférences des axes absentes pour {saxes_key}')
    reference = references_dict[saxes_key]
    if not reference['names'] or any(name not in SEQUOIA_SAXES[saxes_key] for name in reference['names']):
        raise ValueError(f'Références des axes invalides pour {saxes_key}')
    subaxis_scores = encoder.similarity(article_embedding, reference['embeddings']).cpu().numpy()[0]
    if len(subaxis_scores) != len(reference['names']) or not np.isfinite(subaxis_scores).all():
        raise ValueError('Scores des axes invalides')
    subaxis_index = int(subaxis_scores.argmax())
    weights = np.exp(subaxis_scores - subaxis_scores.max())
    subaxis_probabilities = weights / weights.sum()

    return {
        'pillar': pillar,
        'pillar_confidence': confidence,
        'subaxis': reference['names'][subaxis_index],
        'subaxis_similarity': float(subaxis_scores[subaxis_index]),
        'subaxis_probabilities': {str(label): float(score) for label, score in zip(reference['names'], subaxis_probabilities)}
    }
