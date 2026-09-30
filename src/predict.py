import numpy as np
from config.config import PILLAR_MAP

def predict_article(encoder, classifier, subaxis_references, title='', abstract='', keywords='', journal=''):
    text = ' '.join(str(value or '') for value in [title, abstract, keywords, journal]).strip()
    article_embedding = encoder.encode([text], normalize_embeddings=True)

    pillar_probabilities = classifier.predict_proba(article_embedding)[0]
    pillar_index = int(pillar_probabilities.argmax())
    pillar = str(classifier.classes_[pillar_index])

    references_dict = subaxis_references.get("references", subaxis_references) if isinstance(subaxis_references, dict) else subaxis_references
    normalized_pillar = pillar.strip().lower()
    saxes_key = PILLAR_MAP.get(normalized_pillar, pillar)

    if saxes_key not in references_dict:
        return {
            'pillar': pillar,
            'pillar_confidence': float(pillar_probabilities[pillar_index]),
            'subaxis': 'No class',
            'subaxis_similarity': None
        }

    reference = references_dict[saxes_key]
    subaxis_scores = encoder.similarity(article_embedding, reference['embeddings']).cpu().numpy()[0]
    subaxis_index = int(subaxis_scores.argmax())
    subaxis_probabilities = np.exp(subaxis_scores) / np.sum(np.exp(subaxis_scores))

    return {
        'pillar': pillar,
        'pillar_confidence': float(pillar_probabilities[pillar_index]),
        'subaxis': reference['names'][subaxis_index],
        'subaxis_similarity': float(subaxis_scores[subaxis_index]),
        'subaxis_probabilities': {str(label): float(score) for label, score in zip(reference['names'], subaxis_probabilities)}
    }
