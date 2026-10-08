"""Identity checks and conservative deduplication for the editorial report."""
from difflib import SequenceMatcher
import re
import unicodedata


def normalize_doi(value):
    value = str(value or '').strip().lower()
    value = re.sub(r'^https?://(?:dx\.)?doi\.org/|^doi:\s*', '', value)
    return value.rstrip('.,;') if value.startswith('10.') else ''


def normalized_title(value):
    value = unicodedata.normalize('NFKD', str(value or '').casefold())
    value = ''.join(c for c in value if not unicodedata.combining(c))
    return ' '.join(re.sub(r'[^\w\s]', ' ', value).split())


def title_agreement(local, remote):
    left = [normalized_title(s) for s in re.split(r'\||\s+/\s+', str(local or ''))]
    right = [normalized_title(s) for s in re.split(r'\||\s+/\s+', str(remote or ''))]
    return max((SequenceMatcher(None, a, b).ratio() for a in left for b in right if a and b), default=0)


def identity_supported(item):
    """Require saved evidence, not just a legacy 'matched' flag or score."""
    if item.get('openalex_match_status') != 'matched':
        return False
    if not re.fullmatch(r'(?:https://openalex.org/)?W\d+/?', str(item.get('openalex_id') or '')):
        return False
    local_doi, remote_doi = normalize_doi(item.get('doi')), normalize_doi(item.get('openalex_doi'))
    if local_doi and remote_doi and local_doi != remote_doi:
        return False
    similarity = title_agreement(item.get('title'), item.get('openalex_title'))
    try:
        year_gap = abs(int(item['year']) - int(item['openalex_year']))
    except (KeyError, ValueError, TypeError, OverflowError):
        year_gap = None
    if local_doi and local_doi == remote_doi:
        return similarity >= .90 and (year_gap is None or year_gap <= 1)
    return similarity >= .98 and year_gap is not None and year_gap <= 1


def citation_value(value):
    # Never silently truncate a float, accept a boolean, or turn unknown into zero.
    if isinstance(value, bool) or not re.fullmatch(r'\d+', str(value)):
        return None
    return int(value)


def unique_publications(articles):
    """Merge DOI/exact title-year equivalents without merging conflicting DOIs."""
    groups, doi_index, title_index, work_index = [], {}, {}, {}
    for item in sorted(articles, key=lambda a: a['hal_id']):
        doi = normalize_doi(item.get('doi'))
        title = normalized_title(item.get('title'))
        title_key = (title, item.get('year')) if title and item.get('year') else None
        work = str(item.get('openalex_id') or '').rstrip('/').split('/')[-1] if identity_supported(item) else ''
        index = doi_index.get(doi) if doi else None
        candidates = [title_index.get(title_key), work_index.get(work)]
        if index is None:
            for candidate in candidates:
                if candidate is not None:
                    existing_dois = {normalize_doi(a.get('doi')) for a in groups[candidate]} - {''}
                    if not doi or not existing_dois or existing_dois == {doi}:
                        index = candidate
                        break
        if index is None:
            index = len(groups)
            groups.append([])
        groups[index].append(item)
        if doi:
            doi_index[doi] = index
        if title_key:
            title_index.setdefault(title_key, index)
        if work:
            work_index.setdefault(work, index)
    result = []
    for group in groups:
        # Prefer a human axis, then an evidenced match, then the most complete entry.
        selected = min(group, key=lambda a: (a.get('axis_source') != 'human',
                       not identity_supported(a), not bool(a.get('abstract')), a['hal_id']))
        item = dict(selected)
        item['hal_ids'] = [a['hal_id'] for a in group]
        if any('resources' in a for a in group):
            resources = {}
            for article in group:
                for resource in article.get('resources', []):
                    merged = resources.setdefault((resource['kind'], resource['url']), {})
                    merged.update({key: value for key, value in resource.items() if value})
            item['resources'] = [resources[key] for key in sorted(resources)]
        if any('cluster_labs' in a for a in group):
            item['cluster_labs'] = sorted({lab for a in group for lab in a.get('cluster_labs', [])})
        if any('industrial_partners' in a for a in group):
            item['industrial_partners'] = sorted({partner for a in group for partner in a.get('industrial_partners', [])})
        dates = [a['first_validated_at'] for a in group if a.get('first_validated_at')]
        # An old/historical version makes this an established work, not a new entry.
        item['recent'] = all(a['recent'] for a in group)
        item['first_validated_at'] = min(dates) if dates else None
        classifications = {(a['pillar'], a['axis']) for a in group}
        if len(classifications) > 1:
            item['classification_conflict'] = True
            item['axis'] = 'Axe non renseigné'
            item['axis_source'] = 'none'
        values = {a['citations'] for a in group if a.get('citations') is not None}
        if len(values) > 1:
            item['citations'] = None
            item['citations_at'] = None
            item['citation_note'] = 'Valeurs divergentes entre notices'
        result.append(item)
    return result
