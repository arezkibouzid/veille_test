"""HAL dates and declared resources, without inferring ownership or classifications."""
from __future__ import annotations

import argparse
import fcntl
import json
import re
import xml.etree.ElementTree as ET
from datetime import date
from urllib.parse import urlparse, quote

from dashboard.scripts.db import connect_db, database_path, ensure_schema

HAL_RESOURCE_FIELDS = [
    'publicationDate_s', 'producedDate_s', 'relatedData_s', 'relatedSoftware_s',
    'relatedPublication_s', 'softCodeRepository_s', 'softProgrammingLanguage_s',
    'softVersion_s', 'softDevelopmentStatus_s', 'softPlatform_s', 'license_s',
    'number_s', 'country_s', 'seeAlso_s', 'collCode_s',
]


def values(value):
    if value is None:
        return []
    return [str(v).strip() for v in (value if isinstance(value, list) else [value]) if str(v).strip()]


def publication_date(document):
    """Use the stored string, never the ISO search field that pads partial dates."""
    for field in ('publicationDate_s', 'producedDate_s'):
        candidates = values(document.get(field))
        if not candidates:
            continue
        value = candidates[0]
        if not re.fullmatch(r'\d{4}(?:-\d{2}(?:-\d{2})?)?', value):
            continue
        parts = [int(p) for p in value.split('-')]
        try:
            date(parts[0], parts[1] if len(parts) > 1 else 1, parts[2] if len(parts) > 2 else 1)
        except ValueError:
            continue
        return value, {1: 'year', 2: 'month', 3: 'day'}[len(parts)], field
    return None, None, None


def date_label(value):
    if not value:
        return ''
    parts = value.split('-')
    return '/'.join(reversed(parts))


def resource_identity(identifier):
    identifier = str(identifier or '').strip()
    if re.match(r'^https?://(?:dx\.)?doi\.org/', identifier, re.I):
        identifier = re.sub(r'^https?://(?:dx\.)?doi\.org/', '', identifier, flags=re.I)
    elif identifier.lower().startswith('https://zenodo.org/doi/10.'):
        identifier = identifier.split('/doi/', 1)[1]
    if re.fullmatch(r'10\.\d{4,9}/\S+', identifier, re.I):
        identifier = identifier.lower()
        return 'doi:' + identifier, identifier, 'https://doi.org/' + identifier
    match = re.fullmatch(r'(?:https?://[^/]*hal\.science/)?(hal-\d+)(?:v\d+)?/?', identifier, re.I)
    if match:
        identifier = match[1].lower()
        return 'hal:' + identifier, identifier, 'https://hal.science/' + identifier
    if re.fullmatch(r'swh:1:[a-z]{3}:[a-f0-9]{40}(?:;\S+)?', identifier):
        return identifier, identifier, 'https://archive.softwareheritage.org/' + identifier + '/'
    parsed = urlparse(identifier)
    if parsed.scheme in ('http', 'https') and parsed.netloc and not parsed.username and not parsed.password:
        canonical = identifier.rstrip('/')
        return 'url:' + canonical, canonical, canonical
    # Keep an unrecognized identifier as text; never turn it into a clickable URL.
    return 'identifier:' + identifier, identifier, None


def save_documents(connection, documents):
    """Save supplemental HAL metadata only; article labels, texts and scores are untouched."""
    updated = 0
    for document in documents:
        hal_id = document.get('halId_s')
        if not hal_id:
            continue
        in_collection = 'SEQUOIA' in values(document.get('collCode_s'))
        connection.execute('INSERT INTO hal_resource_notices(hal_id,in_collection,metadata_json) VALUES(?,?,?) '
            'ON CONFLICT(hal_id) DO UPDATE SET in_collection=excluded.in_collection, '
            'metadata_json=excluded.metadata_json,fetched_at=CURRENT_TIMESTAMP',
            (hal_id, int(in_collection), json.dumps(document, ensure_ascii=False)))
        pub_date, precision, source = publication_date(document)
        cursor = connection.execute('UPDATE articles SET publication_date=?,publication_date_precision=?, '
            'publication_date_source=?,hal_metadata_updated_at=CURRENT_TIMESTAMP WHERE hal_id=?',
            (pub_date, precision, source, hal_id))
        updated += cursor.rowcount
        connection.execute('DELETE FROM publication_resources WHERE hal_id=?', (hal_id,))
        doc_type = document.get('docType_s')
        if doc_type in ('SOFTWARE', 'PATENT'):
            key, identifier, url = resource_identity(hal_id)
            connection.execute('INSERT INTO hal_resources(resource_id,kind,identifier,url,metadata_hal_id,metadata_json) '
                'VALUES(?,?,?,?,?,?) ON CONFLICT(resource_id) DO UPDATE SET kind=excluded.kind, '
                'metadata_hal_id=excluded.metadata_hal_id,metadata_json=excluded.metadata_json',
                (key, 'software' if doc_type == 'SOFTWARE' else 'patent', identifier, url, hal_id,
                 json.dumps(document, ensure_ascii=False)))
        for field, kind, relation in [('relatedData_s', 'dataset', 'associated_data'),
                                     ('relatedSoftware_s', 'software', 'associated_software'),
                                     ('softCodeRepository_s', 'code', 'code_repository')]:
            for raw in values(document.get(field)):
                key, identifier, url = resource_identity(raw)
                connection.execute('INSERT INTO hal_resources(resource_id,kind,identifier,url) VALUES(?,?,?,?) '
                    'ON CONFLICT(resource_id) DO NOTHING', (key, kind, identifier, url))
                connection.execute('INSERT OR IGNORE INTO publication_resources VALUES(?,?,?,?)',
                                   (hal_id, key, relation, field))
    # Interpret links only once target metadata has been saved, independent of ordering.
    for document in documents:
        hal_id = document.get('halId_s')
        if not hal_id:
            continue
        for identifier in values(document.get('relatedPublication_s')):
            key, target, _ = resource_identity(identifier)
            record = connection.execute('SELECT kind FROM hal_resources WHERE resource_id=?', (key,)).fetchone()
            if record and record['kind'] in ('software', 'patent'):
                connection.execute('INSERT OR IGNORE INTO publication_resources VALUES(?,?,?,?)',
                    (hal_id, key, 'associated_' + record['kind'], 'relatedPublication_s'))
        for relation in document.get('_resource_relations', []):
            key, _, _ = resource_identity(relation['target'])
            record = connection.execute('SELECT kind FROM hal_resources WHERE resource_id=?', (key,)).fetchone()
            if record:
                previous = connection.execute('SELECT source_field FROM publication_resources WHERE hal_id=? AND resource_id=?',
                                              (hal_id, key)).fetchone()
                if previous:
                    connection.execute('DELETE FROM publication_resources WHERE hal_id=? AND resource_id=?', (hal_id, key))
                    connection.execute('INSERT INTO publication_resources VALUES(?,?,?,?)',
                        (hal_id, key, relation['relation'] or 'associated', previous['source_field']))
        for doi, metadata in document.get('_doi_resources', {}).items():
            connection.execute('UPDATE hal_resources SET metadata_json=? WHERE resource_id=? AND metadata_hal_id IS NULL',
                               (json.dumps(metadata, ensure_ascii=False), 'doi:' + doi))
    return {'received': len(documents), 'articles_enriched': updated}


def enrich_tei(documents):
    """Fetch declared relation types and software licences in small HAL export batches."""
    from dashboard.scripts.refresh_hal import _get_with_retry
    selected = {d['halId_s']: d for d in documents if d.get('halId_s') and
                (d.get('docType_s') in ('SOFTWARE', 'PATENT') or d.get('relatedData_s') or d.get('relatedSoftware_s'))}
    ids = sorted(selected)
    ns = {'t': 'http://www.tei-c.org/ns/1.0'}
    for offset in range(0, len(ids), 40):
        batch = [s for s in ids[offset:offset+40] if re.fullmatch(r'hal-\d+', s)]
        if not batch:
            continue
        response = _get_with_retry('https://api.hal.science/search/',
            {'q': 'halId_s:(' + ' OR '.join(batch) + ')', 'rows': len(batch), 'wt': 'xml-tei'})
        root = ET.fromstring(response.content)
        for bibl in root.findall('.//t:biblFull', ns):
            identifier = bibl.find('.//t:idno[@type="halId"]', ns)
            if identifier is None or identifier.text not in selected:
                continue
            doc = selected[identifier.text]
            doc['_resource_relations'] = [{'target': r.get('target'), 'relation': r.get('type'), 'resource_type': r.get('subtype')}
                for r in bibl.findall('.//t:relatedItem', ns) if r.get('target')]
            doc['_licenses'] = [{'label': ''.join(r.itertext()).strip(), 'url': r.get('target')}
                for r in bibl.findall('.//t:licence', ns) if ''.join(r.itertext()).strip()]
            doc['_software_archives'] = values([r.text for r in bibl.findall('.//t:idno[@type="swhid"]', ns) if r.text])
    return documents


def enrich_doi_resources(documents):
    """Resolve declared DOI resources through public DataCite metadata; absence is retained."""
    import requests
    identifiers = {resource_identity(raw)[1] for doc in documents
                   for field in ('relatedData_s', 'relatedSoftware_s') for raw in values(doc.get(field))
                   if resource_identity(raw)[0].startswith('doi:')}
    metadata, failures = {}, []
    for doi in sorted(identifiers):
        try:
            response = requests.get('https://api.datacite.org/dois/' + quote(doi, safe='/'), timeout=12,
                headers={'User-Agent': 'SequoIA-HAL-resource-catalog/1.0'})
            response.raise_for_status()
            attrs = response.json()['data']['attributes']
            if str(attrs.get('doi', '')).lower() != doi:
                raise ValueError('DOI metadata mismatch')
            metadata[doi] = {'title_s': [r['title'] for r in attrs.get('titles', []) if r.get('title')],
                'abstract_s': [r['description'] for r in attrs.get('descriptions', []) if r.get('descriptionType') == 'Abstract'],
                'authFullName_s': [r['name'] for r in attrs.get('creators', []) if r.get('name')],
                'license_s': [r.get('rights') or r.get('rightsIdentifier') for r in attrs.get('rightsList', [])
                              if r.get('rights') or r.get('rightsIdentifier')],
                '_metadata_source': 'DataCite', '_doi': doi}
        except (requests.RequestException, ValueError, KeyError) as exc:
            failures.append({'doi': doi, 'error': str(exc)[:200]})
    for document in documents:
        document['_doi_resources'] = {doi: attrs for doi, attrs in metadata.items()
            if any(resource_identity(raw)[1] == doi for field in ('relatedData_s', 'relatedSoftware_s')
                   for raw in values(document.get(field)))}
    return failures


def fetch_supplemental_documents(existing_ids=()):
    from dashboard.scripts.refresh_hal import _get_with_retry
    fields = ['halId_s', 'docid', 'docType_s', 'title_s', 'abstract_s', 'authFullName_s',
              'labStructName_s', 'uri_s', 'doiId_s'] + HAL_RESOURCE_FIELDS
    url = 'https://api.hal.science/search/'
    documents = []
    cursor = '*'
    while True:
        body = _get_with_retry(url, {'q': 'collCode_s:SEQUOIA', 'fl': ','.join(fields), 'wt': 'json',
            'rows': 1000, 'sort': 'docid asc', 'cursorMark': cursor}).json()
        documents.extend(body['response']['docs'])
        next_cursor = body.get('nextCursorMark')
        if not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor
    found = {d['halId_s'] for d in documents}
    missing = sorted(set(existing_ids) - found)
    # Also enrich historical records no longer stamped in the collection.
    for offset in range(0, len(missing), 75):
        ids = [s for s in missing[offset:offset+75] if re.fullmatch(r'hal-\d+', s)]
        if ids:
            body = _get_with_retry(url, {'q': 'halId_s:(' + ' OR '.join(ids) + ')',
                 'fl': ','.join(fields), 'wt': 'json', 'rows': len(ids)}).json()
            documents.extend(body['response']['docs'])
    # Resolve HAL software/patent links even when their notice is outside SEQUOIA.
    found = {d['halId_s'] for d in documents}
    related = set()
    for document in documents:
        for field in ('relatedSoftware_s', 'relatedData_s', 'relatedPublication_s'):
            for identifier in values(document.get(field)):
                key, normalized, _ = resource_identity(identifier)
                if key.startswith('hal:') and normalized not in found:
                    related.add(normalized)
    related = sorted(related)
    for offset in range(0, len(related), 75):
        ids = related[offset:offset+75]
        body = _get_with_retry(url, {'q': 'halId_s:(' + ' OR '.join(ids) + ')',
             'fl': ','.join(fields), 'wt': 'json', 'rows': len(ids)}).json()
        documents.extend(body['response']['docs'])
    return enrich_tei(documents)


def backfill(progress=print):
    with connect_db(readonly=True) as con:
        ids = [r[0] for r in con.execute('SELECT hal_id FROM articles')]
    progress('Lecture des dates et ressources HAL')
    documents = fetch_supplemental_documents(ids)
    failures = enrich_doi_resources(documents)
    # No network call takes place within the write transaction.
    with connect_db() as con:
        ensure_schema(con)
        result = save_documents(con, documents)
    result['missing_hal_ids'] = sorted(set(ids) - {d['halId_s'] for d in documents})
    result['resource_metadata_failures'] = failures
    return result


if __name__ == '__main__':
    argparse.ArgumentParser(description='Complète les dates et ressources HAL sans reclassifier les articles.').parse_args()
    with database_path().resolve().with_suffix('.pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps(backfill(), ensure_ascii=False), flush=True)
