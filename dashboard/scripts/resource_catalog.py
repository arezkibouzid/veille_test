"""Public resource catalog: explicit HAL evidence and links, no ownership inference."""
import json
from config.config import SEQUOIA_SAXES
from dashboard.scripts.hal_resources import values, publication_date, date_label, resource_identity
from dashboard.scripts.data_rules import normalize_lab

KIND_LABELS = {'dataset': 'Jeu de données', 'software': 'Logiciel', 'code': 'Code source', 'patent': 'Brevet'}
RELATION_LABELS = {'Cites': 'Cite cette ressource', 'References': 'Référence cette ressource',
    'IsSupplementTo': 'Complément de cette ressource', 'IsSupplementedBy': 'Complété par cette ressource',
    'Documents': 'Documente cette ressource', 'IsDocumentedBy': 'Documenté par cette ressource',
    'IsPartOf': 'Fait partie de cette ressource', 'HasPart': 'Inclut cette ressource',
    'associated_data': 'Données associées', 'associated_software': 'Logiciel associé',
    'associated_patent': 'Brevet associé', 'code_repository': 'Dépôt du code'}


def catalog(connection):
    from dashboard.scripts.validation_api import public_article_sql
    notices = {r['hal_id']: {'document': json.loads(r['metadata_json']), 'in_collection': bool(r['in_collection'])}
               for r in connection.execute('SELECT * FROM hal_resource_notices')}
    publications = {r['publication_id']: dict(r) for r in connection.execute(public_article_sql())}
    cluster_labs = {}
    for row in connection.execute('SELECT hal_id,lab FROM publication_labs_cluster ORDER BY lab'):
        cluster_labs.setdefault(row['hal_id'], []).append(row['lab'])
    resources = {r['resource_id']: dict(r) for r in connection.execute('SELECT * FROM hal_resources')}
    links = [dict(r) for r in connection.execute('SELECT * FROM publication_resources ORDER BY hal_id,resource_id')]
    standalone = {hal_id for hal_id, n in notices.items()
                  if n['in_collection'] and n['document'].get('docType_s') in ('SOFTWARE', 'PATENT')}
    sources = {}
    for link in links:
        pub = publications.get(link['hal_id'])
        if (pub and pub['pillar'] in SEQUOIA_SAXES) or link['hal_id'] in standalone:
            sources.setdefault(link['resource_id'], []).append(link)
    eligible = set(sources) | {'hal:' + hal_id for hal_id in standalone}
    items = []
    for key in sorted(eligible):
        resource = resources.get(key)
        if not resource:
            continue
        identifier = resource['identifier']
        own_id = identifier if key.startswith('hal:') else resource['metadata_hal_id']
        notice = notices.get(own_id, {})
        document = notice.get('document') or json.loads(resource['metadata_json'])
        is_collection = own_id in standalone
        # Code repositories already appear as access links on the software card.
        resource_sources = sources.get(key, [])
        if resource['kind'] == 'code' and resource_sources and all(l['hal_id'] in standalone for l in resource_sources):
            continue
        associated = []
        for link in resource_sources:
            pub = publications.get(link['hal_id'])
            if pub:
                associated.append({'hal_id': link['hal_id'], 'title': pub['title'], 'url': 'https://hal.science/' + link['hal_id'],
                    'pillar': pub['pillar'], 'axis': pub['axis'], 'labs': pub['labs'],
                    'cluster_labs': cluster_labs.get(link['hal_id'], []), 'relation': link['relation'],
                    'relation_label': RELATION_LABELS.get(link['relation'], link['relation'])})
            else:
                parent = notices[link['hal_id']]['document']
                associated.append({'hal_id': link['hal_id'], 'title': ' / '.join(values(parent.get('title_s'))),
                    'url': 'https://hal.science/' + link['hal_id'], 'pillar': '', 'axis': '',
                    'labs': ' | '.join(values(parent.get('labStructName_s'))), 'relation': link['relation'],
                    'relation_label': RELATION_LABELS.get(link['relation'], link['relation'])})
        own_publication = publications.get(own_id)
        scientific = associated + ([own_publication] if own_publication else [])
        normalized_labs = sorted({lab for raw in values(document.get('labStructName_s'))
                                  if (lab := normalize_lab(raw))}
                                 | set(cluster_labs.get(own_id, []))
                                 | {lab for pub in associated for lab in pub.get('cluster_labs', [])})
        pillars = sorted({a.get('pillar') for a in scientific if a.get('pillar') in SEQUOIA_SAXES})
        axes = sorted({a.get('axis') for a in scientific if a.get('pillar') in SEQUOIA_SAXES and a.get('axis')})
        pub_date, precision, _ = publication_date(document)
        access = []
        for archive_identifier in document.get('_software_archives', []):
            _, _, url = resource_identity(archive_identifier)
            if url:
                access.append({'label': 'Archive Software Heritage', 'url': url})
        for field, label in [('softCodeRepository_s', 'Code / dépôt'), ('seeAlso_s', 'Lien complémentaire')]:
            for raw in values(document.get(field)):
                _, _, url = resource_identity(raw)
                if url:
                    access.append({'label': label, 'url': url})
        items.append({'id': key, 'kind': resource['kind'], 'kind_label': KIND_LABELS[resource['kind']],
            'identifier': identifier, 'url': resource['url'],
            'title': ' / '.join(values(document.get('title_s'))) or KIND_LABELS[resource['kind']] + ' — ' + identifier,
            'description': ' '.join(values(document.get('abstract_s')))[:600],
            'authors': ', '.join(values(document.get('authFullName_s'))),
            'labs': values(document.get('labStructName_s')),
            'cluster_labs': normalized_labs,
            'publication_date': pub_date, 'date_label': date_label(pub_date), 'date_precision': precision,
            'origin': 'collection' if is_collection else 'associated',
            'origin_label': 'Production référencée dans SEQUOIA' if is_collection else 'Ressource associée — origine à préciser',
            'metadata_source': document.get('_metadata_source') or ('HAL' if document else ''),
            'license': ' | '.join(values(document.get('license_s')) or [r['label'] for r in document.get('_licenses', [])]),
            'version': ' | '.join(values(document.get('softVersion_s'))),
            'languages': values(document.get('softProgrammingLanguage_s')),
            'patent_number': ' | '.join(values(document.get('number_s'))) if resource['kind'] == 'patent' else '',
            'country': document.get('country_s') if resource['kind'] == 'patent' else '',
            'pillars': pillars, 'axes': axes, 'publications': associated, 'access_links': access})
    lab_options = sorted({lab for hal_id, labs in cluster_labs.items() if hal_id in publications for lab in labs}
                         | {lab for item in items for lab in item['cluster_labs']})
    return {'count': len(items), 'items': items, 'lab_options': lab_options}
