"""Bounded scientific analysis grounded in axes, keywords and source abstracts."""
from collections import Counter
from html import escape, unescape
from html.parser import HTMLParser
import re

from config.config import SEQUOIA_SAXES


# Reading angles, not claims about a publication's results or industrial maturity.
AXIS_PURPOSES = {
    'Theoretical and Conceptual Foundations of AI': 'comprendre les propriétés des modèles, leurs capacités de généralisation et leurs limites',
    'Signal, Image and Language': 'extraire et exploiter l’information portée par les signaux, les images et le langage',
    'Frugal AI, Edge AI and Hardware Architectures': 'réduire les contraintes de calcul et de mémoire et étudier le déploiement sur des architectures embarquées',
    'Formal Evaluation of ML Technologies': 'évaluer et vérifier les propriétés des méthodes d’apprentissage',
    'Symbolic / Statistical Hybridization': 'articuler apprentissage statistique, connaissances et raisonnement',
    'Human–AI Interaction and Explainability': 'rendre les décisions des modèles compréhensibles et étudier leur interaction avec les utilisateurs',
    'Acceptability, Responsibility, Regulation and Ethics': 'examiner les conditions d’acceptabilité, de responsabilité et d’encadrement des usages de l’IA',
    'AI Security': 'étudier les vulnérabilités des modèles et les moyens de les protéger',
    'AI for Cybersecurity': 'mobiliser l’IA pour analyser les menaces et les événements de sécurité',
    'Intelligence and Information Security': 'analyser la fiabilité, la circulation et la protection de l’information',
    'Intelligent and Secure Networks': 'étudier l’orchestration, les performances et la sécurité des réseaux',
    'Autonomous Robotics and Interaction': 'étudier la perception, la décision et l’interaction des systèmes robotiques',
    'Physics-Informed AI': 'articuler apprentissage, contraintes physiques et estimation de systèmes environnementaux',
    'Observation, Remote Sensing and Data Integration': 'exploiter les observations et combiner des sources de données pour décrire l’environnement',
    'Ocean Digital Twin': 'représenter et étudier les systèmes océaniques par la modélisation numérique',
}


def source_link(item, label=None):
    from dashboard.scripts.veille_report import article_url, shortened
    label = escape(label or shortened(item.get('title') or item['hal_id'], 160))
    url = article_url(item)
    return f'<a href="{escape(url, quote=True)}">{label}</a>' if url else label


class PlainAbstract(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, text):
        self.parts.append(text)

    def handle_endtag(self, tag):
        if tag in ('p', 'div', 'br'):
            self.parts.append(' ')


def abstract_excerpt(item, limit=360):
    """Quote source text; never infer a result from the title or citation count."""
    from dashboard.scripts.veille_report import shortened
    parser = PlainAbstract()
    parser.feed(unescape(str(item.get('abstract') or '')))
    text = ' '.join(''.join(parser.parts).split())
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-ZÀ-Ý])', text)
    contribution = re.compile(r'\b(?:we (?:propose|present|introduce|develop|show)|this (?:paper|work|study)|'
                              r'nous (?:proposons|présentons|montrons)|cet(?:te)? (?:article|étude)|results? show)\b', re.I)
    sentence = next((s for s in sentences if contribution.search(s)), sentences[0] if sentences else '')
    return shortened(sentence, limit)


def synthesis_blocks(articles, year):
    from dashboard.scripts.veille_report import thematic_keywords, select_new_publications, select_publications
    from dashboard.scripts.report_evidence import unique_publications
    blocks = [('page', ''), ('h1', 'Synthèse scientifique')]
    counts = Counter(a['pillar'] for a in articles)
    ranked = counts.most_common()
    if ranked:
        pillar, count = ranked[0]
        blocks.append(('p', f'Le corpus est principalement représenté par <b>{escape(pillar)}</b> '
                       f'({count} publications, {100 * count / max(1, len(articles)):.0f} %). '
                       'La lecture par axe permet de relier les fondements méthodologiques, la sécurité des systèmes '
                       'et les applications environnementales. Les paragraphes suivants distinguent les sujets '
                       'documentés dans le corpus des contributions illustrées par des travaux précis.'))
    for pillar in SEQUOIA_SAXES:
        items = [a for a in articles if a['pillar'] == pillar]
        if not items:
            continue
        if pillar == 'AI, Environment and Ocean':
            blocks.append(('page', ''))
            blocks.append(('h1', 'Synthèse scientifique · Suite'))
        blocks.append(('h2', pillar))
        axes = Counter(a['axis'] for a in items if a['axis'] in SEQUOIA_SAXES[pillar])
        current = sum(a.get('year') == year for a in items)
        blocks.append(('analysis_lead', f'Ce pilier rassemble <b>{len(items)} publications</b>, dont {current} publiées en {year}. '
                       + ('Les deux axes les plus représentés regroupent '
                          f'{sum(n for _, n in axes.most_common(2)) / len(items):.0%} des travaux de ce pilier.'.replace('%', ' %')
                          if len(axes) >= 2 else '')))
        for axis, count in axes.most_common(2):
            group = [a for a in items if a['axis'] == axis]
            topics = thematic_keywords(group)
            text = f'<b>{escape(axis)}.</b> Cet axe vise à {escape(AXIS_PURPOSES.get(axis, "étudier les méthodes et applications documentées"))}. '
            text += f'Il représente {count} publications ({100 * count / len(items):.0f} % du pilier). '
            if topics:
                text += 'Les sujets récurrents comprennent ' + ', '.join(
                    f'{escape(word)} ({n} publications)' for word, n in topics[:3]) + '. '
            blocks.append(('analysis', text))
            with_abstract = [a for a in unique_publications(group) if a.get('abstract')]
            candidates = select_new_publications(with_abstract, year, limit=1) or select_publications(with_abstract, limit=1)
            if not candidates:
                candidates = sorted(with_abstract, key=lambda a: (-(a.get('year') or 0), a['hal_id']))[:1]
            if candidates:
                item = candidates[0]
                qualifier = f'Contribution de {year}' if item.get('year') == year else 'Travail de référence dans cet axe'
                blocks.append(('analysis_source', f'<b>{qualifier} :</b> {source_link(item)}.'))
                blocks.append(('excerpt', f'Extrait du résumé : « {escape(abstract_excerpt(item))} »'))
        new = select_new_publications(unique_publications(items), year, limit=3)
        if new:
            new_axes = list(dict.fromkeys(a['axis'] for a in new if a['axis'] in SEQUOIA_SAXES[pillar]))
            blocks.append(('p', f'<b>Lecture des nouveautés.</b> Les travaux sélectionnés pour {year} portent sur '
                           + '; '.join(escape(axis) for axis in new_axes) + '. '
                           'Ils sont présentés avec leurs références et leurs ressources dans la fiche du pilier.'))
    # Shared keywords establish thematic proximity, not an invented collaboration.
    shared = {}
    for pillar in SEQUOIA_SAXES:
        for word, n in thematic_keywords([a for a in articles if a['pillar'] == pillar], limit=None):
            shared.setdefault(word.casefold(), []).append((pillar, word, n))
    overlaps = sorted([rows for rows in shared.values() if len(rows) >= 2],
                      key=lambda rows: (-sum(r[2] for r in rows), rows[0][1]))[:3]
    if overlaps:
        blocks.append(('h2', 'Convergences entre piliers'))
        for rows in overlaps:
            blocks.append(('p', f'<b>{escape(rows[0][1])}.</b> Ce sujet est documenté dans '
                           + ' et '.join(f'{escape(p)} ({n} publications)' for p, _, n in rows)
                           + '. Ce rapprochement thématique suggère une piste de dialogue entre les équipes concernées.'))
    return blocks


def partner_blocks(articles, year, catalog_resources=()):
    from dashboard.scripts.veille_report import safe_url, shortened
    from dashboard.scripts.report_evidence import unique_publications
    blocks = []
    selected, used = [], set()
    # First cover each pillar, then offer at most two complementary axes.
    ranked = sorted(unique_publications(articles), key=lambda a: (
        not bool(a.get('resources')), a.get('year') != year, not a['recent'],
        -(a.get('citations') or 0), -(a.get('year') or 0), a['hal_id']))
    for pillar in SEQUOIA_SAXES:
        item = next((a for a in ranked if a['pillar'] == pillar and a.get('abstract')), None)
        if item:
            selected.append(item); used.add((pillar, item['axis']))
    for item in ranked:
        if len(selected) >= 5:
            break
        if item.get('resources') and item.get('abstract') and (item['pillar'], item['axis']) not in used:
            selected.append(item); used.add((item['pillar'], item['axis']))
    if not selected:
        return blocks
    blocks.extend([('page', ''), ('h1', 'Perspectives pour les partenaires'),
                   ('p', 'Les pistes suivantes proposent des points de départ pour une expérimentation ou un projet commun. '
                    'Elles s’appuient sur des travaux identifiés ; elles ne constituent pas des offres de transfert déjà établies.')])
    for item in selected:
        axis = item['axis']
        blocks.append(('h2', item['pillar'] + ' · ' + axis))
        purpose = AXIS_PURPOSES.get(axis)
        if purpose:
            blocks.append(('p', f'<b>Besoin à explorer :</b> {escape(purpose.capitalize())}.'))
        blocks.append(('analysis_source', f'<b>Travail support :</b> {source_link(item)}.'))
        blocks.append(('excerpt', f'Apport documenté dans le résumé : « {escape(abstract_excerpt(item, 260))} »'))
        resources = item.get('resources', [])[:2]
        kinds = {r['kind'] for r in resources}
        if 'software' in kinds or 'code' in kinds:
            action = 'Examiner le logiciel ou le code disponible, puis définir un test sur un cas d’usage du partenaire, avec des critères de performance et de reproductibilité.'
        elif 'dataset' in kinds:
            action = 'Examiner les données associées et leur couverture, puis construire un protocole d’évaluation sur des données représentatives du besoin du partenaire.'
        elif 'patent' in kinds:
            action = 'Examiner le périmètre du brevet avec l’équipe concernée et le service de valorisation, puis préciser le cas d’usage et les modalités d’une étude de faisabilité.'
        else:
            action = 'Échanger avec l’équipe sur la méthode décrite, puis définir une étude de faisabilité ou une comparaison avec les pratiques du partenaire sur un cas précis.'
        blocks.append(('p', '<b>Collaboration proposée :</b> ' + action))
        for r in resources:
            url = safe_url(r.get('url'))
            if not url:
                continue
            labels = {'software': 'Logiciel', 'dataset': 'Jeu de données', 'code': 'Code source', 'patent': 'Brevet'}
            title = shortened(r.get('title') or labels.get(r['kind'], 'Ressource'), 120)
            license_text = f' · Licence déclarée : {escape(r["license"])}' if r.get('license') else ''
            blocks.append(('p', f'<b>Ressource à explorer :</b> <a href="{escape(url, quote=True)}">{escape(title)}</a>{license_text}'))
        if item.get('cluster_labs'):
            blocks.append(('p', '<b>Laboratoires associés au travail :</b> ' + ' · '.join(map(escape, item['cluster_labs']))))
    used_resources = {r['url'] for item in selected for r in item.get('resources', [])}
    tools = [r for r in catalog_resources if r['kind'] == 'software' and r.get('description')
             and r.get('url') not in used_resources]
    tools.sort(key=lambda r: (not bool(r.get('access_links')), not bool(r.get('license')), r['id']))
    for resource in tools[:min(1, max(0, 5 - len(selected)))]:
        url = safe_url(resource.get('url'))
        if not url:
            continue
        blocks.append(('h2', 'Logiciel à explorer · ' + escape(shortened(resource['title'], 95))))
        blocks.append(('analysis_source', f'<b>Ressource :</b> <a href="{escape(url, quote=True)}">{escape(resource["title"])}</a>'))
        blocks.append(('excerpt', 'Description de la ressource : « ' + escape(abstract_excerpt({'abstract': resource['description']}, 260)) + ' »'))
        blocks.append(('p', '<b>Collaboration proposée :</b> Évaluer les fonctions documentées sur un cas d’usage du partenaire et préciser les adaptations nécessaires avec l’équipe concernée.'))
        if resource.get('license'):
            blocks.append(('p', '<b>Licence déclarée :</b> ' + escape(resource['license'])))
        if resource.get('cluster_labs'):
            blocks.append(('p', '<b>Laboratoires référencés :</b> ' + ' · '.join(map(escape, resource['cluster_labs']))))
    return blocks
