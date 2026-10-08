"""Rapports immuables du corpus humainement validé, sans réseau ni écriture ML."""
from __future__ import annotations

import calendar
from collections import Counter
from datetime import datetime, timezone
import hashlib
from html import escape
import json
import os
from pathlib import Path
import re
import shutil
import uuid
from urllib.parse import urlsplit

from config.config import PILLAR_MAP, SEQUOIA_SAXES
from dashboard.scripts.db import connect_db, database_path
from dashboard.scripts.report_evidence import identity_supported, citation_value, unique_publications
from dashboard.scripts.hal_resources import date_label

FORMAT_VERSION = 12
UNASSIGNED_AXIS = 'Axe non renseigné'


def reports_directory():
    return Path(os.getenv('SEQUOIA_REPORT_DIR') or database_path().resolve().parent / 'reports')


def three_months_before(now):
    month_index = now.year * 12 + now.month - 1 - 3
    year, month_zero = divmod(month_index, 12)
    month = month_zero + 1
    return now.replace(year=year, month=month,
                       day=min(now.day, calendar.monthrange(year, month)[1]))


def parsed_date(value):
    if not value:
        return None
    try:
        result = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result.astimezone(timezone.utc)
    except ValueError:
        return None


def display_date(value):
    date = parsed_date(value)
    return date.strftime('%d/%m/%Y') if date else 'date inconnue'


def canonical(value):
    return PILLAR_MAP.get(str(value or '').strip().lower())


def load_corpus(now, *, include_resources=False):
    """Une transaction de lecture couvre corpus et référence précédente."""
    with connect_db(readonly=True) as con:
        con.execute('BEGIN')
        rows = con.execute("""
            SELECT a.hal_id, a.title, a.authors, a.abstract, a.year, a.publication_date,
                   a.publication_date_precision,
                   a.journal, a.conference, a.doi, a.url, a.keywords,
                   a.openalex_id, a.openalex_title, a.openalex_doi, a.openalex_year,
                   a.openalex_match_status,
                   a.predicted_pillar, a.predicted_axis, a.axis_pillar,
                   v.validated_pillar, v.validated_axis,
                   CASE WHEN v.validation_source='historical_import' OR EXISTS (
                       SELECT 1 FROM validation_history h WHERE h.hal_id=a.hal_id
                       AND h.validation_source='historical_import') THEN NULL
                   ELSE COALESCE((SELECT MIN(h.validated_at) FROM validation_history h
                                  WHERE h.hal_id=a.hal_id), v.validated_at) END AS first_validated_at,
                   CASE WHEN a.openalex_match_status='matched' THEN a.citations END AS citations,
                   CASE WHEN a.openalex_match_status='matched' THEN a.openalex_updated_at END AS citations_at
            FROM articles a JOIN validations v ON v.hal_id=a.hal_id
            WHERE a.status='validated' ORDER BY a.hal_id
        """).fetchall()
        previous = con.execute('SELECT * FROM veille_reports ORDER BY rowid DESC LIMIT 1').fetchone()
        resource_rows = con.execute('SELECT p.hal_id,r.kind,r.url FROM publication_resources p '
            'JOIN hal_resources r USING(resource_id) WHERE r.url IS NOT NULL '
            'ORDER BY p.hal_id,r.kind,r.url').fetchall()
        from dashboard.scripts.resource_catalog import catalog
        resource_catalog = catalog(con)
        resource_metadata = {(r['kind'], r['url']): r for r in resource_catalog['items']}
        cluster_labs = {}
        for row in con.execute('SELECT hal_id,lab FROM publication_labs_cluster ORDER BY lab'):
            cluster_labs.setdefault(row['hal_id'], []).append(row['lab'])
        industrial_partners = {}
        for row in con.execute('SELECT hal_id,partner FROM publication_partners ORDER BY partner'):
            industrial_partners.setdefault(row['hal_id'], []).append(row['partner'])
    resources = {}
    for resource in resource_rows:
        if safe_url(resource['url']):
            metadata = resource_metadata.get((resource['kind'], resource['url']), {})
            entry = {'kind': resource['kind'], 'url': resource['url'],
                     'title': metadata.get('title', ''), 'license': metadata.get('license', '')}
            if entry not in resources.setdefault(resource['hal_id'], []):
                resources[resource['hal_id']].append(entry)
    articles = []
    since = three_months_before(now)
    for row in rows:
        item = dict(row)
        item['resources'] = resources.get(item['hal_id'], [])
        item['cluster_labs'] = cluster_labs.get(item['hal_id'], [])
        item['industrial_partners'] = industrial_partners.get(item['hal_id'], [])
        pillar = canonical(item.pop('validated_pillar'))
        if pillar is None:
            raise ValueError(f"Pilier humain inconnu pour {item['hal_id']}")
        human_axis = item.pop('validated_axis')
        axis_pillar = item.pop('axis_pillar')
        predicted_pillar = canonical(axis_pillar or item.pop('predicted_pillar'))
        item.pop('predicted_pillar', None)
        predicted_axis = item.pop('predicted_axis')
        if pillar == 'No class':
            axis, source = 'Hors périmètre', 'none'
        elif human_axis in SEQUOIA_SAXES.get(pillar, {}):
            axis, source = human_axis, 'human'
        elif human_axis:
            axis, source = UNASSIGNED_AXIS, 'none'
        elif predicted_pillar == pillar and predicted_axis in SEQUOIA_SAXES.get(pillar, {}):
            axis, source = predicted_axis, 'model'
        else:
            axis, source = UNASSIGNED_AXIS, 'none'
        date = parsed_date(item['first_validated_at'])
        item.update(pillar=pillar, axis=axis, axis_source=source,
                    recent=bool(date and since <= date <= now))
        try:
            item['year'] = int(item['year']) if item['year'] is not None else None
        except (ValueError, TypeError, OverflowError):
            item['year'] = None
        count = citation_value(item['citations'])
        item['citation_note'] = None
        if count is not None and not identity_supported(item):
            item['citation_note'] = 'Correspondance OpenAlex non vérifiable'
            count = None
        if count is not None and not parsed_date(item['citations_at']):
            item['citation_note'] = 'Date de relevé inconnue'
            count = None
        item['citations'] = count
        if item['citations'] is None:
            item['citations_at'] = None
        articles.append(item)
    result = (articles, dict(previous) if previous else None, since)
    return (*result, resource_catalog['items']) if include_resources else result


def safe_url(value):
    value = str(value or '').strip()
    try:
        parsed = urlsplit(value)
    except ValueError:
        return None
    return value if parsed.scheme in {'http', 'https'} and parsed.netloc else None


def article_url(item):
    if safe_url(item.get('url')):
        return safe_url(item['url'])
    doi = str(item.get('doi') or '').strip()
    if safe_url(doi):
        return safe_url(doi)
    doi = re.sub(r'^doi:\s*', '', doi, flags=re.I)
    return safe_url('https://doi.org/' + doi) if doi.startswith('10.') else None


def keywords_for(items, limit=6):
    counts, labels = Counter(), {}
    for item in items:
        seen = set()
        for raw in re.split(r'[;|]', str(item.get('keywords') or '')):
            label = ' '.join(raw.split())
            key = label.casefold()
            if 2 <= len(label) <= 100 and key not in seen:
                counts[key] += 1
                labels.setdefault(key, label)
                seen.add(key)
    ranked = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    return [(labels[key], count) for key, count in (ranked if limit is None else ranked[:limit])]


def thematic_keywords(items, limit=3):
    generic = {'ai', 'ia', 'artificialintelligence', 'intelligenceartificielle',
               'machinelearning', 'deeplearning', 'classification', 'learning',
               'neuralnetwork', 'neuralnetworks', 'deepneuralnetwork', 'deepneuralnetworks', 'training'}
    return [(word, count) for word, count in keywords_for(items, limit=None)
            if count >= 2 and re.sub(r'\W', '', word.casefold()) not in generic
            and not re.search(r'\b(?:cs|math|physics)[a-z]{2}\b', word.casefold())
            and not word.casefold().startswith(('fos ', 'computer science ', 'computer and information sciences'))][:limit]


def sorted_articles(items):
    return sorted(items, key=lambda a: (-(a.get('year') or 0), str(a.get('title') or ''), a['hal_id']))


def editorial_corpus(articles):
    # No class is a workflow category, not a scientific pillar of the Cluster.
    # Statistics use the same validated HAL records as the dashboard filters.
    return [a for a in articles if a['pillar'] in SEQUOIA_SAXES]


def shortened(value, limit):
    value = ' '.join(str(value or '').split())
    return value if len(value) <= limit else value[:limit].rsplit(' ', 1)[0] + '…'


def select_publications(items, limit=3):
    """Most cited unique publications with verifiable nonzero counts."""
    return sorted([a for a in items if a.get('citations') is not None and a['citations'] > 0],
                  key=lambda a: (-a['citations'], -(a.get('year') or 0), a['hal_id']))[:limit]


def select_new_publications(items, year, excluded=(), limit=3):
    """Current-year publications: recent additions first, then publication date."""
    excluded = set(excluded)
    candidates = [a for a in items if a['hal_id'] not in excluded and a.get('year') == year]
    def date_key(item):
        parts = str(item.get('publication_date') or '').split('-')
        return tuple(-int(parts[i]) if i < len(parts) and parts[i].isdigit() else 0 for i in range(3))
    return sorted(candidates, key=lambda a: (not a['recent'], date_key(a),
                  -(parsed_date(a.get('first_validated_at')).timestamp() if parsed_date(a.get('first_validated_at')) else 0),
                  a['hal_id']))[:limit]


def report_blocks(payload, previous=None):
    """A decision brief: overview, three pillar sheets, compact methodology."""
    raw = payload['articles']
    articles = editorial_corpus(raw)
    year = parsed_date(payload['generated_at']).year
    blocks = []

    def add(kind, text):
        blocks.append((kind, text))

    def chart(title, rows, kind='bars'):
        add('chart', json.dumps({'title': title, 'rows': rows, 'kind': kind}, ensure_ascii=False))

    counts = Counter(a['pillar'] for a in articles)
    add('kicker', 'CLUSTER SEQUOIA / NOTE DE VEILLE')
    add('title', 'Panorama scientifique')
    add('p', f"Édition du {display_date(payload['generated_at'])}")
    add('metric', f"<b>{len(articles)}</b> publications dans les trois piliers du Cluster")
    add('h1', 'Répartition des publications')
    from dashboard.scripts.report_charts import PALETTE
    chart('Publications par pilier', [
        {'label': p, 'value': counts[p], 'display': f"{counts[p]} · {100*counts[p]/max(1,len(articles)):.0f} %", 'color': PALETTE[i]}
        for i,p in enumerate(SEQUOIA_SAXES)])
    add('h2', 'Années de publication')
    years = Counter(a['year'] for a in articles if a.get('year') is not None and a['year'] <= year)
    visible_years = list(range(year-2, year+1))
    chart('Publications par année', [
        {'label': str(y), 'value': years[y], 'color': '#315fcb' if y < year else '#168b85'}
        for y in visible_years], kind='years')
    from dashboard.scripts.report_analysis import AXIS_PURPOSES
    identities = {hal_id: item['hal_id'] for item in unique_publications(articles) for hal_id in item['hal_ids']}
    shown = set()

    reference_number = 0
    for pillar, configured_axes in SEQUOIA_SAXES.items():
        items = [a for a in articles if a['pillar'] == pillar]
        add('pillar', escape(pillar))
        current_count = sum(a.get('year') == year for a in items)
        add('metric', f"Ce pilier rassemble <b>{len(items)} publications</b>, dont <b>{current_count}</b> publiées en {year}.")
        if not items:
            add('p', 'Aucune publication disponible dans ce pilier.')
            continue
        axes = Counter(a['axis'] for a in items)
        add('h2', 'Répartition par axe scientifique')
        color = PALETTE[list(SEQUOIA_SAXES).index(pillar)]
        rows = [{'label': axis, 'value': axes[axis],
                 'display': f"{axes[axis]} · {100*axes[axis]/len(items):.0f} %", 'color': color}
                for axis in configured_axes]
        if axes[UNASSIGNED_AXIS]:
            rows.append({'label': 'Axe non précisé', 'value': axes[UNASSIGNED_AXIS],
                         'display': f"{axes[UNASSIGNED_AXIS]} · {100*axes[UNASSIGNED_AXIS]/len(items):.0f} %", 'color': '#9aa9bc'})
        chart('Publications par axe scientifique', rows)
        industrial = [a for a in unique_publications(items)
                      if a.get('industrial_partners') and identities[a['hal_id']] not in shown
                      and (a.get('year') or 0) <= year
                      and str(a.get('publication_date') or '') <= str(payload['generated_at'])[:10]]
        industrial.sort(key=lambda a: (-(a.get('year') or 0),
                        tuple(-int(part) if part.isdigit() else 0 for part in (str(a.get('publication_date') or '').split('-') + ['0', '0'])[:3]),
                        a['hal_id']))
        industrial = industrial[:3]
        sections = []
        if industrial:
            sections.append(('Publications récentes avec des partenaires industriels', industrial))
            shown.update(identities[a['hal_id']] for a in industrial)
        for axis in configured_axes:
            group = [a for a in items if a['axis'] == axis]
            available = [a for a in unique_publications(group) if identities[a['hal_id']] not in shown]
            cited = select_publications(available, limit=3)
            shown.update(identities[a['hal_id']] for a in cited)
            available = [a for a in available if identities[a['hal_id']] not in shown]
            newcomers = select_new_publications(available, year, limit=3)
            shown.update(identities[a['hal_id']] for a in newcomers)
            sections.append(('axis', {'axis': axis, 'group': group}))
            if cited:
                sections.append(('Publications les plus citées', cited))
            if newcomers:
                sections.append(('Nouvelles publications', newcomers))
        for section_title, selected in sections:
            if section_title == 'axis':
                axis, group = selected['axis'], selected['group']
                add('h1', escape(axis))
                purpose = AXIS_PURPOSES.get(axis)
                if purpose:
                    add('p', f"Cet axe vise à {escape(purpose)}.")
                add('metric', f"<b>{len(group)}</b> publications dans cet axe")
                words = thematic_keywords(group)
                if words:
                    add('h2', 'Sujets récurrents')
                    add('p', '; '.join(f"{escape(word)} ({count} publications)" for word, count in words) + '.')
                continue
            if not selected:
                continue
            add('h2', section_title)
            for item in selected:
                reference_number += 1
                label = 'Nouveauté' if item.get('year') == year else 'Ajout récent' if item['recent'] else ''
                title = escape(shortened(item.get('title') or 'Sans titre', 210))
                url = article_url(item)
                if url:
                    title = f'<a href="{escape(url, quote=True)}">{title}</a>'
                authors = re.split(r';|,', str(item.get('authors') or ''))
                authors = ', '.join(a.strip() for a in authors[:2] if a.strip())
                if len(authors) > 110:
                    authors = shortened(authors, 110)
                if len(re.split(r';|,', str(item.get('authors') or ''))) > 2:
                    authors += ' et al.'
                venue = shortened(item.get('journal') or item.get('conference') or '', 65)
                meta = ' · '.join(str(v) for v in (date_label(item.get('publication_date')) or item.get('year') or 'Année inconnue', authors, venue) if v)
                badge = f" · {label}" if label else ""
                text = f"<b>[{reference_number}]{badge} · {title}</b><br/>{escape(meta)}<br/>"
                details = []
                if section_title == 'Publications récentes avec des partenaires industriels':
                    details.append('<b>Partenaires :</b> ' + ' · '.join(map(escape, item.get('industrial_partners', []))))
                    if item['axis'] != UNASSIGNED_AXIS:
                        details.append(escape(item['axis']))
                if item.get('citations') is not None:
                    details.append(f"<b>{item['citations']} citations</b> (OpenAlex, {display_date(item['citations_at'])})")
                details.append(f"HAL : {escape(item['hal_id'])}")
                text += ' · '.join(details)
                resource_links = []
                labels = {'dataset': 'Données associées', 'software': 'Logiciel associé', 'code': 'Code', 'patent': 'Brevet'}
                for resource in item.get('resources', [])[:2]:
                    url = safe_url(resource.get('url'))
                    if url:
                        resource_links.append(f'<a href="{escape(url, quote=True)}">{labels.get(resource["kind"], "Ressource")}</a>')
                if resource_links:
                    text += '<br/>' + ' · '.join(resource_links)
                add('reference', text)
    return blocks


def render_html(path, blocks):
    elements = []
    for kind, text in blocks:
        if kind == 'chart':
            from dashboard.scripts.report_charts import chart_svg
            elements.append('<figure class="report-chart">' + chart_svg(json.loads(text)) + '</figure>')
            continue
        if kind == 'table':
            table = json.loads(text)
            header = ''.join(f'<th>{escape(cell)}</th>' for cell in table['headers'])
            rows = ''.join('<tr>' + ''.join(f'<td>{escape(cell)}</td>' for cell in row) + '</tr>' for row in table['rows'])
            elements.append(f'<div class="table-wrap"><table><thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table></div>')
            continue
        if kind == 'page':
            elements.append('<hr class="page"/>')
            continue
        tag = {'title': 'h1', 'h1': 'h2', 'pillar': 'h2', 'h2': 'h3', 'h3': 'h4'}.get(kind, 'p')
        elements.append(f'<{tag} class="{kind}">{text}</{tag}>')
    path.write_text('''<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Rapport de veille SequoIA</title>
<style>body{font:16px/1.65 system-ui,sans-serif;color:#20304d;background:#f3f6fb;margin:0}
main{max-width:980px;margin:auto;padding:45px;background:white}h1,h2,h3,h4{line-height:1.3;color:#172554}
h1{font-size:36px}.kicker{color:#315fcb;letter-spacing:.15em;font-weight:bold}
.metric{font-size:18px;padding:12px 0;color:#172554}.note{font-size:12px;color:#52627b}
.report-chart{margin:15px 0 22px}.report-chart svg{display:block;width:100%;height:auto}
.table-wrap{overflow-x:auto}table{width:100%;border-collapse:collapse;margin:16px 0;font-size:13px}
th{text-align:left;background:#172554;color:white;padding:10px}td{padding:9px;border-bottom:1px solid #d9e1ee}
td:not(:first-child){text-align:right;white-space:nowrap}.page{margin-top:50px;border:0;border-top:3px solid #315fcb}
.pillar{border-top:3px solid #315fcb;padding-top:25px;margin-top:60px}.reference{padding:12px 16px;background:#f7f9fd;overflow-wrap:anywhere}
.excerpt{border-left:3px solid #328c8e;padding-left:15px;color:#52627b}a{color:#315fcb}
@media(max-width:650px){main{padding:20px}h1{font-size:28px}}
@media print{body{background:white}.pillar,.page{break-before:page}a{color:inherit}}
</style></head><body><main>''' + '\n'.join(elements) + '</main></body></html>', encoding='utf-8')


def render_pdf(path, blocks):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak, KeepTogether, Table, TableStyle, Spacer
    font_dir = Path('/usr/share/fonts/truetype/dejavu')
    if (font_dir / 'DejaVuSans.ttf').exists() and (font_dir / 'DejaVuSans-Bold.ttf').exists():
        for name, filename in [('Sequoia', 'DejaVuSans.ttf'), ('SequoiaBold', 'DejaVuSans-Bold.ttf')]:
            if name not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont(name, str(font_dir / filename)))
        pdfmetrics.registerFontFamily('Sequoia', normal='Sequoia', bold='SequoiaBold',
                                      italic='Sequoia', boldItalic='SequoiaBold')
        font, bold = 'Sequoia', 'SequoiaBold'
    else:
        font, bold = 'Helvetica', 'Helvetica-Bold'
    base = ParagraphStyle('body', fontName=font, fontSize=8.5, leading=12, spaceAfter=7,
                          textColor=colors.HexColor('#20304d'), alignment=TA_LEFT,
                          splitLongWords=True)
    styles = {'p': base}
    for kind, size in [('title', 28), ('h1', 17), ('pillar', 19), ('h2', 10.5), ('h3', 10), ('kicker', 8)]:
        styles[kind] = ParagraphStyle(kind, parent=base, fontName=bold, fontSize=size,
                                      leading=size * 1.25, spaceBefore=9, spaceAfter=7,
                                      keepWithNext=True, textColor=colors.HexColor('#172554'))
    styles['reference'] = ParagraphStyle('reference', parent=base, fontSize=7.4, leading=10.3,
                                       spaceAfter=8)
    styles['note'] = ParagraphStyle('note', parent=base, fontSize=7, leading=10,
                                  textColor=colors.HexColor('#52627b'))
    styles['metric'] = ParagraphStyle('metric', parent=base, fontSize=10, leading=14, spaceAfter=10)
    table_body = ParagraphStyle('table_body', parent=base, fontSize=7.3, leading=10, spaceAfter=0)
    table_head = ParagraphStyle('table_head', parent=table_body, fontName=bold, textColor=colors.white)
    styles['excerpt'] = ParagraphStyle('excerpt', parent=base, fontSize=8, leading=12,
                                      leftIndent=12, textColor=colors.HexColor('#52627b'))
    for kind in ('analysis', 'analysis_source', 'analysis_lead'):
        styles[kind] = ParagraphStyle(kind, parent=base, keepWithNext=True)
    story = []
    pending_reference = []
    pending_heading = None
    for index, (kind, text) in enumerate(blocks):
        if pending_reference and kind != 'excerpt':
            story.append(KeepTogether(pending_reference))
            pending_reference = []
        if kind == 'chart':
            from dashboard.scripts.report_charts import chart_drawing
            story.extend([chart_drawing(json.loads(text)), Spacer(1, 6)])
            continue
        if kind == 'page':
            story.append(PageBreak())
            continue
        if kind == 'table':
            data = json.loads(text)
            cells = [[Paragraph(escape(cell), table_head) for cell in data['headers']]]
            cells += [[Paragraph(escape(cell), table_body) for cell in row] for row in data['rows']]
            table = Table(cells, colWidths=[(A4[0] - 84) * w for w in data['widths']], repeatRows=1, hAlign='LEFT')
            table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#172554')),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f3f6fb')]),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('LEFTPADDING', (0,0), (-1,-1), 7), ('RIGHTPADDING', (0,0), (-1,-1), 7),
                ('TOPPADDING', (0,0), (-1,-1), 5), ('BOTTOMPADDING', (0,0), (-1,-1), 5),
                ('LINEBELOW', (0,-1), (-1,-1), .5, colors.HexColor('#d9e1ee')),
            ]))
            story.extend([table, Spacer(1, 7)])
            continue
        if kind == 'pillar':
            story.append(PageBreak())
        # Helvetica also supports these punctuation characters; avoid exotic arrows.
        text = text.replace('→', 'à').translate(str.maketrans({
            '\u2010': '-', '\u2011': '-', '\u2012': '-', '\u2013': '-', '\u2014': '-',
            '\u2212': '-', '\u00a0': ' ', '\u202f': ' ',
        }))
        paragraph = Paragraph(text, styles.get(kind, base))
        if kind == 'h2' and index + 1 < len(blocks) and blocks[index + 1][0] == 'reference':
            pending_heading = paragraph
            continue
        if kind == 'reference':
            pending_reference = [pending_heading, paragraph] if pending_heading is not None else [paragraph]
            pending_heading = None
        elif kind == 'excerpt' and pending_reference:
            pending_reference.append(paragraph)
            story.append(KeepTogether(pending_reference))
            pending_reference = []
        else:
            story.append(paragraph)
    if pending_reference:
        story.append(KeepTogether(pending_reference))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor('#d9e1ee'))
        canvas.line(42, 39, A4[0] - 42, 39)
        canvas.setFont(font, 8)
        canvas.setFillColor(colors.HexColor('#52627b'))
        canvas.drawString(42, 26, 'SequoIA - Veille scientifique')
        canvas.drawRightString(A4[0] - 42, 26, str(doc.page))
        canvas.restoreState()
    SimpleDocTemplate(str(path), pagesize=A4, rightMargin=42, leftMargin=42,
                      topMargin=45, bottomMargin=55, title='Rapport de veille scientifique SequoIA',
                      author='Cluster SequoIA').build(story, onFirstPage=footer, onLaterPages=footer)


def report_summary(row):
    return {key: row[key] for key in ('id', 'generated_at', 'recent_since', 'article_count', 'recent_count')}


def generate_report(*, now=None, progress=None):
    """Appelé sous le verrou global des pipelines ; publication après les deux rendus."""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now = now.astimezone(timezone.utc)
    articles, previous, since, resources = load_corpus(now, include_resources=True)
    if not articles:
        raise ValueError('Aucune publication disponible pour le rapport.')
    serialized = json.dumps({'format_version': FORMAT_VERSION, 'edition_year': now.year, 'articles': articles,
                             'resource_catalog': resources},
                            sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    fingerprint = hashlib.sha256(serialized.encode('utf-8')).hexdigest()
    if previous and previous['fingerprint'] == fingerprint:
        directory = reports_directory() / previous['id']
        if all((directory / filename).is_file() for filename in ('report.html', 'report.pdf')):
            return {**report_summary(previous), 'reused': True}
    report_id = uuid.uuid4().hex
    scientific_articles = editorial_corpus(articles)
    payload = {'id': report_id, 'format_version': FORMAT_VERSION, 'generated_at': now.isoformat(timespec='seconds'),
               'recent_since': since.isoformat(timespec='seconds'), 'articles': articles, 'resource_catalog': resources,
               'article_count': len(scientific_articles), 'recent_count': sum(a['recent'] for a in scientific_articles)}
    old_payload = json.loads(previous['payload_json']) if previous else None
    blocks = report_blocks(payload, old_payload)
    root = reports_directory()
    directory = root / report_id
    temporary = root / ('.pending-' + report_id)
    temporary.mkdir(parents=True, exist_ok=False)
    if progress:
        progress('Création du rapport HTML et PDF')
    render_html(temporary / 'report.html', blocks)
    render_pdf(temporary / 'report.pdf', blocks)
    os.replace(temporary, directory)
    with connect_db() as con:
        superseded = [row[0] for row in con.execute('SELECT id FROM veille_reports')]
        con.execute('INSERT INTO veille_reports '
                    '(id,fingerprint,generated_at,recent_since,article_count,recent_count,payload_json) '
                    'VALUES (?,?,?,?,?,?,?)',
                    (report_id, fingerprint, payload['generated_at'], payload['recent_since'],
                     payload['article_count'], payload['recent_count'], json.dumps(payload, ensure_ascii=False)))
        con.execute('DELETE FROM veille_reports WHERE id != ?', (report_id,))
    for old_id in superseded:
        if re.fullmatch(r'[0-9a-f]{32}', old_id):
            shutil.rmtree(root / old_id, ignore_errors=True)
    return {**report_summary(payload), 'reused': False}
