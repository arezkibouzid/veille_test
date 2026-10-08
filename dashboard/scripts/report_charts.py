"""Shared vector charts for the printable PDF and the HTML report."""
import textwrap

PALETTE = ['#315fcb', '#7860b7', '#168b85']


def chart_drawing(data, width=511.276):
    from reportlab.graphics.shapes import Drawing, Rect, String, Line
    from reportlab.lib.colors import HexColor
    rows = data['rows']
    if data['kind'] == 'years':
        height = 115
        drawing = Drawing(width, height)
        maximum = max([row['value'] for row in rows] + [1])
        step = width / max(1, len(rows))
        for index, row in enumerate(rows):
            x = step * (index + .5)
            bar_height = 65 * row['value'] / maximum
            drawing.add(Rect(x - 24, 25, 48, bar_height, fillColor=HexColor(row.get('color', PALETTE[0])), strokeColor=None))
            drawing.add(String(x, 30 + bar_height, str(row['value']), textAnchor='middle', fontName='Helvetica-Bold', fontSize=9, fillColor=HexColor('#172554')))
            drawing.add(String(x, 9, row['label'], textAnchor='middle', fontName='Helvetica', fontSize=8, fillColor=HexColor('#52627b')))
        drawing.add(Line(0, 25, width, 25, strokeColor=HexColor('#d9e1ee'), strokeWidth=.5))
        return drawing
    label_width = width * .51
    plot_width = width * .32
    row_height = 24
    height = len(rows) * row_height + 28
    drawing = Drawing(width, height)
    maximum = max([row['value'] for row in rows] + [1])
    for tick in sorted({0, maximum // 2, maximum}):
        x = label_width + plot_width * tick / maximum
        drawing.add(Line(x, 20, x, height - 3, strokeColor=HexColor('#e4eaf3'), strokeWidth=.5))
        drawing.add(String(x, 7, str(tick), textAnchor='middle', fontName='Helvetica', fontSize=7, fillColor=HexColor('#65758b')))
    for index, row in enumerate(rows):
        y = height - 16 - index * row_height
        lines = textwrap.wrap(row['label'].replace('–', '-'), width=49) or ['']
        for offset, line in enumerate(lines[:2]):
            drawing.add(String(0, y + (3 if len(lines) > 1 else 0) - offset * 9,
                               line, fontName='Helvetica', fontSize=7.4, fillColor=HexColor('#20304d')))
        drawing.add(Rect(label_width, y - 3, plot_width * row['value'] / maximum, 10,
                         fillColor=HexColor(row.get('color', PALETTE[0])), strokeColor=None))
        drawing.add(String(label_width + plot_width + 9, y - 1, row.get('display', str(row['value'])),
                           fontName='Helvetica-Bold', fontSize=8, fillColor=HexColor('#172554')))
    return drawing


def chart_svg(data):
    from html import escape
    from reportlab.graphics import renderSVG
    svg = renderSVG.drawToString(chart_drawing(data))
    svg = svg[svg.index('<svg'):]
    description = data['title'] + '. ' + '; '.join(f"{row['label']}: {row['value']}" for row in data['rows'])
    return svg.replace('<svg ', f'<svg role="img" aria-label="{escape(description, quote=True)}" ', 1)
