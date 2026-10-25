"""Small reportlab helpers: Unicode fonts, styles, tables, rendered equations, figures."""
import hashlib, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import get_data_path
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, Spacer, Table, TableStyle, Preformatted

HERE = os.path.dirname(os.path.abspath(__file__))
FD = os.path.join(get_data_path(), "fonts", "ttf")
for n, f in [("DV", "DejaVuSans.ttf"), ("DV-B", "DejaVuSans-Bold.ttf"), ("DV-I", "DejaVuSans-Oblique.ttf"),
             ("DV-BI", "DejaVuSans-BoldOblique.ttf"), ("DVM", "DejaVuSansMono.ttf")]:
    pdfmetrics.registerFont(TTFont(n, os.path.join(FD, f)))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")

PAGE_W, PAGE_H = A4
MARGIN = 1.7 * cm
CONTENT_W = PAGE_W - 2 * MARGIN

BODY = ParagraphStyle("body", fontName="DV", fontSize=8.6, leading=11.6, alignment=TA_LEFT, spaceAfter=3.5)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=7.4, leading=9.6, spaceAfter=2)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=7.2, leading=9.2, spaceAfter=0)
CELLB = ParagraphStyle("cellb", parent=CELL, fontName="DV-B")
H1 = ParagraphStyle("h1", fontName="DV-B", fontSize=14, leading=18, spaceBefore=8, spaceAfter=6, textColor=colors.HexColor("#12355b"))
H2 = ParagraphStyle("h2", fontName="DV-B", fontSize=10.5, leading=14, spaceBefore=8, spaceAfter=4, textColor=colors.HexColor("#12355b"))
H3 = ParagraphStyle("h3", fontName="DV-B", fontSize=9, leading=12, spaceBefore=5, spaceAfter=2)
CAP = ParagraphStyle("cap", parent=BODY, fontSize=7.6, leading=9.8, textColor=colors.HexColor("#333333"), spaceAfter=7)
BUL = ParagraphStyle("bul", parent=BODY, leftIndent=11, bulletIndent=2, spaceAfter=2)
Q = ParagraphStyle("q", parent=BODY, fontName="DV-B", spaceBefore=5, spaceAfter=1.5, textColor=colors.HexColor("#7a1f1f"))
MONO = ParagraphStyle("mono", fontName="DVM", fontSize=6.8, leading=8.4)


def P(t, st=BODY):
    return Paragraph(t, st)


def bullets(items, st=BUL):
    return [Paragraph(i, st, bulletText="•") for i in items]


def table(rows, widths, header=True, font=CELL, zebra=True, align_center_cols=()):
    """rows: list of lists of str/Flowable; widths in fractions of CONTENT_W."""
    data = []
    for r_i, r in enumerate(rows):
        data.append([c if not isinstance(c, str) else Paragraph(c, CELLB if (header and r_i == 0) else font) for c in r])
    t = Table(data, colWidths=[w * CONTENT_W for w in widths], repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
          ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
          ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfe8f3")))
    if zebra:
        for i in range(1, len(data)):
            if i % 2 == 0:
                st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f5f7fa")))
    t.setStyle(TableStyle(st))
    return t


def figure(path, width_frac=1.0, max_h=None):
    im = PILImage.open(path)
    w, h = im.size
    W = CONTENT_W * width_frac
    H = W * h / w
    if max_h and H > max_h:
        W, H = W * max_h / H, max_h
    return Image(path, width=W, height=H)


EQ_DIR = os.path.join(HERE, "eqs"); os.makedirs(EQ_DIR, exist_ok=True)


def eq(latex, tag=None, fs=11):
    """Render a mathtext equation to PNG (cached) and return a Table [equation | (tag)]."""
    for bad in (r"\bigl", r"\bigr", r"\Bigl", r"\Bigr", r"\!"):
        latex = latex.replace(bad, "")
    latex = latex.replace("\\|", r"\Vert ")
    h = hashlib.md5((latex + str(fs)).encode()).hexdigest()[:12]
    path = os.path.join(EQ_DIR, h + ".png")
    if not os.path.exists(path):
        fig = plt.figure(figsize=(0.1, 0.1))
        fig.text(0, 0, f"${latex}$", fontsize=fs)
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.04, transparent=False, facecolor="white")
        plt.close(fig)
    im = PILImage.open(path)
    w, h_ = im.size
    scale = 72 / 300 * 0.98
    W, H = w * scale, h_ * scale
    if W > CONTENT_W * 0.9:
        s = CONTENT_W * 0.9 / W
        W, H = W * s, H * s
    img = Image(path, width=W, height=H)
    t = Table([[img, Paragraph(f"({tag})" if tag else "", BODY)]], colWidths=[CONTENT_W * 0.92, CONTENT_W * 0.08])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (1, 0), (1, 0), "RIGHT"),
                           ("TOPPADDING", (0, 0), (-1, -1), 1), ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]))
    return t
