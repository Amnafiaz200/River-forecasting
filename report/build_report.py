"""Builds report/Assignment2_Report.pdf from q1_text.py, q2_text.py and the results/ JSON files."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reportlab.platypus import SimpleDocTemplate, PageBreak
from rl_helpers import *
import q1_text

HERE = os.path.dirname(os.path.abspath(__file__))
GITHUB = "https://github.com/&lt;YOUR-USERNAME&gt;/&lt;YOUR-REPO&gt;   (replace with the public repository URL before submitting)"


def page_number(canvas, doc):
    canvas.setFont("DV", 7)
    canvas.drawRightString(PAGE_W - MARGIN, 0.9 * cm, f"AI651 Assignment 2 — page {doc.page}")


def main(with_q2=True):
    story = []
    story.append(P("AI651 — Deep Learning for Space, Time and Graphs (Fall 2026)", H3))
    story.append(P("Assignment 2: Report", H1))
    story.append(P(f"<b>Code repository (must be PUBLIC at submission):</b> {GITHUB}", BODY))
    story.append(P("Contents: Question 1 (river forecasting proposal, six required headings); Question 2 (Motion or Memory?, Tables 1–6, Figures 1–6, Q2.1–Q2.17); "
                   "AI-use disclosure (last page).", SMALL))
    story += q1_text.build()
    if with_q2:
        import q2_text
        story.append(PageBreak())
        story += q2_text.build()
    import ai_disclosure
    story.append(PageBreak())
    story += ai_disclosure.build()
    out = os.path.join(HERE, "Assignment2_Report.pdf")
    SimpleDocTemplate(out, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=1.5 * cm, bottomMargin=1.6 * cm,
                      title="AI651 Assignment 2").build(story, onFirstPage=page_number, onLaterPages=page_number)
    print("wrote", out)


if __name__ == "__main__":
    main(with_q2=os.path.exists(os.path.join(HERE, "q2_text.py")) and "--no-q2" not in sys.argv)
