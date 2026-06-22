"""
PDF meeting report generation using reportlab.

Produces a single formatted summary document: TL;DR, key decisions, action
items table, participants, sentiment, and topics. The full transcript is
intentionally NOT embedded here — it belongs in the separate .txt export,
keeping this report short enough to actually be read.
"""

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from utils.logger import get_logger

log = get_logger(__name__)


def _build_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontSize=20,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionHeading",
            parent=styles["Heading2"],
            textColor=colors.HexColor("#1F2937"),
            spaceBefore=16,
            spaceAfter=6,
        )
    )
    return styles


def export_summary_pdf(analysis: dict, meeting_insights: dict, output_path: str, context: str = "") -> str:
    """Writes a formatted PDF report and returns the output path."""
    styles = _build_styles()
    doc = SimpleDocTemplate(
        output_path,
        pagesize=LETTER,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
    )

    story = [Paragraph("Meeting Summary Report", styles["ReportTitle"])]

    if context:
        story.append(Paragraph(f"<i>Context: {context}</i>", styles["Normal"]))

    story.append(Spacer(1, 12))

    # --- TL;DR ---
    story.append(Paragraph("TL;DR", styles["SectionHeading"]))
    story.append(Paragraph(analysis.get("tldr") or "Not available.", styles["Normal"]))

    # --- Key Decisions ---
    story.append(Paragraph("Key Decisions", styles["SectionHeading"]))
    decisions = analysis.get("key_decisions") or []
    if decisions:
        for d in decisions:
            story.append(Paragraph(f"• {d}", styles["Normal"]))
    else:
        story.append(Paragraph("None identified.", styles["Normal"]))

    # --- Action Items table ---
    story.append(Paragraph("Action Items", styles["SectionHeading"]))
    action_items = analysis.get("action_items") or []
    table_data = [["Task", "Owner", "Deadline", "Priority"]]
    if action_items:
        for item in action_items:
            table_data.append(
                [
                    item.get("task") or "Not specified",
                    item.get("owner") or "Not specified",
                    item.get("deadline") or "Not specified",
                    item.get("priority") or "Medium",
                ]
            )
    else:
        table_data.append(["No action items identified.", "—", "—", "—"])

    action_table = Table(table_data, colWidths=[2.4 * inch, 1.3 * inch, 1.3 * inch, 1 * inch])
    action_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F2937")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ]
        )
    )
    story.append(action_table)

    # --- Participants ---
    story.append(Paragraph("Participants", styles["SectionHeading"]))
    participants = analysis.get("participants") or []
    story.append(Paragraph(", ".join(participants) if participants else "Not identifiable from transcript.", styles["Normal"]))

    # --- Sentiment ---
    story.append(Paragraph("Sentiment", styles["SectionHeading"]))
    sentiment = analysis.get("sentiment") or {}
    sentiment_line = sentiment.get("overall", "Unknown")
    if sentiment.get("explanation"):
        sentiment_line += f" — {sentiment['explanation']}"
    story.append(Paragraph(sentiment_line, styles["Normal"]))

    # --- Topics ---
    story.append(Paragraph("Topics Discussed", styles["SectionHeading"]))
    topics = analysis.get("topics") or []
    if topics:
        for t in topics:
            name = t.get("topic", "Untitled") if isinstance(t, dict) else str(t)
            desc = t.get("description", "") if isinstance(t, dict) else ""
            line = f"• {name}" + (f" — {desc}" if desc else "")
            story.append(Paragraph(line, styles["Normal"]))
    else:
        story.append(Paragraph("No topics identified.", styles["Normal"]))

    # --- Meeting Insights ---
    story.append(Paragraph("Meeting Insights", styles["SectionHeading"]))
    insight_rows = [
        ["Duration", str(meeting_insights.get("duration_seconds", "Unknown"))],
        ["Word Count", str(meeting_insights.get("word_count", 0))],
        ["Speaking Rate (WPM)", str(meeting_insights.get("speaking_rate_wpm", 0))],
        ["Estimated Speakers", f"~{meeting_insights.get('estimated_speakers', 1)} (approximate)"],
    ]
    insight_table = Table(insight_rows, colWidths=[2.2 * inch, 3.8 * inch])
    insight_table.setStyle(
        TableStyle(
            [
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F1F5F9")),
            ]
        )
    )
    story.append(insight_table)

    doc.build(story)
    log.info("PDF report written to %s", output_path)
    return output_path
