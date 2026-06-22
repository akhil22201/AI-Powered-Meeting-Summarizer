"""
Pure rendering helpers: turn raw data (dicts, lists) into the markdown or
table shapes the Gradio UI displays. No Gradio imports here — these
functions just return strings/lists, which keeps them trivially testable
without spinning up a UI.
"""

from core import insights


def render_section_markdown(title: str, body: str, icon: str) -> str:
    if not body:
        body = "_Nothing generated for this section._"
    return f"### {icon} {title}\n\n{body}\n"


def render_list_markdown(title: str, items: list, icon: str, empty_text: str) -> str:
    if not items:
        return f"### {icon} {title}\n\n_{empty_text}_\n"
    bullet_lines = "\n".join(f"- {item}" for item in items)
    return f"### {icon} {title}\n\n{bullet_lines}\n"


def render_sentiment_markdown(sentiment: dict) -> str:
    overall = sentiment.get("overall", "Unknown") if sentiment else "Unknown"
    explanation = sentiment.get("explanation", "") if sentiment else ""

    badge_colors = {
        "Positive": "🟢",
        "Neutral": "⚪",
        "Negative": "🔴",
        "Mixed": "🟡",
        "Unknown": "⚫",
    }
    badge = badge_colors.get(overall, "⚫")

    body = f"{badge} **{overall}**"
    if explanation:
        body += f"\n\n_{explanation}_"
    return f"### 💬 Sentiment\n\n{body}\n"


def render_topics_markdown(topics: list) -> str:
    if not topics:
        return "### 🗂️ Topics Discussed\n\n_No topics identified._\n"
    lines = []
    for t in topics:
        topic_name = t.get("topic", "Untitled") if isinstance(t, dict) else str(t)
        description = t.get("description", "") if isinstance(t, dict) else ""
        if description:
            lines.append(f"- **{topic_name}** — {description}")
        else:
            lines.append(f"- **{topic_name}**")
    return "### 🗂️ Topics Discussed\n\n" + "\n".join(lines) + "\n"


def action_items_to_table(action_items: list) -> list[list[str]]:
    """Convert structured action-item dicts into rows for a gr.Dataframe.

    Always returns at least one row so the table never renders as a blank
    void — an explicit "no action items" row is clearer than an empty grid.
    """
    if not action_items:
        return [["No action items identified.", "—", "—", "—"]]

    rows = []
    for item in action_items:
        if not isinstance(item, dict):
            continue
        rows.append(
            [
                item.get("task") or "Not specified",
                item.get("owner") or "Not specified",
                item.get("deadline") or "Not specified",
                item.get("priority") or "Medium",
            ]
        )
    return rows or [["No action items identified.", "—", "—", "—"]]


def render_insights_markdown(meeting_insights: dict) -> str:
    duration_label = insights.format_duration(meeting_insights.get("duration_seconds", 0))
    word_count = meeting_insights.get("word_count", 0)
    wpm = meeting_insights.get("speaking_rate_wpm", 0)
    speakers = meeting_insights.get("estimated_speakers", 1)
    wpm_note = insights.speaking_rate_note(wpm)

    lines = [
        "### 📊 Meeting Insights",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| ⏱️ Duration | {duration_label} |",
        f"| 📝 Word Count | {word_count:,} |",
        f"| 🗣️ Speaking Rate | {wpm} WPM{wpm_note} |",
        f"| 👥 Estimated Speakers | ~{speakers} _(approximate)_ |",
    ]
    return "\n".join(lines) + "\n"
