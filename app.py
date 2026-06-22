import subprocess
import os
import re
import gradio as gr
import requests
import json
from packaging.version import parse as parse_version

OLLAMA_SERVER_URL = "http://localhost:11434"
WHISPER_MODEL_DIR = "./whisper.cpp/models"

# Gradio moved `theme`/`css` from the Blocks() constructor to launch() in v6,
# and dropped `show_copy_button` support on some components. Detect the
# installed version once so this file runs unmodified on old or new Gradio.
_GRADIO_V6_PLUS = parse_version(gr.__version__) >= parse_version("6.0.0")


# ---------------------------------------------------------------------------
# Backend logic — UNCHANGED, just lightly hardened so the UI has something
# useful to show on failure instead of a raw traceback.
# ---------------------------------------------------------------------------


def get_available_models() -> list[str]:
    response = requests.get(f"{OLLAMA_SERVER_URL}/api/tags")
    if response.status_code == 200:
        models = response.json()["models"]
        llm_model_names = [model["model"] for model in models]
        return llm_model_names
    else:
        raise Exception(
            f"Failed to retrieve models from Ollama server: {response.text}"
        )


def get_available_whisper_models() -> list[str]:
    valid_models = ["base", "small", "medium", "large", "large-V3"]
    model_files = [f for f in os.listdir(WHISPER_MODEL_DIR) if f.endswith(".bin")]

    whisper_models = [
        os.path.splitext(f)[0].replace("ggml-", "")
        for f in model_files
        if any(valid_model in f for valid_model in valid_models) and "test" not in f
    ]

    whisper_models = list(set(whisper_models))
    return whisper_models


def summarize_with_model(llm_model_name: str, context: str, text: str) -> dict:
    """Single Ollama call that returns a fully structured meeting analysis."""
    prompt = f"""You are an assistant that analyzes meeting transcripts and
returns ONLY a single valid JSON object — no prose, no markdown fences,
no commentary before or after.

Context: {context if context else 'No additional context provided.'}

Transcript:
{text}

Return a JSON object with EXACTLY this shape:

{{
  "tldr": "2-3 sentence high-level summary",
  "key_decisions": ["decision 1", "decision 2"],
  "action_items": [
    {{"task": "...", "owner": "Not specified", "deadline": "Not specified", "priority": "High|Medium|Low"}}
  ],
  "participants": ["name or role 1", "name or role 2"],
  "sentiment": {{"overall": "Positive|Neutral|Negative|Mixed", "explanation": "one short sentence"}},
  "topics": [{{"topic": "short topic name", "description": "one short sentence"}}]
}}

Rules:
- If a list would be empty, return an empty array [] — never omit the key.
- If owner or deadline isn't mentioned for a task, use the string "Not specified".
- priority must be inferred from urgency/context; default to "Medium" if unclear.
- participants should list names if spoken, otherwise inferred roles (e.g. "Speaker 1").
- Output must be valid JSON and nothing else.
"""

    headers = {"Content-Type": "application/json"}
    data = {
        "model": llm_model_name,
        "prompt": prompt,
        "format": "json",
    }

    response = requests.post(
        f"{OLLAMA_SERVER_URL}/api/generate", json=data, headers=headers, stream=True
    )

    empty_result = {
        "tldr": "",
        "key_decisions": [],
        "action_items": [],
        "participants": [],
        "sentiment": {"overall": "Unknown", "explanation": ""},
        "topics": [],
    }

    if response.status_code != 200:
        raise Exception(
            f"Failed to summarize with model {llm_model_name}: {response.text}"
        )

    full_response = ""
    try:
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode("utf-8")
                json_line = json.loads(decoded_line)
                full_response += json_line.get("response", "")
                if json_line.get("done", False):
                    break
    except json.JSONDecodeError:
        print("Error: Invalid JSON in Ollama stream.")
        empty_result["tldr"] = "Failed to parse server response."
        return empty_result

    try:
        parsed = json.loads(full_response)
    except json.JSONDecodeError:
        print("Error: Model did not return valid JSON. Raw output kept in tldr.")
        empty_result["tldr"] = full_response.strip()
        return empty_result

    result = empty_result.copy()
    result["tldr"] = parsed.get("tldr", "") or ""
    result["key_decisions"] = parsed.get("key_decisions") or []
    result["action_items"] = parsed.get("action_items") or []
    result["participants"] = parsed.get("participants") or []
    result["sentiment"] = parsed.get("sentiment") or empty_result["sentiment"]
    result["topics"] = parsed.get("topics") or []

    return result


def preprocess_audio_file(audio_file_path: str) -> str:
    output_wav_file = f"{os.path.splitext(audio_file_path)[0]}_converted.wav"
    cmd = f'ffmpeg -y -i "{audio_file_path}" -ar 16000 -ac 1 "{output_wav_file}"'
    subprocess.run(cmd, shell=True, check=True)
    return output_wav_file


# ---------------------------------------------------------------------------
# Frontend helpers
# ---------------------------------------------------------------------------


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
        topic_name = t.get("topic", "Untopic") if isinstance(t, dict) else str(t)
        description = t.get("description", "") if isinstance(t, dict) else ""
        if description:
            lines.append(f"- **{topic_name}** — {description}")
        else:
            lines.append(f"- **{topic_name}**")
    return "### 🗂️ Topics Discussed\n\n" + "\n".join(lines) + "\n"


def action_items_to_table(action_items: list) -> list[list[str]]:
    if not action_items:
        return [["No action items identified.", "—", "—", "—"]]

    rows = []
    for item in action_items:
        if not isinstance(item, dict):
            continue
        rows.append(
            [
                item.get("task", "Not specified") or "Not specified",
                item.get("owner", "Not specified") or "Not specified",
                item.get("deadline", "Not specified") or "Not specified",
                item.get("priority", "Medium") or "Medium",
            ]
        )
    return rows or [["No action items identified.", "—", "—", "—"]]


NORMAL_WPM_RANGE = (110, 170)


def get_audio_duration_seconds(audio_file_path: str) -> float:
    try:
        cmd = (
            f'ffprobe -v error -show_entries format=duration '
            f'-of default=noprint_wrappers=1:nokey=1 "{audio_file_path}"'
        )
        result = subprocess.run(
            cmd, shell=True, check=True, capture_output=True, text=True
        )
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        return 0.0


def estimate_speakers_from_whisper_txt(whisper_output_file: str) -> int:
    try:
        with open(whisper_output_file, "r", encoding="utf-8") as f:
            content = f.read()
    except (FileNotFoundError, OSError):
        return 1

    blocks = [b for b in re.split(r"\n\s*\n", content) if b.strip()]
    turn_count = max(len(blocks), 1)
    estimated = min(max(1, round(turn_count / 3)), 8)
    return estimated


def compute_meeting_insights(transcript: str, duration_seconds: float, estimated_speakers: int) -> dict:
    word_count = len(transcript.split())

    if duration_seconds > 0:
        minutes = duration_seconds / 60
        speaking_rate_wpm = round(word_count / minutes) if minutes > 0 else 0
    else:
        speaking_rate_wpm = 0

    return {
        "duration_seconds": duration_seconds,
        "word_count": word_count,
        "speaking_rate_wpm": speaking_rate_wpm,
        "estimated_speakers": estimated_speakers,
    }


def format_duration(seconds: float) -> str:
    if seconds <= 0:
        return "Unknown"
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    return f"{minutes}m {secs}s"


def render_insights_markdown(insights: dict) -> str:
    duration_label = format_duration(insights.get("duration_seconds", 0))
    word_count = insights.get("word_count", 0)
    wpm = insights.get("speaking_rate_wpm", 0)
    speakers = insights.get("estimated_speakers", 1)

    wpm_note = ""
    if wpm:
        low, high = NORMAL_WPM_RANGE
        if wpm < low:
            wpm_note = " _(slower than typical conversational pace)_"
        elif wpm > high:
            wpm_note = " _(faster than typical conversational pace)_"

    lines = [
        "### 📊 Meeting Insights",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| ⏱️ Duration | {duration_label} |",
        f"| 📝 Word Count | {word_count:,} |",
        f"| 🗣️ Speaking Rate | {wpm} WPM{wpm_note} |",
        f"| 👥 Estimated Speakers | ~{speakers} _(approximate)_ |",
    ]
    return "\n".join(lines) + "\n"


def empty_outputs(error_message: str = "") -> tuple:
    return (
        render_section_markdown("TL;DR", "", "🧭"),
        render_list_markdown("Key Decisions", [], "✅", "None identified."),
        action_items_to_table([]),
        render_list_markdown("Participants", [], "👥", "Not identifiable from transcript."),
        render_sentiment_markdown({}),
        render_topics_markdown([]),
        render_insights_markdown({}),
        None,
        "",
        error_message,
    )


def process_meeting(
    audio,
    context: str,
    whisper_model_name: str,
    llm_model_name: str,
    progress=gr.Progress(track_tqdm=False),
):
    if audio is None:
        return empty_outputs("⚠️ **No audio file provided.** Upload a file before submitting.")

    audio_file_wav = None
    whisper_output_file = None

    try:
        progress(0.05, desc="Reading audio metadata…")
        duration_seconds = get_audio_duration_seconds(audio)

        progress(0.15, desc="Preprocessing audio…")
        audio_file_wav = preprocess_audio_file(audio)

        progress(0.35, desc=f"Transcribing with Whisper ({whisper_model_name})…")
        whisper_command = (
            f'"C:\\Users\\akhil\\AI-Powered-Meeting-Summarizer\\whisper.cpp\\whisper-cli.exe" '
            f'-m "C:\\Users\\akhil\\AI-Powered-Meeting-Summarizer\\whisper.cpp\\models\\ggml-{whisper_model_name}.bin" '
            f'-f "{audio_file_wav}" -otxt'
        )
        subprocess.run(whisper_command, shell=True, check=True)

        whisper_output_file = audio_file_wav + ".txt"
        with open(whisper_output_file, "r", encoding="utf-8") as f:
            transcript = f.read()

        transcript_file = "transcript.txt"
        with open(transcript_file, "w", encoding="utf-8") as transcript_f:
            transcript_f.write(transcript)

        progress(0.55, desc="Estimating speakers…")
        estimated_speakers = estimate_speakers_from_whisper_txt(whisper_output_file)
        insights = compute_meeting_insights(transcript, duration_seconds, estimated_speakers)

        progress(0.7, desc=f"Analyzing with {llm_model_name}…")
        analysis = summarize_with_model(llm_model_name, context, transcript)

        progress(0.95, desc="Formatting results…")

        os.remove(audio_file_wav)
        os.remove(whisper_output_file)

        progress(1.0, desc="Done")

        return (
            render_section_markdown("TL;DR", analysis["tldr"], "🧭"),
            render_list_markdown("Key Decisions", analysis["key_decisions"], "✅", "None identified."),
            action_items_to_table(analysis["action_items"]),
            render_list_markdown(
                "Participants", analysis["participants"], "👥", "Not identifiable from transcript."
            ),
            render_sentiment_markdown(analysis["sentiment"]),
            render_topics_markdown(analysis["topics"]),
            render_insights_markdown(insights),
            transcript_file,
            transcript,
            "",
        )

    except subprocess.CalledProcessError as e:
        return empty_outputs(
            f"❌ **A command failed while processing the file.**\n\n"
            f"`{e.cmd}`\n\nCheck that ffmpeg and whisper.cpp are installed and "
            f"the paths in the script are correct for your machine."
        )
    except requests.exceptions.ConnectionError:
        return empty_outputs(
            f"❌ **Couldn't reach the Ollama server** at `{OLLAMA_SERVER_URL}`.\n\n"
            f"Make sure Ollama is running (`ollama serve`) and try again."
        )
    except FileNotFoundError as e:
        return empty_outputs(
            f"❌ **File not found:** `{e.filename}`.\n\n"
            f"Whisper may have failed to produce a transcript — check the "
            f"console log for details."
        )
    except Exception as e:
        return empty_outputs(f"❌ **Unexpected error:** {e}")
    finally:
        for path in (audio_file_wav, whisper_output_file):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# Theme — "Control Room" palette: deep charcoal-navy surfaces with an
# electric indigo signal color. Chosen deliberately over a default
# dark+orange combo: indigo/violet reads as "intelligence tooling" in the
# same visual family as Linear, Notion AI, Otter — orange reads as
# "alert/marketing CTA", which fights against a calm analysis tool.
# ---------------------------------------------------------------------------

COLOR_BG = "#0A0E14"          # page background
COLOR_SURFACE = "#11151D"     # card/panel background
COLOR_SURFACE_RAISED = "#161B26"  # nested/hovered surface
COLOR_BORDER = "#232A38"      # hairline borders
COLOR_BORDER_STRONG = "#2E3648"
COLOR_TEXT = "#E7EAF0"        # primary text
COLOR_TEXT_MUTED = "#8993A8"  # secondary text
COLOR_ACCENT = "#6366F1"      # signature indigo
COLOR_ACCENT_SOFT = "#818CF8"
COLOR_ACCENT_DIM = "rgba(99, 102, 241, 0.12)"
COLOR_SUCCESS = "#34D399"
COLOR_WARNING = "#FBBF24"
COLOR_DANGER = "#F87171"

theme = gr.themes.Base(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Manrope"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill=COLOR_BG,
    body_background_fill_dark=COLOR_BG,
    background_fill_primary=COLOR_SURFACE,
    background_fill_primary_dark=COLOR_SURFACE,
    background_fill_secondary=COLOR_SURFACE_RAISED,
    background_fill_secondary_dark=COLOR_SURFACE_RAISED,
    block_background_fill=COLOR_SURFACE,
    block_background_fill_dark=COLOR_SURFACE,
    block_border_color=COLOR_BORDER,
    block_border_color_dark=COLOR_BORDER,
    block_label_text_color=COLOR_TEXT_MUTED,
    block_label_text_color_dark=COLOR_TEXT_MUTED,
    block_title_text_color=COLOR_TEXT,
    block_title_text_color_dark=COLOR_TEXT,
    body_text_color=COLOR_TEXT,
    body_text_color_dark=COLOR_TEXT,
    body_text_color_subdued=COLOR_TEXT_MUTED,
    border_color_accent=COLOR_ACCENT,
    border_color_accent_dark=COLOR_ACCENT,
    button_primary_background_fill=COLOR_ACCENT,
    button_primary_background_fill_hover=COLOR_ACCENT_SOFT,
    button_primary_text_color="#FFFFFF",
    button_secondary_background_fill=COLOR_SURFACE_RAISED,
    button_secondary_background_fill_hover=COLOR_BORDER_STRONG,
    button_secondary_text_color=COLOR_TEXT,
    button_secondary_border_color=COLOR_BORDER_STRONG,
    input_background_fill=COLOR_BG,
    input_background_fill_dark=COLOR_BG,
    input_border_color=COLOR_BORDER,
    input_border_color_dark=COLOR_BORDER,
    slider_color=COLOR_ACCENT,
    shadow_drop="0 1px 2px rgba(0,0,0,0.5)",
)

CUSTOM_CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@500;600;700;800&family=Inter:wght@400;500;600&display=swap');

:root {{
    --accent: {COLOR_ACCENT};
    --accent-soft: {COLOR_ACCENT_SOFT};
    --accent-dim: {COLOR_ACCENT_DIM};
    --surface: {COLOR_SURFACE};
    --surface-raised: {COLOR_SURFACE_RAISED};
    --border: {COLOR_BORDER};
    --border-strong: {COLOR_BORDER_STRONG};
    --text: {COLOR_TEXT};
    --text-muted: {COLOR_TEXT_MUTED};
    --success: {COLOR_SUCCESS};
    --warning: {COLOR_WARNING};
    --danger: {COLOR_DANGER};
}}

.gradio-container {{
    max-width: 1320px !important;
    margin: 0 auto !important;
    font-family: 'Inter', ui-sans-serif, sans-serif !important;
}}

body, .gradio-container {{
    background: radial-gradient(ellipse 80% 50% at 50% -10%, rgba(99,102,241,0.08), transparent),
                {COLOR_BG} !important;
}}

/* ---------- Header ---------- */
#app-header {{
    padding: 28px 4px 8px 4px !important;
    border-bottom: 1px solid var(--border);
    margin-bottom: 20px !important;
}}
#brand-row {{
    display: flex;
    align-items: center;
    gap: 14px;
}}
#brand-mark {{
    width: 42px;
    height: 42px;
    border-radius: 11px;
    background: linear-gradient(135deg, var(--accent), #A78BFA);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    box-shadow: 0 4px 18px rgba(99,102,241,0.35);
    flex-shrink: 0;
}}
#brand-text h1 {{
    font-family: 'Manrope', sans-serif !important;
    font-weight: 800 !important;
    letter-spacing: -0.025em !important;
    font-size: 1.65rem !important;
    margin: 0 !important;
    color: var(--text) !important;
    line-height: 1.2 !important;
}}
#brand-text p {{
    margin: 2px 0 0 0 !important;
    color: var(--text-muted) !important;
    font-size: 0.88rem !important;
}}
#privacy-badge {{
    margin-left: auto;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.78rem;
    color: var(--success);
    background: rgba(52, 211, 153, 0.1);
    border: 1px solid rgba(52, 211, 153, 0.25);
    padding: 6px 12px;
    border-radius: 999px;
    font-weight: 500;
    white-space: nowrap;
}}

/* ---------- Step indicator ---------- */
#step-indicator {{
    display: flex;
    align-items: center;
    gap: 0;
    padding: 4px 0 22px 0;
}}
.step-pill {{
    display: flex;
    align-items: center;
    gap: 9px;
    padding: 7px 16px 7px 10px;
    border-radius: 999px;
    background: var(--surface);
    border: 1px solid var(--border);
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--text-muted);
}}
.step-pill .num {{
    width: 22px;
    height: 22px;
    border-radius: 50%;
    background: var(--surface-raised);
    color: var(--text-muted);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    font-weight: 700;
}}
.step-pill.active {{
    background: var(--accent-dim);
    border-color: rgba(99,102,241,0.4);
    color: var(--accent-soft);
}}
.step-pill.active .num {{
    background: var(--accent);
    color: #fff;
}}
.step-connector {{
    flex: 1;
    height: 1px;
    background: var(--border);
    margin: 0 2px;
    min-width: 16px;
}}

/* Global safety net: Gradio's Markdown component wraps output in
   .prose/.md spans that default to a light background outside dark-mode
   contexts. Neutralize everywhere, not just inside .section-card. */
[data-testid="markdown"],
[data-testid="markdown-wrapper"],
.prose,
span.md {{
    background: transparent !important;
}}
.prose, .prose p, .prose li, .prose td, .prose th {{
    color: var(--text) !important;
}}

/* Gradio's internal `.styler` wrapper carries its own slate background
   from the base theme — this is the actual element responsible for the
   pale rectangle that appears behind every Markdown block if left
   untouched. */
.section-card .styler {{
    background: transparent !important;
}}

/* ---------- Cards ---------- */
.section-card {{
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    padding: 18px 20px !important;
    background: linear-gradient(155deg, var(--surface), var(--surface) 60%, rgba(99,102,241,0.025)) !important;
    transition: border-color 0.15s ease;
    margin-bottom: 4px;
}}
/* Make side-by-side cards (Sentiment / Topics) equal height regardless
   of content length, instead of each hugging its own text. */
.tabitem {{
    padding-top: 16px !important;
}}
.tabitem > .form, .tabitem > div > .form {{
    gap: 14px !important;
}}
.tabitem .row {{
    align-items: stretch !important;
}}
.tabitem .row > .form,
.tabitem .row > div {{
    display: flex !important;
}}
.tabitem .row .section-card {{
    flex: 1;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
}}
.section-card:hover {{
    border-color: var(--border-strong) !important;
}}

/* Gradio wraps Markdown output in .prose / .md spans that carry their own
   light-theme background by default — must be neutralized everywhere or
   every card shows a pale box behind the text. */
.section-card .prose,
.section-card .md,
.section-card [data-testid="markdown"],
.section-card [data-testid="markdown-wrapper"],
.section-card span.md {{
    background: transparent !important;
    color: var(--text-muted) !important;
}}
.section-card h3 {{
    margin-top: 0 !important;
    margin-bottom: 10px !important;
    font-size: 0.92rem !important;
    font-weight: 700 !important;
    color: var(--text) !important;
    font-family: 'Manrope', sans-serif !important;
}}
.section-card p, .section-card li {{
    color: var(--text-muted) !important;
    font-size: 0.92rem !important;
    line-height: 1.6 !important;
    background: transparent !important;
}}
.section-card table {{
    font-size: 0.88rem !important;
    background: transparent !important;
}}
.section-card strong {{
    color: var(--text) !important;
}}

/* KPI-style insights table */
.section-card table th {{
    color: var(--text-muted) !important;
    font-weight: 600 !important;
    border-bottom: 1px solid var(--border) !important;
    text-align: left !important;
}}
.section-card table td {{
    border-bottom: 1px solid var(--border) !important;
    color: var(--text) !important;
    padding: 8px 4px !important;
}}

/* ---------- Error box ---------- */
/* Only paint the red border/background once there's real text inside —
   otherwise an empty Markdown component still renders its wrapper div and
   would show as a bare red outline with nothing in it. */
#error-box:has(p) {{
    border: 1px solid rgba(248,113,113,0.3) !important;
    background: rgba(248,113,113,0.08) !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
}}
#error-box {{
    border: none !important;
    background: transparent !important;
    min-height: 0 !important;
}}
#error-box p {{
    color: var(--danger) !important;
    margin: 0 !important;
}}

/* ---------- Buttons ---------- */
button.primary {{
    font-weight: 700 !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 14px rgba(99,102,241,0.3) !important;
    transition: transform 0.1s ease, box-shadow 0.15s ease !important;
}}
button.primary:hover {{
    box-shadow: 0 6px 20px rgba(99,102,241,0.45) !important;
    transform: translateY(-1px);
}}
button.secondary {{
    font-weight: 600 !important;
    border-radius: 10px !important;
}}

/* ---------- Tabs ---------- */
.tabs > .tab-nav {{
    border-bottom: 1px solid var(--border) !important;
    gap: 4px !important;
}}
.tabs > .tab-nav button {{
    font-weight: 600 !important;
    color: var(--text-muted) !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 9px 16px !important;
}}
.tabs > .tab-nav button.selected {{
    color: var(--accent-soft) !important;
    background: var(--accent-dim) !important;
}}

/* ---------- Inputs ---------- */
label span {{
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    color: var(--text) !important;
}}
.gr-box, textarea, input[type="text"] {{
    border-radius: 10px !important;
}}

footer {{display: none !important;}}
"""

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


def build_blocks() -> gr.Blocks:
    if _GRADIO_V6_PLUS:
        return gr.Blocks(title="Meeting Summarizer")
    return gr.Blocks(theme=theme, css=CUSTOM_CSS, title="Meeting Summarizer")


if __name__ == "__main__":
    ollama_models = get_available_models()
    whisper_models = get_available_whisper_models()

    with build_blocks() as demo:
        with gr.Column(elem_id="app-header"):
            with gr.Row(elem_id="brand-row"):
                gr.HTML(
                    """
                    <div id="brand-mark">🎙️</div>
                    <div id="brand-text">
                        <h1>Meeting Summarizer</h1>
                        <p>Transcript &amp; structured analysis for every recording</p>
                    </div>
                    <div id="privacy-badge">● Runs fully locally</div>
                    """
                )

        gr.HTML(
            """
            <div id="step-indicator">
                <div class="step-pill active"><span class="num">1</span> Upload</div>
                <div class="step-connector"></div>
                <div class="step-pill"><span class="num">2</span> Transcribe</div>
                <div class="step-connector"></div>
                <div class="step-pill"><span class="num">3</span> Analyze</div>
                <div class="step-connector"></div>
                <div class="step-pill"><span class="num">4</span> Export</div>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=5):
                audio_input = gr.Audio(
                    type="filepath",
                    label="Upload an audio file",
                )

                context_input = gr.Textbox(
                    label="Context (optional)",
                    placeholder="e.g. Weekly product sync between Design and Engineering",
                    lines=2,
                )

                with gr.Row():
                    whisper_dropdown = gr.Dropdown(
                        choices=whisper_models,
                        label="Whisper model",
                        value=whisper_models[0] if whisper_models else None,
                        info="Larger = more accurate, slower",
                    )
                    llm_dropdown = gr.Dropdown(
                        choices=ollama_models,
                        label="Summarization model",
                        value=ollama_models[0] if ollama_models else None,
                        info="Served locally via Ollama",
                    )

                with gr.Row():
                    clear_btn = gr.Button("Clear", variant="secondary")
                    submit_btn = gr.Button("Summarize meeting", variant="primary")

                error_box = gr.Markdown(visible=True, elem_id="error-box")

            with gr.Column(scale=6):
                with gr.Tabs():
                    with gr.Tab("📋 Summary"):
                        with gr.Group(elem_classes="section-card"):
                            tldr_md = gr.Markdown(render_section_markdown("TL;DR", "", "🧭"))
                        with gr.Group(elem_classes="section-card"):
                            decisions_md = gr.Markdown(
                                render_list_markdown("Key Decisions", [], "✅", "None identified.")
                            )
                        with gr.Group(elem_classes="section-card"):
                            participants_md = gr.Markdown(
                                render_list_markdown(
                                    "Participants", [], "👥", "Not identifiable from transcript."
                                )
                            )
                        with gr.Row():
                            with gr.Group(elem_classes="section-card"):
                                sentiment_md = gr.Markdown(render_sentiment_markdown({}))
                            with gr.Group(elem_classes="section-card"):
                                topics_md = gr.Markdown(render_topics_markdown([]))

                    with gr.Tab("📌 Action Items"):
                        with gr.Group(elem_classes="section-card"):
                            gr.Markdown("### 📌 Tasks, Owners & Deadlines")
                            action_items_table = gr.Dataframe(
                                headers=["Task", "Owner", "Deadline", "Priority"],
                                value=action_items_to_table([]),
                                interactive=False,
                                wrap=True,
                            )

                    with gr.Tab("📊 Insights"):
                        with gr.Group(elem_classes="section-card"):
                            insights_md = gr.Markdown(render_insights_markdown({}))

                    with gr.Tab("📄 Transcript"):
                        with gr.Group(elem_classes="section-card"):
                            transcript_preview = gr.Textbox(
                                label="Transcript",
                                lines=16,
                                interactive=False,
                            )
                        transcript_file_out = gr.File(label="Download transcript (.txt)")

        submit_btn.click(
            fn=process_meeting,
            inputs=[audio_input, context_input, whisper_dropdown, llm_dropdown],
            outputs=[
                tldr_md,
                decisions_md,
                action_items_table,
                participants_md,
                sentiment_md,
                topics_md,
                insights_md,
                transcript_file_out,
                transcript_preview,
                error_box,
            ],
        )

        def clear_all():
            return (
                None,
                "",
                render_section_markdown("TL;DR", "", "🧭"),
                render_list_markdown("Key Decisions", [], "✅", "None identified."),
                action_items_to_table([]),
                render_list_markdown(
                    "Participants", [], "👥", "Not identifiable from transcript."
                ),
                render_sentiment_markdown({}),
                render_topics_markdown([]),
                render_insights_markdown({}),
                None,
                "",
                "",
            )

        clear_btn.click(
            fn=clear_all,
            inputs=[],
            outputs=[
                audio_input,
                context_input,
                tldr_md,
                decisions_md,
                action_items_table,
                participants_md,
                sentiment_md,
                topics_md,
                insights_md,
                transcript_file_out,
                transcript_preview,
                error_box,
            ],
        )

        demo.queue()

    if _GRADIO_V6_PLUS:
        demo.launch(debug=True, share=True, theme=theme, css=CUSTOM_CSS)
    else:
        demo.launch(debug=True, share=True)