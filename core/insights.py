"""
Locally-computed meeting metrics. None of these need an LLM call —
keeping them separate from llm_client.py keeps the "free" computations
visibly distinct from the ones that cost an inference call.
"""

import config


def compute_meeting_insights(
    transcript: str, duration_seconds: float, estimated_speakers: int
) -> dict:
    """Returns word count, speaking rate (WPM), and packages the already
    known duration/speaker estimate into one insights dict."""
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
    """Human-readable duration, e.g. '1h 2m 5s'."""
    if seconds <= 0:
        return "Unknown"
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    return f"{minutes}m {secs}s"


def speaking_rate_note(wpm: int) -> str:
    """Returns a short qualifier if WPM falls outside the typical
    conversational range, else an empty string."""
    if not wpm:
        return ""
    low, high = config.NORMAL_WPM_RANGE
    if wpm < low:
        return " (slower than typical conversational pace)"
    if wpm > high:
        return " (faster than typical conversational pace)"
    return ""
