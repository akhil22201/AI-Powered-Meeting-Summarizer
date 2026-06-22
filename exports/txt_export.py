"""Plain-text transcript export — kept here even though it's a one-liner
so every export format is invoked the same way from pipeline/UI code."""


def export_transcript_txt(transcript: str, output_path: str) -> str:
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(transcript)
    return output_path
