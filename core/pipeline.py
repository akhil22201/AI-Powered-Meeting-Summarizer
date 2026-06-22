"""
Orchestrates the full Upload -> Transcribe -> Analyze flow.

This is the only module that calls across transcription, llm_client, and
insights together — UI code should only ever call run_pipeline(), never
reach into the lower-level modules directly. Keeping the orchestration in
one place means the temp-file lifecycle (create/cleanup) is guaranteed
correct regardless of which UI calls it.
"""

import uuid

import config
from core import insights, llm_client, transcription
from utils.logger import get_logger

log = get_logger(__name__)


def run_pipeline(
    audio_file_path: str,
    context: str,
    whisper_model_name: str,
    llm_model_name: str,
    progress_callback=None,
) -> dict:
    """Runs the complete pipeline and returns a single result dict:

    {
        "transcript": str,
        "transcript_file": str (path),
        "analysis": dict (from llm_client.analyze_transcript),
        "insights": dict (from insights.compute_meeting_insights),
    }

    `progress_callback`, if given, is called as progress_callback(fraction, desc)
    so the UI layer can drive any progress bar without this module knowing
    Gradio exists.

    Raises whatever the lower-level modules raise (subprocess.CalledProcessError,
    requests.exceptions.ConnectionError, FileNotFoundError, RuntimeError) —
    callers (the UI layer) are responsible for catching and displaying these.
    """

    def report(fraction: float, desc: str) -> None:
        log.info("[%.0f%%] %s", fraction * 100, desc)
        if progress_callback:
            progress_callback(fraction, desc)

    audio_file_wav = None
    try:
        report(0.05, "Reading audio metadata…")
        duration_seconds = transcription.get_audio_duration_seconds(audio_file_path)

        report(0.15, "Preprocessing audio…")
        audio_file_wav = transcription.preprocess_audio_file(audio_file_path)

        report(0.35, f"Transcribing with Whisper ({whisper_model_name})…")
        transcript = transcription.transcribe_audio(audio_file_wav, whisper_model_name)

        transcript_file = str(
            config.OUTPUT_DIR / f"transcript_{uuid.uuid4().hex[:8]}.txt"
        )
        with open(transcript_file, "w", encoding="utf-8") as f:
            f.write(transcript)

        report(0.55, "Estimating speakers…")
        estimated_speakers = transcription.estimate_speakers(transcript)
        meeting_insights = insights.compute_meeting_insights(
            transcript, duration_seconds, estimated_speakers
        )

        report(0.7, f"Analyzing with {llm_model_name}…")
        analysis = llm_client.analyze_transcript(llm_model_name, context, transcript)

        report(1.0, "Done")

        return {
            "transcript": transcript,
            "transcript_file": transcript_file,
            "analysis": analysis,
            "insights": meeting_insights,
        }
    finally:
        # Guaranteed cleanup of the intermediate WAV even if a later step
        # raises — the transcript .txt output is the only artifact meant
        # to survive the run.
        transcription.cleanup_temp_file(audio_file_wav)
