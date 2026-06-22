"""
Speech-to-text layer: audio preprocessing (ffmpeg), transcription
(whisper.cpp), and audio-derived metadata (duration, rough speaker count).
"""

import os
import re
import subprocess
import uuid

import config
from utils.logger import get_logger

log = get_logger(__name__)


def get_available_whisper_models() -> list[str]:
    """Scans the configured model directory for usable whisper.cpp .bin files."""
    model_dir = config.WHISPER_MODEL_DIR
    if not os.path.isdir(model_dir):
        log.warning("Whisper model directory not found: %s", model_dir)
        return []

    model_files = [f for f in os.listdir(model_dir) if f.endswith(".bin")]

    whisper_models = [
        os.path.splitext(f)[0].replace("ggml-", "")
        for f in model_files
        if any(kw in f for kw in config.VALID_WHISPER_MODEL_KEYWORDS)
        and "test" not in f
    ]

    return sorted(set(whisper_models))


def preprocess_audio_file(audio_file_path: str) -> str:
    """Converts any input audio file to 16kHz mono WAV — the format
    whisper.cpp expects. Returns the path to the converted file.

    Uses a UUID suffix instead of reusing the original filename so two
    concurrent requests on the same uploaded filename never collide.
    """
    unique_id = uuid.uuid4().hex[:8]
    output_wav_file = str(
        config.TEMP_DIR / f"{os.path.splitext(os.path.basename(audio_file_path))[0]}_{unique_id}.wav"
    )

    cmd = (
        f'{config.FFMPEG_BIN} -y -i "{audio_file_path}" '
        f'-ar 16000 -ac 1 "{output_wav_file}"'
    )
    log.info("Preprocessing audio: %s -> %s", audio_file_path, output_wav_file)
    subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
    return output_wav_file


def transcribe_audio(audio_file_wav: str, whisper_model_name: str) -> str:
    """Runs whisper.cpp on a preprocessed WAV file and returns the
    transcript text. Raises subprocess.CalledProcessError or
    FileNotFoundError on failure — callers should handle these.
    """
    model_path = os.path.join(
        config.WHISPER_MODEL_DIR, f"ggml-{whisper_model_name}.bin"
    )

    whisper_command = (
        f'"{config.WHISPER_CLI_PATH}" -m "{model_path}" '
        f'-f "{audio_file_wav}" -otxt'
    )
    log.info("Running whisper.cpp with model=%s", whisper_model_name)
    subprocess.run(whisper_command, shell=True, check=True, capture_output=True, text=True)

    whisper_output_file = audio_file_wav + ".txt"
    with open(whisper_output_file, "r", encoding="utf-8") as f:
        transcript = f.read()

    # Clean up whisper.cpp's own output file immediately — the transcript
    # text itself is returned to the caller, who owns persisting it.
    if os.path.exists(whisper_output_file):
        os.remove(whisper_output_file)

    return transcript


def get_audio_duration_seconds(audio_file_path: str) -> float:
    """Reads duration via ffprobe. Returns 0.0 if ffprobe isn't available
    or the file can't be probed — callers should treat 0.0 as 'unknown'.
    """
    try:
        cmd = (
            f'{config.FFPROBE_BIN} -v error -show_entries format=duration '
            f'-of default=noprint_wrappers=1:nokey=1 "{audio_file_path}"'
        )
        result = subprocess.run(
            cmd, shell=True, check=True, capture_output=True, text=True
        )
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as e:
        log.warning("Could not determine audio duration: %s", e)
        return 0.0


def estimate_speakers(transcript: str) -> int:
    """Very rough speaker-count heuristic.

    whisper.cpp's plain text output has no per-line timestamps, so true
    pause-based diarization isn't possible from this alone. As a pragmatic
    proxy, paragraph-like breaks in the transcript are treated as a stand-in
    for a change in speaker turn.

    This is intentionally approximate — true diarization needs a dedicated
    model (e.g. pyannote.audio) or whisper.cpp's --diarize flag on stereo
    recordings. The UI must label this value as "estimated".
    """
    blocks = [b for b in re.split(r"\n\s*\n", transcript) if b.strip()]
    turn_count = max(len(blocks), 1)
    estimated = min(max(1, round(turn_count / 3)), 8)
    return estimated


def cleanup_temp_file(path: str) -> None:
    """Best-effort delete that never raises — safe to call in finally blocks."""
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError as e:
            log.warning("Failed to remove temp file %s: %s", path, e)
