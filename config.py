"""
Central configuration for the AI Meeting Assistant.

Every path, URL, and tunable constant lives here so nothing is hardcoded
inside business logic. Override any of these via environment variables
(or a .env file) without touching code.
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # python-dotenv is optional — the app still works with plain env vars
    pass

# --- Project root ---
BASE_DIR = Path(__file__).resolve().parent

# --- Whisper.cpp ---
WHISPER_CLI_PATH = os.getenv(
    "WHISPER_CLI_PATH",
    str(BASE_DIR / "whisper.cpp" / "whisper-cli.exe"),
)
WHISPER_MODEL_DIR = os.getenv(
    "WHISPER_MODEL_DIR",
    str(BASE_DIR / "whisper.cpp" / "models"),
)
VALID_WHISPER_MODEL_KEYWORDS = ["base", "small", "medium", "large", "large-V3"]

# --- Ollama ---
OLLAMA_SERVER_URL = os.getenv("OLLAMA_SERVER_URL", "http://localhost:11434")
OLLAMA_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "300"))

# --- ffmpeg / ffprobe ---
FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")
FFPROBE_BIN = os.getenv("FFPROBE_BIN", "ffprobe")

# --- App directories ---
TEMP_DIR = Path(os.getenv("TEMP_DIR", str(BASE_DIR / "temp")))
TEMP_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(BASE_DIR / "outputs")))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = Path(os.getenv("LOG_DIR", str(BASE_DIR / "logs")))
LOG_DIR.mkdir(parents=True, exist_ok=True)

# --- App behaviour ---
NORMAL_WPM_RANGE = (110, 170)

APP_NAME = "AI Powered Meeting Summarizer"
APP_VERSION = "2.0.0"
