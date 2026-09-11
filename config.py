"""Application configuration: every tunable parameter and shared constant.

Centralising the parameters here makes the pipeline easy to tune and
guarantees that every module uses the same values.
"""

import os

from dotenv import load_dotenv

# Load the .env file (holds the Groq API key) into the environment variables
load_dotenv()

# ------------------------------------------------ App / server
APP_TITLE = "Dịch Giọng Nói Việt - Anh"  # Gradio window title
SERVER_NAME = "0.0.0.0"                  # listen on every network interface
DEFAULT_SERVER_PORT = 7860               # fallback port (host may set $PORT)
SHARE_LINK = True                        # create a temporary public link

# ----------------------------------------------------- Audio
AUDIO_SAMPLE_RATE = 16000       # Whisper works best with 16 kHz mono audio
PLACEHOLDER_SAMPLE_RATE = 22050  # sample rate of the fallback beep
PLACEHOLDER_FREQUENCY_HZ = 440   # frequency of the fallback beep (A4)
PLACEHOLDER_AMPLITUDE = 0.3      # volume of the fallback beep
PLACEHOLDER_DURATION_S = 0.8     # length of the fallback beep

# ---------------------------------------------------- Models
# Speech-to-text (faster-whisper)
WHISPER_MODEL_NAME = "large-v3-turbo"  # Vietnamese speech model
STT_LANGUAGE = "vi"                    # speech recognition language
STT_BEAM_SIZE = 1                      # smaller beam = faster decoding
STT_VAD_FILTER = True                  # skip silence before decoding (VAD)

# Translation (Groq API)
TRANSLATION_MODEL = "openai/gpt-oss-20b"  # model used by the Groq API
GROQ_API_KEY_ENV = "GROQ_API_KEY"         # env var name holding the API key

MAX_TOKENS_CAP = 8192      # hard cap on tokens per translation request
TOKENS_PER_WORD = 3        # estimate: ~3 tokens per English word
TOKENS_HEADROOM = 1280     # safety margin on top of the word estimate
MAX_CHUNK_CHARS = 1200     # max size of one text chunk sent to the API

# Text-to-speech (Kokoro)
TTS_SAMPLE_RATE = 24000            # Kokoro-TTS native sample rate
KOKORO_VOICE = "af_heart"          # warm natural English voice
KOKORO_SPEED = 1.0                 # playback speed multiplier
KOKORO_LANG_CODE = "a"             # 'a' = American English (vn -> en pipeline)
KOKORO_REPO_ID = "hexgrad/Kokoro-82M"

# ---------------------------------------------------- Errors
ERROR_PREFIX = "[ERROR]"  # shared prefix of every error returned to the UI