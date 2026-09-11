"""Low-level audio helpers and small shared utilities.

Each function does exactly one thing so the pipeline nodes stay short,
focused and easy to test in isolation.
"""

import importlib.util
import io

import numpy as np
import soundfile as sf

from config import (
    ERROR_PREFIX,
    PLACEHOLDER_AMPLITUDE,
    PLACEHOLDER_DURATION_S,
    PLACEHOLDER_FREQUENCY_HZ,
    PLACEHOLDER_SAMPLE_RATE,
)


def read_audio(source, sample_rate=None):
    """Đọc audio thành mono float32, tùy chọn resample về sample_rate."""
    data, source_rate = sf.read(source, dtype="float32")
    data = to_mono(data)
    if sample_rate and source_rate != sample_rate:
        data = resample_audio(data, source_rate, sample_rate)
    return data, source_rate


def to_mono(data: np.ndarray) -> np.ndarray:
    """Converts multi-channel audio into mono by averaging the channels."""
    if data.ndim > 1:
        return np.mean(data, axis=1)
    return data


def resample_audio(data: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resamples the audio to target_rate using linear interpolation."""
    target_len = int(len(data) * target_rate / source_rate)
    return np.interp(
        np.linspace(0, len(data) - 1, target_len),
        np.arange(len(data)),
        data,
    ).astype(np.float32)


def encode_wav(data: np.ndarray, sample_rate: int) -> bytes:
    """Encodes float audio samples into a WAV byte buffer."""
    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, data, sample_rate, format="WAV")
    return wav_buffer.getvalue()


def generate_placeholder_audio() -> bytes:
    """Generates a short 440 Hz tone used when no TTS backend is available."""
    sample_rate = PLACEHOLDER_SAMPLE_RATE
    duration = PLACEHOLDER_DURATION_S
    tone = PLACEHOLDER_AMPLITUDE * np.sin(
        2 * np.pi * PLACEHOLDER_FREQUENCY_HZ * np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    ).astype(np.float32)
    return encode_wav(tone, sample_rate)


def module_available(module_name: str) -> bool:
    """Returns True if the given Python module can be imported."""
    return importlib.util.find_spec(module_name) is not None


def torch_cuda_available() -> bool:
    """Returns True if PyTorch is installed and CUDA is available."""
    if not module_available("torch"):
        return False
    import torch
    return torch.cuda.is_available()


def failure_message(detail: str) -> str:
    """Formats an error message using the shared [ERROR] prefix."""
    return f"{ERROR_PREFIX} {detail}"