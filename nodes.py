"""Pipeline nodes: each class wraps one step of the speech-translation flow.

AudioInputNode           loads and normalises the input audio file
SpeechToTextNode         Vietnamese speech -> Vietnamese text (faster-whisper)
TranslationNode          Vietnamese text -> English text (Groq API)
TextToSpeechNode         English text -> English audio (Kokoro-TTS)
SpeechTranslationPipeline  orchestrates the four nodes above
"""

from __future__ import annotations

import io
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

import config
from audio_utils import (
    encode_wav,
    failure_message,
    generate_placeholder_audio,
    module_available,
    read_audio,
    torch_cuda_available,
)


class AudioInputNode:
    """Loads an audio file and returns its samples as a ready-to-use WAV."""

    def __init__(self, sample_rate: int = config.AUDIO_SAMPLE_RATE):
        self.sample_rate = sample_rate

    def prepare_audio(self, audio_path: Optional[str]) -> Tuple[bytes, int]:
        """Returns (wav bytes, sample rate) for the given audio file."""
        if not audio_path:
            raise ValueError("No audio file was provided.")
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        data, _ = read_audio(str(path), self.sample_rate)
        return encode_wav(data, self.sample_rate), self.sample_rate


class SpeechToTextNode:
    """Transcribes Vietnamese speech into Vietnamese text (faster-whisper)."""

    def __init__(self, model_name: str = config.WHISPER_MODEL_NAME):
        self.model_name = model_name
        self.device = "cuda" if torch_cuda_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        print(f"[STT] Using {self.device.upper()} ({self.compute_type})")
        self.whisper_model = None
        self._load_model()

    def _load_model(self):
        if not module_available("faster_whisper"):
            print("[STT] faster-whisper is not installed; transcription will be skipped.")
            return
        try:
            from faster_whisper import WhisperModel
            print(f"[STT] Loading model '{self.model_name}'...")
            self.whisper_model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
                download_root=None,
            )
        except Exception as exc:
            print(f"[STT] Model load failed: {exc}")
            self.whisper_model = None

    def transcribe(self, audio_bytes: bytes, sample_rate: int = config.AUDIO_SAMPLE_RATE) -> str:
        """Transcribes the audio bytes and returns the Vietnamese transcript."""
        if self.whisper_model is None:
            return "[STT unavailable: faster-whisper model could not be loaded.]"
        try:
            audio_array, _ = read_audio(io.BytesIO(audio_bytes))
            segments, _ = self.whisper_model.transcribe(
                audio_array,
                language=config.STT_LANGUAGE,
                beam_size=config.STT_BEAM_SIZE,
                vad_filter=config.STT_VAD_FILTER,
            )
            return self._join_segment_texts(segments)
        except Exception as exc:
            return f"[STT error: {exc}]"

    def _join_segment_texts(self, segments) -> str:
        """Joins the non-empty transcribed segments into a single transcript."""
        transcript = " ".join(segment.text.strip() for segment in segments if segment.text)
        return transcript.strip() or "[STT returned no speech.]"


class TranslationNode:
    """Translates Vietnamese text into English using the Groq API (LLM)."""

    def __init__(self, model_name: str = config.TRANSLATION_MODEL):
        self.model_name = model_name
        self.client = None
        self.api_key_missing = False
        self._load_model()

    def _load_model(self):
        if not module_available("groq"):
            print("[LLM] groq package is not installed.")
            return
        api_key = os.getenv(config.GROQ_API_KEY_ENV)
        if not api_key:
            print("[LLM] GROQ_API_KEY not set!")
            self.api_key_missing = True
            return
        try:
            from groq import Groq
            self.client = Groq(api_key=api_key)
        except Exception as exc:
            print(f"[LLM] Groq initialization failed: {exc}")
            self.client = None

    def translate(self, text: str) -> str:
        """Translates a full text, splitting it into safe-sized chunks."""
        if self.api_key_missing:
            return failure_message("GROQ_API_KEY not set. Cannot translate. See setup instructions.")
        if self.client is None:
            return failure_message("LLM client not loaded. Check console output above.")
        cleaned = (text or "").strip()
        if not cleaned:
            return failure_message("No text to translate.")

        chunks = self._split_for_translation(cleaned)
        parts = [self._translate_chunk(chunk) for chunk in chunks]
        return self._join_translation_parts(parts)

    def _join_translation_parts(self, parts: list[str]) -> str:
        """Joins the translated chunks, or returns the first failure message."""
        for part in parts:
            if part.startswith(config.ERROR_PREFIX):
                return part
        return " ".join(parts).strip()

    def _split_for_translation(self, text: str, max_chars: int = config.MAX_CHUNK_CHARS) -> list[str]:
        """Splits a long text into chunks that fit within max_chars."""
        if len(text) <= max_chars:
            return [text]
        sentences = self._split_into_sentences(text)
        chunks = self._group_sentences_into_chunks(sentences, max_chars)
        result_chunks = self._split_long_chunks_by_word_limit(chunks, max_chars)
        return result_chunks or [text]

    def _split_into_sentences(self, text: str) -> list[str]:
        """Splits text at sentence-ending punctuation or line breaks."""
        sentences = re.split(r"(?<=[.!?…;])\s+|\n+", text)
        if len(sentences) == 1:
            sentences = re.split(r"(?<=[,])\s+", text)
        return sentences

    def _group_sentences_into_chunks(self, sentences: list[str], max_chars: int) -> list[str]:
        """Groups non-empty sentences into chunks that fit within max_chars."""
        chunks, current_chunk = [], ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if not current_chunk or len(current_chunk) + 1 + len(sentence) <= max_chars:
                current_chunk = f"{current_chunk} {sentence}".strip()
            else:
                chunks.append(current_chunk)
                current_chunk = sentence
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    def _split_long_chunks_by_word_limit(self, chunks: list[str], max_chars: int) -> list[str]:
        """Splits remaining over-long chunks on word boundaries."""
        result_chunks = []
        for chunk in chunks:
            if len(chunk) <= max_chars:
                result_chunks.append(chunk)
                continue
            words, char_count = [], 0
            for word in chunk.split():
                added_chars = len(word) + (1 if words else 0)
                if words and char_count + added_chars > max_chars:
                    result_chunks.append(" ".join(words))
                    words, char_count = [word], len(word)
                else:
                    words.append(word)
                    char_count += added_chars
            if words:
                result_chunks.append(" ".join(words))
        return result_chunks

    def _translate_chunk(self, text: str) -> str:
        """Translates one chunk, retrying once if the response was cut off."""
        messages = self._build_translation_messages(text)
        max_tokens = min(config.MAX_TOKENS_CAP, int(len(text.split()) * config.TOKENS_PER_WORD) + config.TOKENS_HEADROOM)

        for attempt in range(2):
            try:
                content, finish_reason = self._request_translation(messages, max_tokens)
                if content:
                    return content
                if finish_reason == "length" and attempt == 0:
                    max_tokens = min(config.MAX_TOKENS_CAP, max_tokens * 2)
                    continue
                return failure_message(f"Empty translation response from Groq (finish_reason={finish_reason}). Try a shorter clip or check API limits.")
            except Exception as exc:
                print(f"[LLM] Translation failed: {exc}")
                return failure_message(f"Translation failed: {str(exc)}")

    def _build_translation_messages(self, text: str) -> list[dict]:
        """Builds the system and user messages for the translation request."""
        prompt = (
            "Translate the following Vietnamese text to English. "
            "Output ONLY the English translation — no explanations, no markdown, no quotes.\n\n"
            f"{text}"
        )
        return [
            {"role": "system", "content": "You are a professional Vietnamese to English translator. Output ONLY the translated English text."},
            {"role": "user", "content": prompt},
        ]

    def _request_translation(self, messages: list[dict], max_tokens: int) -> Tuple[str, object]:
        """Sends the request to Groq and returns (content, finish_reason)."""
        message = self.client.chat.completions.create(
            messages=messages,
            model=self.model_name,
            temperature=0.0,
            max_completion_tokens=max_tokens,
            reasoning_effort="low",
        )
        choice = message.choices[0] if message.choices else None
        content = (choice.message.content or "").strip() if choice else ""
        finish_reason = getattr(choice, "finish_reason", None) if choice else None
        return content, finish_reason
class TextToSpeechNode:
    """Text-to-speech wrapper using Kokoro-TTS (local, no API required)."""

    # Kokoro-TTS native sample rate
    SAMPLE_RATE = config.TTS_SAMPLE_RATE

    def __init__(self):
        self.backend = self._select_backend()

    def _select_backend(self):
        if module_available("kokoro"):
            try:
                from kokoro import KPipeline
                print("[TTS] Using Kokoro-TTS local backend.")
                # lang_code 'a' = American English (the pipeline translates Vietnamese -> English)
                self._pipeline = KPipeline(
                    lang_code=config.KOKORO_LANG_CODE,
                    repo_id=config.KOKORO_REPO_ID,
                )
                return "kokoro"
            except Exception as exc:
                print(f"[TTS] Kokoro-TTS initialization failed: {exc}")
        print("[TTS] No TTS backend installed; generating placeholder audio.")
        return None

    def synthesize(self, text: str) -> Tuple[bytes, str]:
        """Returns (audio bytes, format) for the given English text."""
        if self.backend == "kokoro":
            return self._synthesize_with_kokoro(text)
        return self._placeholder_result()

    def _placeholder_result(self) -> Tuple[bytes, str]:
        """Returns the fallback beep whenever TTS cannot produce audio."""
        return generate_placeholder_audio(), "wav"

    def _synthesize_with_kokoro(self, text: str) -> Tuple[bytes, str]:
        try:
            # Use a natural English voice (af_heart is a warm female voice)
            generator = self._pipeline(text, voice=config.KOKORO_VOICE, speed=config.KOKORO_SPEED)
            audio_chunks = self._collect_audio_chunks(generator)
            if not audio_chunks:
                return self._placeholder_result()
            audio_np = self._combine_audio_chunks(audio_chunks)
            return encode_wav(audio_np, self.SAMPLE_RATE), "wav"
        except Exception as exc:
            print(f"[TTS] Kokoro-TTS synthesis failed: {exc}")
            return self._placeholder_result()

    def _collect_audio_chunks(self, generator) -> list:
        """Collects the non-null audio chunks produced by the TTS pipeline."""
        audio_chunks = []
        for result in generator:
            if result.audio is not None:
                audio_chunks.append(result.audio)
        return audio_chunks

    def _combine_audio_chunks(self, audio_chunks: list) -> np.ndarray:
        """Concatenates the audio chunks into a single numpy array."""
        import torch
        full_audio = torch.cat(audio_chunks) if len(audio_chunks) > 1 else audio_chunks[0]
        return full_audio.numpy()


class SpeechTranslationPipeline:
    """Orchestrates the audio, STT, translation and TTS nodes."""

    def __init__(self):
        self.audio_input_node = AudioInputNode()
        self.speech_to_text_node = SpeechToTextNode()
        self.translation_node = TranslationNode()
        self.text_to_speech_node = TextToSpeechNode()

    def run(self, audio_path: Optional[str]) -> Tuple[str, str, Optional[str], str]:
        """Runs the full pipeline and returns
        (transcript, translation, audio_output_path, status_message).
        """
        start_time = time.perf_counter()
        try:
            audio_bytes, sample_rate = self._run_timed_step("audio", lambda: self.audio_input_node.prepare_audio(audio_path))
            transcript = self._run_timed_step("stt", lambda: self.speech_to_text_node.transcribe(audio_bytes, sample_rate))
            translation = self._run_timed_step("llm", lambda: self.translation_node.translate(transcript))
            audio_out_bytes, audio_format = self._run_timed_step("tts", lambda: self.text_to_speech_node.synthesize(translation))
            output_path = self._save_audio_to_temp_file(audio_out_bytes, audio_format)

            elapsed = time.perf_counter() - start_time
            return transcript, translation, output_path, f"Completed in {elapsed:.2f}s"
        except Exception as exc:
            return "", f"[Pipeline error] {exc}", None, f"Failed: {exc}"

    def _save_audio_to_temp_file(self, audio_bytes: bytes, audio_format: str) -> str:
        """Writes the audio bytes to a temporary file and returns its path."""
        suffix = ".wav" if audio_format == "wav" else ".mp3"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as output_file:
            output_file.write(audio_bytes)
            return output_file.name

    def _run_timed_step(self, name: str, callback) -> object:
        """Runs a pipeline step while measuring and printing its duration."""
        start_time = time.perf_counter()
        result = callback()
        print(f"[{name.upper()}] completed in {time.perf_counter() - start_time:.2f}s")
        return result