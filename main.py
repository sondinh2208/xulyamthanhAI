from __future__ import annotations

import importlib.util
import io
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
import numpy as np
import soundfile as sf
from dotenv import load_dotenv

load_dotenv()


def _read_audio(source, sample_rate: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """Đọc audio thành mono float32, tùy chọn resample về sample_rate."""
    data, sr = sf.read(source, dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if sample_rate and sr != sample_rate:
        target_len = int(len(data) * sample_rate / sr)
        data = np.interp(np.linspace(0, len(data) - 1, target_len), np.arange(len(data)), data).astype(np.float32)
    return data, sr


class AudioInputNode:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def prepare_audio(self, audio_path: Optional[str]) -> Tuple[bytes, int]:
        if not audio_path:
            raise ValueError("No audio file was provided.")
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        data, _ = _read_audio(str(path), self.sample_rate)
        buffer = io.BytesIO()
        sf.write(buffer, data, self.sample_rate, format="WAV")
        return buffer.getvalue(), self.sample_rate


class STTNode:
    def __init__(self, model_name: str = "large-v3-turbo"):
        self.model_name = model_name
        self.device = "cuda" if self._cuda_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        print(f"[STT] Using {self.device.upper()} ({self.compute_type})")
        self.model = None
        self._load_model()

    def _cuda_available(self) -> bool:
        if importlib.util.find_spec("torch") is None:
            return False
        import torch
        return torch.cuda.is_available()

    def _load_model(self):
        if importlib.util.find_spec("faster_whisper") is None:
            print("[STT] faster-whisper is not installed; transcription will be skipped.")
            return
        try:
            from faster_whisper import WhisperModel
            print(f"[STT] Loading model '{self.model_name}'...")
            self.model = WhisperModel(self.model_name, device=self.device, compute_type=self.compute_type, download_root=None)
        except Exception as exc:
            print(f"[STT] Model load failed: {exc}")
            self.model = None

    def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000) -> str:
        if self.model is None:
            return "[STT unavailable: faster-whisper model could not be loaded.]"
        try:
            audio_array, _ = _read_audio(io.BytesIO(audio_bytes))
            segments, _ = self.model.transcribe(audio_array, language="vi", beam_size=1, vad_filter=True)
            transcript = " ".join(segment.text.strip() for segment in segments if segment.text)
            return transcript.strip() or "[STT returned no speech.]"
        except Exception as exc:
            return f"[STT error: {exc}]"


class LLMNode:
    def __init__(self, model_name: str = "openai/gpt-oss-20b"):
        self.model_name = model_name
        self.client = None
        self.api_key_missing = False
        self._load_model()

    def _load_model(self):
        if importlib.util.find_spec("groq") is None:
            print("[LLM] groq package is not installed.")
            return
        api_key = os.getenv("GROQ_API_KEY")
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
        if self.api_key_missing:
            return "[ERROR] GROQ_API_KEY not set. Cannot translate. See setup instructions."
        if self.client is None:
            return "[ERROR] LLM client not loaded. Check console output above."
        cleaned = (text or "").strip()
        if not cleaned:
            return "[ERROR] No text to translate."

        chunks = self._split_for_translation(cleaned)
        parts = [self._translate_chunk(chunk) for chunk in chunks]
        for part in parts:
            if part.startswith("[ERROR]"):
                return part
        return " ".join(parts).strip()

    def _split_for_translation(self, text: str, max_chars: int = 1200) -> list[str]:
        if len(text) <= max_chars:
            return [text]
        sentences = re.split(r"(?<=[.!?…;])\s+|\n+", text)
        if len(sentences) == 1:
            sentences = re.split(r"(?<=[,])\s+", text)

        chunks, current = [], ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if not current or len(current) + 1 + len(sentence) <= max_chars:
                current = f"{current} {sentence}".strip()
            else:
                chunks.append(current)
                current = sentence
        if current:
            chunks.append(current)

        final = []
        for chunk in chunks:
            if len(chunk) <= max_chars:
                final.append(chunk)
                continue
            buf, size = [], 0
            for word in chunk.split():
                add = len(word) + (1 if buf else 0)
                if buf and size + add > max_chars:
                    final.append(" ".join(buf))
                    buf, size = [word], len(word)
                else:
                    buf.append(word)
                    size += add
            if buf:
                final.append(" ".join(buf))
        return final or [text]

    def _translate_chunk(self, text: str) -> str:
        prompt = (
            "Translate the following Vietnamese text to English. "
            "Output ONLY the English translation — no explanations, no markdown, no quotes.\n\n"
            f"{text}"
        )
        messages = [
            {"role": "system", "content": "You are a professional Vietnamese to English translator. Output ONLY the translated English text."},
            {"role": "user", "content": prompt},
        ]
        max_tokens = min(8192, int(len(text.split()) * 3) + 1280)

        for attempt in range(2):
            try:
                message = self.client.chat.completions.create(
                    messages=messages,
                    model=self.model_name,
                    temperature=0.0,
                    max_completion_tokens=max_tokens,
                    reasoning_effort="low",
                )
                choice = message.choices[0] if message.choices else None
                content = (choice.message.content or "").strip() if choice else ""
                if content:
                    return content
                finish_reason = getattr(choice, "finish_reason", None) if choice else None
                if finish_reason == "length" and attempt == 0:
                    max_tokens = min(8192, max_tokens * 2)
                    continue
                return f"[ERROR] Empty translation response from Groq (finish_reason={finish_reason}). Try a shorter clip or check API limits."
            except Exception as exc:
                print(f"[LLM] Translation failed: {exc}")
                return f"[ERROR] Translation failed: {str(exc)}"


class VienewTTSWrapper:
    """Text-to-Speech wrapper using Kokoro-TTS (local, no API required)."""

    # Kokoro-TTS sample rate
    SAMPLE_RATE = 24000

    def __init__(self):
        self.backend = self._select_backend()

    def _select_backend(self):
        if importlib.util.find_spec("kokoro") is not None:
            try:
                from kokoro import KPipeline
                print("[TTS] Using Kokoro-TTS local backend.")
                # 'a' = American English (the pipeline translates Vietnamese -> English)
                self._pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
                return "kokoro"
            except Exception as exc:
                print(f"[TTS] Kokoro-TTS initialization failed: {exc}")
        print("[TTS] No TTS backend installed; generating placeholder audio.")
        return None

    def synthesize(self, text: str) -> Tuple[bytes, str]:
        if self.backend is None:
            return self._generate_placeholder_audio(), "wav"
        if self.backend == "kokoro":
            return self._synthesize_with_kokoro(text)
        return self._generate_placeholder_audio(), "wav"

    def _synthesize_with_kokoro(self, text: str) -> Tuple[bytes, str]:
        try:
            import torch

            # Use a natural English voice (af_heart is a warm female voice)
            voice = "af_heart"
            generator = self._pipeline(text, voice=voice, speed=1.0)

            audio_chunks = []
            for result in generator:
                if result.audio is not None:
                    audio_chunks.append(result.audio)

            if not audio_chunks:
                return self._generate_placeholder_audio(), "wav"

            # Concatenate all audio chunks
            full_audio = torch.cat(audio_chunks) if len(audio_chunks) > 1 else audio_chunks[0]
            audio_np = full_audio.numpy()

            # Write as WAV
            buffer = io.BytesIO()
            sf.write(buffer, audio_np, self.SAMPLE_RATE, format="WAV")
            return buffer.getvalue(), "wav"
        except Exception as exc:
            print(f"[TTS] Kokoro-TTS synthesis failed: {exc}")
            return self._generate_placeholder_audio(), "wav"

    def _generate_placeholder_audio(self) -> bytes:
        sr = 22050
        audio = 0.3 * np.sin(2 * np.pi * 440 * np.linspace(0, 0.8, int(sr * 0.8), endpoint=False)).astype(np.float32)
        buffer = io.BytesIO()
        sf.write(buffer, audio, sr, format="WAV")
        return buffer.getvalue()


class SpeechTranslationPipeline:
    def __init__(self):
        self.audio_node = AudioInputNode()
        self.stt_node = STTNode()
        self.llm_node = LLMNode()
        self.tts_node = VienewTTSWrapper()

    def run(self, audio_path: Optional[str]) -> Tuple[str, str, Optional[str], str]:
        start_time = time.perf_counter()
        try:
            audio_bytes, sample_rate = self._timed("audio", lambda: self.audio_node.prepare_audio(audio_path))
            transcript = self._timed("stt", lambda: self.stt_node.transcribe(audio_bytes, sample_rate))
            translation = self._timed("llm", lambda: self.llm_node.translate(transcript))
            audio_out_bytes, audio_format = self._timed("tts", lambda: self.tts_node.synthesize(translation))

            suffix = ".wav" if audio_format == "wav" else ".mp3"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                f.write(audio_out_bytes)
                output_path = f.name

            elapsed = time.perf_counter() - start_time
            return transcript, translation, output_path, f"Completed in {elapsed:.2f}s"
        except Exception as exc:
            return "", f"[Pipeline error] {exc}", None, f"Failed: {exc}"

    def _timed(self, name: str, fn) -> object:
        start = time.perf_counter()
        result = fn()
        print(f"[{name.upper()}] completed in {time.perf_counter() - start:.2f}s")
        return result


def build_ui() -> gr.Blocks:
    pipeline = SpeechTranslationPipeline()

    def translate_audio(audio_path: Optional[str]):
        if not audio_path:
            return "", "", None, "⚠️ Vui lòng ghi âm hoặc tải tệp âm thanh trước!"
        return pipeline.run(audio_path)

    custom_css = """
    :root { --gradient-primary: linear-gradient(135deg,#667eea 0%,#764ba2 50%,#f093fb 100%); --glass-bg: rgba(255,255,255,.08); --glass-border: rgba(255,255,255,.15); --text-primary:#f8fafc; --text-secondary:#94a3b8; --accent:#8b5cf6; }
    .gradio-container { background: radial-gradient(ellipse at top,#1e1b4b 0%,#0f172a 45%,#020617 100%) !important; font-family:'Inter',system-ui,sans-serif !important; }
    .hero-section { text-align:center; padding:40px 20px 10px; position:relative; }
    .hero-title { font-size:3em; font-weight:800; background:linear-gradient(135deg,#a78bfa 0%,#60a5fa 50%,#22d3ee 100%); background-size:200% 200%; -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; margin:0 0 8px; letter-spacing:-1px; animation:gradient-shift 6s ease infinite; }
    @keyframes gradient-shift { 0%,100% { background-position:0% 50%; } 50% { background-position:100% 50%; } }
    .hero-subtitle { color:var(--text-secondary); font-size:1.15em; margin:0 0 20px; }
    .hero-subtitle .flag { font-size:1.3em; }
    .glass-card { background:var(--glass-bg); border:1px solid var(--glass-border); border-radius:16px; padding:16px 20px; backdrop-filter:blur(12px); box-shadow:0 8px 32px rgba(0,0,0,.2); transition:all .3s ease; margin-bottom:12px; }
    .glass-card:hover { border-color:rgba(139,92,246,.4); box-shadow:0 12px 40px rgba(139,92,246,.12); }
    .card-title { display:flex; align-items:center; gap:10px; font-size:1.05em; font-weight:700; color:var(--text-primary); }
    .card-title .card-icon { width:36px; height:36px; display:flex; align-items:center; justify-content:center; border-radius:10px; background:linear-gradient(135deg,rgba(139,92,246,.3),rgba(6,182,212,.3)); font-size:1.1em; }
    .textbox-output textarea { background:rgba(15,23,42,.6) !important; border:1px solid var(--glass-border) !important; border-radius:12px !important; color:var(--text-primary) !important; font-size:1em !important; line-height:1.6 !important; }
    .textbox-output textarea:focus { border-color:var(--accent) !important; box-shadow:0 0 0 3px rgba(139,92,246,.15) !important; }
    .audio-input, .audio-output { border-radius:12px !important; overflow:hidden; }
    ::-webkit-scrollbar { width:8px; height:8px; }
    ::-webkit-scrollbar-track { background:transparent; }
    ::-webkit-scrollbar-thumb { background:rgba(139,92,246,.4); border-radius:4px; }
    ::-webkit-scrollbar-thumb:hover { background:rgba(139,92,246,.6); }
    .footer { text-align:center; color:var(--text-secondary); font-size:.85em; padding:20px 0 10px; border-top:1px solid rgba(255,255,255,.06); margin-top:30px; }
    .footer .heart { color:#f472b6; }
    """

    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.purple,
        secondary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
    )

    with gr.Blocks(title="Dịch Giọng Nói Việt - Anh", css=custom_css, theme=theme) as demo:
        gr.HTML("""
        <div class='hero-section'>
            <h1 class='hero-title'>Dịch Giọng Nói</h1>
            <p class='hero-subtitle'>
                <span class='flag'>🇻🇳</span> Tiếng Việt
                <span style='color:#8b5cf6; font-weight:700;'>→</span>
                <span class='flag'>🇬🇧</span> English
            </p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("""
                <div class='glass-card'>
                    <div class='card-title'><span class='card-icon'>🎙️</span> Input: Tiếng Việt</div>
                </div>
                """)
                audio_input = gr.Audio(
                    label="Ghi âm hoặc tải tệp",
                    sources=["microphone", "upload"],
                    type="filepath",
                    streaming=False,
                    elem_classes=["audio-input"],
                )

            with gr.Column(scale=1):
                gr.HTML("""
                <div class='glass-card'>
                    <div class='card-title'><span class='card-icon'>🔊</span> Output: Tiếng Anh</div>
                </div>
                """)
                audio_output = gr.Audio(
                    label="Phát âm tiếng Anh",
                    type="filepath",
                    interactive=False,
                    autoplay=True,
                    elem_classes=["audio-output"],
                )

        status_box = gr.Textbox(
            label="📊 Trạng Thái",
            lines=2,
            interactive=False,
            placeholder="🎧 Chờ ghi âm hoặc tải tệp âm thanh — quá trình dịch sẽ chạy tự động...",
            elem_classes=["textbox-output"],
        )

        with gr.Row():
            with gr.Column():
                gr.HTML("""
                <div class='glass-card'>
                    <div class='card-title'><span class='card-icon'>📝</span> Văn Bản Gốc (Tiếng Việt)</div>
                </div>
                """)
                original_text = gr.Textbox(
                    label="Transcript",
                    lines=5,
                    interactive=False,
                    placeholder="Văn bản gốc sẽ hiện ở đây...",
                    elem_classes=["textbox-output"],
                )

            with gr.Column():
                gr.HTML("""
                <div class='glass-card'>
                    <div class='card-title'><span class='card-icon'>🌍</span> Văn Bản Dịch (Tiếng Anh)</div>
                </div>
                """)
                translated_text = gr.Textbox(
                    label="Translation",
                    lines=5,
                    interactive=False,
                    placeholder="Văn bản dịch sẽ hiện ở đây...",
                    elem_classes=["textbox-output"],
                )

        # Tự động kích hoạt pipeline ngay khi (1) dừng ghi âm micro hoặc (2) tải tệp âm thanh lên
        audio_input.stop_recording(
            fn=translate_audio,
            inputs=[audio_input],
            outputs=[original_text, translated_text, audio_output, status_box],
        )
        audio_input.upload(
            fn=translate_audio,
            inputs=[audio_input],
            outputs=[original_text, translated_text, audio_output, status_box],
        )

        gr.HTML("""
        <div class='footer'>
            Made with <span class='heart'>❤</span> · Dịch Giọng Nói Việt - Anh
        </div>
        """)

    return demo


if __name__ == "__main__":
    demo = build_ui()
    port = int(os.environ.get("PORT", os.getenv("GRADIO_SERVER_PORT", "7860")))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=True)
