from __future__ import annotations

import asyncio
import importlib.util
import io
import os
import re
import tempfile
import time
import socket
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
import numpy as np
import soundfile as sf
from dotenv import load_dotenv

load_dotenv()


class AudioInputNode:
    """Load mic audio from a file path or uploaded audio into memory."""

    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate

    def prepare_audio(self, audio_path: Optional[str]) -> Tuple[bytes, int]:
        if not audio_path:
            raise ValueError("No audio file was provided.")

        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        data, sr = sf.read(str(path), dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1)

        if sr != self.sample_rate:
            data = self._resample(data, sr, self.sample_rate)

        buffer = io.BytesIO()
        sf.write(buffer, data, self.sample_rate, format="WAV")
        return buffer.getvalue(), self.sample_rate

    def _resample(self, data: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
        if src_sr == dst_sr:
            return data
        target_len = int(len(data) * dst_sr / src_sr)
        return np.interp(
            np.linspace(0, len(data) - 1, target_len),
            np.arange(len(data)),
            data,
        ).astype(np.float32)


class STTNode:
    """Speech-to-text wrapper around faster-whisper. Loads the model once at startup."""

    def __init__(self, model_name: str = "large-v3-turbo") -> None:
        self.model_name = model_name
        self.model = None
        self.device = "cuda" if self._cuda_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        self._print_gpu_info()
        self._load_model()

    def _cuda_available(self) -> bool:
        if importlib.util.find_spec("torch") is None:
            return False
        import torch

        return torch.cuda.is_available()

    def _print_gpu_info(self) -> None:
        """Print GPU detection info at startup."""
        if self.device == "cuda":
            try:
                import torch

                print(f"[STT] 🚀 GPU DETECTED - Using CUDA acceleration")
                print(f"      Device: {torch.cuda.get_device_name(0)}")
                print(f"      Compute type: {self.compute_type}")
                print(f"      Expected STT latency: ~0.3-0.5s (FAST)")
            except Exception as e:
                print(f"[STT] Error getting GPU info: {e}")
        else:
            print(f"[STT] ⚠️  GPU NOT DETECTED - Using CPU (slower)")
            print(f"      Compute type: {self.compute_type}")
            print(f"      Expected STT latency: ~5-10s (SLOW)")
            print(f"      To enable GPU: Run 'python setup_gpu.py' for instructions")

    def _load_model(self) -> None:
        if importlib.util.find_spec("faster_whisper") is None:
            print("[STT] faster-whisper is not installed; transcription will be skipped.")
            return

        try:
            from faster_whisper import WhisperModel

            print(f"[STT] Loading model '{self.model_name}' (this may take a minute on first run)...")
            self.model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
                download_root=None,
            )
            print(f"[STT] ✅ Model loaded successfully on {self.device}.")
        except Exception as exc:  # pragma: no cover - defensive fallback
            print(f"[STT] ❌ Model load failed: {exc}")
            self.model = None

    def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000) -> str:
        if self.model is None:
            return "[STT unavailable: faster-whisper model could not be loaded.]"

        try:
            buffer = io.BytesIO(audio_bytes)
            audio_array, sr = sf.read(buffer, dtype="float32")
            if audio_array.ndim > 1:
                audio_array = np.mean(audio_array, axis=1)

            segments, _ = self.model.transcribe(
                audio_array,
                language="vi",
                beam_size=1,
                vad_filter=True,
                
            )
            transcript = " ".join(segment.text.strip() for segment in segments if segment.text)
            return transcript.strip() or "[STT returned no speech.]"
        except Exception as exc:  # pragma: no cover - defensive fallback
            return f"[STT error: {exc}]"


class LLMNode:
    """Translate Vietnamese text into English with Groq API (openai/gpt-oss-20b)."""

    def __init__(self, model_name: str = "openai/gpt-oss-20b") -> None:
        self.model_name = model_name
        self.client = None
        self.api_key_missing = False
        self._load_model()



    def _load_model(self) -> None:
        if importlib.util.find_spec("groq") is None:
            print("[LLM] ❌ groq package is not installed.")
            print("      Run: pip install groq")
            return

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("[LLM] ⚠️  GROQ_API_KEY not set!")
            print("      Get free API key: https://console.groq.com/keys")
            print("      Then set: $env:GROQ_API_KEY='your-key-here'")
            self.api_key_missing = True
            return

        try:
            from groq import Groq  # type: ignore

            self.client = Groq(api_key=api_key)
            print(f"[LLM] ✅ Loaded Groq model '{self.model_name}' (⚡ Ultra-fast inference).")
        except Exception as exc:  # pragma: no cover - defensive fallback
            print(f"[LLM] ❌ Groq initialization failed: {exc}")
            self.client = None

    def translate(self, text: str) -> str:
        if self.api_key_missing:
            return f"[ERROR] GROQ_API_KEY not set. Cannot translate. See setup instructions."
        
        if self.client is None:
            return f"[ERROR] LLM client not loaded. Check console output above."

        cleaned = (text or "").strip()
        if not cleaned:
            return "[ERROR] No text to translate."

        # Long transcripts: translate in chunks so output never hits the token ceiling.
        chunks = self._split_for_translation(cleaned)
        if len(chunks) == 1:
            return self._translate_chunk(chunks[0])

        parts: list[str] = []
        for i, chunk in enumerate(chunks, start=1):
            print(f"[LLM] Translating chunk {i}/{len(chunks)} ({len(chunk)} chars)...")
            part = self._translate_chunk(chunk)
            if part.startswith("[ERROR]"):
                return part
            parts.append(part)
        return " ".join(parts).strip()

    def _split_for_translation(self, text: str, max_chars: int = 1200) -> list[str]:
        """Split long Vietnamese text into sentence-ish chunks for stable translation."""
        if len(text) <= max_chars:
            return [text]

        # Prefer sentence boundaries; fall back to commas/newlines (common in VI transcripts).
        sentences = re.split(r"(?<=[.!?…])\s+|(?<=[;])\s+|\n+", text)
        if len(sentences) == 1 and len(text) > max_chars:
            sentences = re.split(r"(?<=[,])\s+", text)
        chunks: list[str] = []
        current = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if not current:
                current = sentence
            elif len(current) + 1 + len(sentence) <= max_chars:
                current = f"{current} {sentence}"
            else:
                chunks.append(current)
                current = sentence
        if current:
            chunks.append(current)

        # Hard-split any oversized leftover piece.
        final: list[str] = []
        for chunk in chunks:
            if len(chunk) <= max_chars:
                final.append(chunk)
                continue
            words = chunk.split()
            buf: list[str] = []
            size = 0
            for word in words:
                add = len(word) + (1 if buf else 0)
                if buf and size + add > max_chars:
                    final.append(" ".join(buf))
                    buf = [word]
                    size = len(word)
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

        # gpt-oss-20b is a reasoning model: reasoning tokens share the same completion budget.
        # Long text + low max_tokens => empty content with finish_reason=length.
        approx_out = max(256, min(4096, int(len(text.split()) * 3) + 256))
        max_completion_tokens = approx_out + 1024  # headroom for reasoning

        try:
            message = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a professional Vietnamese to English translator. "
                            "Output ONLY the translated English text."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                model=self.model_name,
                temperature=0.0,
                max_completion_tokens=max_completion_tokens,
                reasoning_effort="low",
            )
            choice = message.choices[0] if message.choices else None
            content = (choice.message.content or "").strip() if choice else ""
            finish_reason = getattr(choice, "finish_reason", None) if choice else None

            if not content:
                # Retry once with a larger budget if reasoning ate the first budget.
                if finish_reason == "length":
                    print("[LLM] Empty content (reasoning used budget). Retrying with higher limit...")
                    message = self.client.chat.completions.create(
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a professional Vietnamese to English translator. "
                                    "Output ONLY the translated English text."
                                ),
                            },
                            {"role": "user", "content": prompt},
                        ],
                        model=self.model_name,
                        temperature=0.0,
                        max_completion_tokens=min(8192, max_completion_tokens * 2),
                        reasoning_effort="low",
                    )
                    choice = message.choices[0] if message.choices else None
                    content = (choice.message.content or "").strip() if choice else ""

            if not content:
                return (
                    "[ERROR] Empty translation response from Groq "
                    f"(finish_reason={finish_reason}). Try a shorter clip or check API limits."
                )
            return content
        except Exception as exc:  # pragma: no cover - defensive fallback
            print(f"[LLM] Translation failed: {exc}")
            return f"[ERROR] Translation failed: {str(exc)}"


class VienewTTSWrapper:
    """Modular wrapper for a future Vienew SDK, with an edge-tts fallback for local testing."""

    def __init__(self) -> None:
        self.backend = self._select_backend()

    def _select_backend(self):
        if importlib.util.find_spec("vienew") is not None:
            try:
                from vienew import TTSClient  # type: ignore

                print("[TTS] Using Vienew backend.")
                return TTSClient()
            except Exception as exc:  # pragma: no cover - defensive fallback
                print(f"[TTS] Vienew backend unavailable: {exc}")

        if importlib.util.find_spec("edge_tts") is not None:
            print("[TTS] Using edge-tts fallback backend.")
            return "edge_tts"

        print("[TTS] No TTS backend is installed; generating a placeholder audio file.")
        return None

    def synthesize(self, text: str) -> Tuple[bytes, str]:
        if self.backend is None:
            return self._generate_placeholder_audio(), "wav"

        if self.backend == "edge_tts":
            return self._synthesize_with_edge_tts(text)

        try:
            audio_bytes = self.backend.speak(text)
            if isinstance(audio_bytes, (bytes, bytearray)):
                return bytes(audio_bytes), "wav"
        except Exception as exc:  # pragma: no cover - defensive fallback
            print(f"[TTS] Vienew synthesis failed: {exc}")

        return self._synthesize_with_edge_tts(text)

    def _synthesize_with_edge_tts(self, text: str) -> Tuple[bytes, str]:
        try:
            import edge_tts

            async def _run() -> bytes:
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                    output_path = temp_file.name
                communicate = edge_tts.Communicate(text, voice="en-US-AriaNeural")
                await communicate.save(output_path)
                return Path(output_path).read_bytes()

            audio_bytes = asyncio.run(_run())
            return audio_bytes, "mp3"
        except Exception as exc:  # pragma: no cover - defensive fallback
            print(f"[TTS] edge-tts failed: {exc}")
            return self._generate_placeholder_audio(), "wav"

    def _generate_placeholder_audio(self) -> bytes:
        sr = 22050
        length = int(sr * 0.8)
        t = np.linspace(0, 0.8, length, endpoint=False)
        audio = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        buffer = io.BytesIO()
        sf.write(buffer, audio, sr, format="WAV")
        return buffer.getvalue()


class SpeechTranslationPipeline:
    """Coordinates the audio -> STT -> LLM -> TTS flow."""

    def __init__(self) -> None:
        self.audio_node = AudioInputNode()
        self.stt_node = STTNode()
        self.llm_node = LLMNode()
        self.tts_node = VienewTTSWrapper()

    def run(self, audio_path: Optional[str]) -> Tuple[str, str, Optional[str], str]:
        start_time = time.perf_counter()
        print("[PIPELINE] Starting...")

        try:
            audio_bytes, sample_rate = self._step("audio", lambda: self.audio_node.prepare_audio(audio_path))
            transcript = self._step("stt", lambda: self.stt_node.transcribe(audio_bytes, sample_rate))
            translation = self._step("llm", lambda: self.llm_node.translate(transcript))
            audio_out_bytes, audio_format = self._step("tts", lambda: self.tts_node.synthesize(translation))

            output_path = self._write_audio_to_tempfile(audio_out_bytes, audio_format)
            elapsed = time.perf_counter() - start_time
            print(f"[PIPELINE] Completed in {elapsed:.2f}s")
            return transcript, translation, output_path, f"Completed in {elapsed:.2f}s"
        except Exception as exc:  # pragma: no cover - defensive fallback
            elapsed = time.perf_counter() - start_time
            print(f"[PIPELINE] Failed after {elapsed:.2f}s: {exc}")
            return "", f"[Pipeline error] {exc}", None, f"Failed: {exc}"



    def _step(self, step_name: str, action) -> object:
        start = time.perf_counter()
        result = action()
        elapsed = time.perf_counter() - start
        print(f"[{step_name.upper()}] completed in {elapsed:.2f}s")
        return result

    def _write_audio_to_tempfile(self, audio_bytes: bytes, audio_format: str) -> str:
        suffix = ".wav" if audio_format == "wav" else ".mp3"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_file.write(audio_bytes)
            return temp_file.name


def build_ui() -> gr.Blocks:
    pipeline = SpeechTranslationPipeline()

    def translate_audio(audio_path: Optional[str]):
        if not audio_path:
            return "", "", None, "⚠️ Vui lòng ghi âm hoặc tải tệp âm thanh trước!"

        transcript, translation, audio_output_path, status = pipeline.run(audio_path)
        return transcript, translation, audio_output_path, status

    custom_css = """
    /* ===== Global ===== */
    :root {
        --gradient-primary: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        --glass-bg: rgba(255, 255, 255, 0.08);
        --glass-border: rgba(255, 255, 255, 0.15);
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --accent: #8b5cf6;
    }

    .gradio-container {
        background: radial-gradient(ellipse at top, #1e1b4b 0%, #0f172a 45%, #020617 100%) !important;
        font-family: 'Inter', system-ui, sans-serif !important;
    }

    /* ===== Hero Header ===== */
    .hero-section {
        text-align: center;
        padding: 40px 20px 10px;
        position: relative;
    }

    .hero-title {
        font-size: 3em;
        font-weight: 800;
        background: linear-gradient(135deg, #a78bfa 0%, #60a5fa 50%, #22d3ee 100%);
        background-size: 200% 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 0 8px;
        letter-spacing: -1px;
        animation: gradient-shift 6s ease infinite;
    }

    @keyframes gradient-shift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }

    .hero-subtitle {
        color: var(--text-secondary);
        font-size: 1.15em;
        margin: 0 0 20px;
    }

    .hero-subtitle .flag {
        font-size: 1.3em;
    }

    /* ===== Glass Cards ===== */
    .glass-card {
        background: var(--glass-bg);
        border: 1px solid var(--glass-border);
        border-radius: 16px;
        padding: 16px 20px;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
        margin-bottom: 12px;
    }

    .glass-card:hover {
        border-color: rgba(139, 92, 246, 0.4);
        box-shadow: 0 12px 40px rgba(139, 92, 246, 0.12);
    }

    .card-title {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 1.05em;
        font-weight: 700;
        color: var(--text-primary);
    }

    .card-title .card-icon {
        width: 36px;
        height: 36px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 10px;
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.3), rgba(6, 182, 212, 0.3));
        font-size: 1.1em;
    }

    /* ===== Buttons ===== */
    .translate-btn {
        background: var(--gradient-primary) !important;
        border: none !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 1.1em !important;
        padding: 14px 32px !important;
        border-radius: 14px !important;
        box-shadow: 0 4px 20px rgba(139, 92, 246, 0.35) !important;
        transition: all 0.3s ease !important;
        letter-spacing: 0.3px;
    }

    .translate-btn:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 32px rgba(139, 92, 246, 0.5) !important;
        filter: brightness(1.1);
    }

    .translate-btn:active {
        transform: translateY(0) !important;
    }

    /* ===== Textboxes ===== */
    .textbox-output textarea {
        background: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        font-size: 1em !important;
        line-height: 1.6 !important;
    }

    .textbox-output textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.15) !important;
    }

    /* ===== Audio ===== */
    .audio-input, .audio-output {
        border-radius: 12px !important;
        overflow: hidden;
    }

    /* ===== Scrollbar ===== */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: transparent;
    }

    ::-webkit-scrollbar-thumb {
        background: rgba(139, 92, 246, 0.4);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: rgba(139, 92, 246, 0.6);
    }

    /* ===== Footer ===== */
    .footer {
        text-align: center;
        color: var(--text-secondary);
        font-size: 0.85em;
        padding: 20px 0 10px;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
        margin-top: 30px;
    }

    .footer .heart {
        color: #f472b6;
    }
    """

    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.purple,
        secondary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
    )

    with gr.Blocks(title="Dịch Giọng Nói Việt - Anh", css=custom_css, theme=theme) as demo:
        # ===== Hero Section =====
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

        # ===== Main Translation =====
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("""
                <div class='glass-card'>
                    <div class='card-title'>
                        <span class='card-icon'>🎙️</span>
                        Input: Tiếng Việt
                    </div>
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
                    <div class='card-title'>
                        <span class='card-icon'>🔊</span>
                        Output: Tiếng Anh
                    </div>
                </div>
                """)
                audio_output = gr.Audio(
                    label="Phát âm tiếng Anh",
                    type="filepath",
                    interactive=False,
                    elem_classes=["audio-output"],
                )

        with gr.Row():
            translate_button = gr.Button(
                "🚀 Dịch & Phát Âm",
                size="lg",
                variant="primary",
                elem_classes=["translate-btn"],
            )

        status_box = gr.Textbox(
            label="📊 Trạng Thái",
            lines=2,
            interactive=False,
            placeholder="Nhấn nút để bắt đầu dịch...",
            elem_classes=["textbox-output"],
        )

        with gr.Row():
            with gr.Column():
                gr.HTML("""
                <div class='glass-card'>
                    <div class='card-title'>
                        <span class='card-icon'>📝</span>
                        Văn Bản Gốc (Tiếng Việt)
                    </div>
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
                    <div class='card-title'>
                        <span class='card-icon'>🌍</span>
                        Văn Bản Dịch (Tiếng Anh)
                    </div>
                </div>
                """)
                translated_text = gr.Textbox(
                    label="Translation",
                    lines=5,
                    interactive=False,
                    placeholder="Văn bản dịch sẽ hiện ở đây...",
                    elem_classes=["textbox-output"],
                )

        translate_button.click(
            fn=translate_audio,
            inputs=[audio_input],
            outputs=[original_text, translated_text, audio_output, status_box],
        )

        # ===== Footer =====
        gr.HTML("""
        <div class='footer'>
            Made with <span class='heart'>❤</span> · Dịch Giọng Nói Việt - Anh
        </div>
        """)

    return demo


if __name__ == "__main__":
    print("=" * 70)
    print("SPEECH TRANSLATION PIPELINE - Initializing")
    print("=" * 70)
    print()
    
    demo = build_ui()
    
    print()
    print("=" * 70)
    print("✅ Pipeline ready!")
    print("=" * 70)
    print()

    # Determine server port: prefer environment variable, then try 7860,
    # otherwise pick an ephemeral free port.
    env_port = os.getenv("GRADIO_SERVER_PORT")
    def _parse_env_port(val: str) -> int | None:
        try:
            return int(val)
        except Exception:
            return None

    server_port = None
    if env_port:
        server_port = _parse_env_port(env_port)

    if server_port is None:
        preferred = 7860
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", preferred))
                server_port = preferred
        except OSError:
            # preferred port taken, get ephemeral port
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", 0))
                server_port = s.getsockname()[1]

    print(f"Open your browser at: http://127.0.0.1:{server_port}")

    demo.launch(server_name="127.0.0.1", server_port=server_port, share=False)