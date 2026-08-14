from __future__ import annotations

import asyncio
import importlib.util
import io
import os
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
    """Translate Vietnamese text into English with Groq API (llama-3.3-70b-versatile)."""

    def __init__(self, model_name: str = "llama-3.3-70b-versatile") -> None:
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

        prompt = (
            "You are a professional translator. Translate the following Vietnamese text to English. "
            "Output ONLY the English text, no explanations, no markdown.\n\n"
            f"{text}"
        )

        try:
            message = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a professional Vietnamese to English translator. Output ONLY the translated text."},
                    {"role": "user", "content": prompt}
                ],
                model=self.model_name,
                temperature=0.0,
                max_tokens=300,
            )
            result = message.choices[0].message.content.strip() if message.choices else ""
            if not result:
                return f"[ERROR] Empty translation response from Groq."
            return result
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

    def run(self, audio_path: Optional[str]) -> Tuple[str, str, str, str]:
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
            return "", f"[Pipeline error] {exc}", "", f"Failed: {exc}"

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
            return "", "", None, "Please record audio first."

        transcript, translation, audio_output_path, status = pipeline.run(audio_path)
        return transcript, translation, audio_output_path, status

    custom_css = """
    .header-title {
        text-align: center;
        font-size: 2.5em;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 20px 0;
        margin-bottom: 10px;
    }
    
    .header-subtitle {
        text-align: center;
        font-size: 1.1em;
        color: #666;
        margin-bottom: 30px;
    }
    
    .info-box {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border-left: 4px solid #667eea;
        padding: 15px;
        border-radius: 8px;
        margin: 15px 0;
    }
    
    .status-success {
        color: #22c55e;
        font-weight: 600;
    }
    
    .status-error {
        color: #ef4444;
        font-weight: 600;
    }
    
    .card-section {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
    }
    """

    with gr.Blocks(title="Vietnamese Speech -> English Speech Translator", css=custom_css, theme=gr.themes.Soft()) as demo:
        # Header
        gr.HTML("<div class='header-title'>🌐 Speech Translation Pipeline</div>")
        gr.HTML("<div class='header-subtitle'>Vietnamese 🇻🇳 → English 🇬🇧 in Real-time</div>")
        
        # Info Box
        gr.HTML("""
        <div class='info-box'>
            <b>📌 Hướng dẫn:</b> Nhấn nút ghi âm, nói tiếng Việt, sau đó nhấn "Dịch & Phát âm". 
            Hệ thống sẽ tự động dịch sang tiếng Anh và phát âm cho bạn.
        </div>
        """)

        with gr.Tabs():
            # Tab 1: Main Translation
            with gr.TabItem("🎤 Dịch Nhanh", id="tab_translate"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎙️ Input: Tiếng Việt")
                        audio_input = gr.Audio(
                            label="Ghi âm hoặc tải tệp",
                            sources=["microphone", "upload"],
                            type="filepath",
                            streaming=False,
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 🔊 Output: Tiếng Anh")
                        audio_output = gr.Audio(
                            label="Phát âm tiếng Anh",
                            type="filepath",
                            interactive=False
                        )

                with gr.Row():
                    translate_button = gr.Button("🚀 Dịch & Phát Âm", size="lg", variant="primary")

                # Status
                status_box = gr.Textbox(
                    label="📊 Trạng Thái",
                    lines=2,
                    interactive=False,
                    placeholder="Nhấn nút để bắt đầu dịch..."
                )

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 📝 Văn Bản Gốc (Tiếng Việt)")
                        original_text = gr.Textbox(
                            label="Transcript",
                            lines=5,
                            interactive=False,
                            placeholder="Văn bản gốc sẽ hiện ở đây..."
                        )
                    
                    with gr.Column():
                        gr.Markdown("### 📝 Văn Bản Dịch (Tiếng Anh)")
                        translated_text = gr.Textbox(
                            label="Translation",
                            lines=5,
                            interactive=False,
                            placeholder="Văn bản dịch sẽ hiện ở đây..."
                        )

                translate_button.click(
                    fn=translate_audio,
                    inputs=[audio_input],
                    outputs=[original_text, translated_text, audio_output, status_box],
                )

            # Tab 2: Information
            with gr.TabItem("ℹ️ Thông Tin", id="tab_info"):
                gr.Markdown("""
                ## 🚀 Về Ứng Dụng
                
                **Vietnamese Speech-to-Speech Translator** là một ứng dụng dịch giọng nói real-time với:
                
                ### ✨ Tính Năng
                - 🎤 **Ghi âm**: Nói tiếng Việt trực tiếp
                - 🧠 **AI Dịch Thuật**: Sử dụng API Groq
                - 🔊 **Phát Âm**: Nghe tiếng Anh được phát âm tự động
                - ⚡ **Nhanh**: Dưới 5 giây toàn bộ quy trình
                - 🖥️ **GPU Accelerated**: Sử dụng NVIDIA RTX 4050
                
                ### 🔧 Công Nghệ
                - **STT**: faster-whisper (Model: large-v3-turbo)
                - **LLM**: Groq LLM (Model: llama-3.3-70b-versatile)
                - **TTS**: edge-tts (fallback) / Vienew (tùy chọn)
                - **GUI**: Gradio
                
                ### 📊 Hiệu Suất
                - STT (Whisper): ~0.3-0.5s ⚡
                - Translation (Groq): ~1-2s
                - TTS (Text-to-Speech): ~1-2s
                - **Tổng cộng**: ~3-5s
                
                ### 📁 Cấu Trúc
                ```
                main.py              - Ứng dụng chính
                setup_gpu.py          - Kiểm tra GPU
                requirements.txt      - Dependencies
                .env                  - API Key (bảo mật)
                ```
                """)

            # Tab 3: Settings
            with gr.TabItem("⚙️ Cài Đặt", id="tab_settings"):
                gr.Markdown("""
                ## ⚙️ Cấu Hình
                
                ### 🔑 API Key
                - **Status**: ✅ Đã cấu hình
                - **Model**: llama-3.3-70b-versatile
                - **Tệp**: `.env` (giữ bảo mật)
                
                ### 🖥️ GPU
                - **Device**: NVIDIA RTX 4050
                - **CUDA**: 12.7
                - **PyTorch**: 2.5.1
                - **Status**: ✅ Hoạt động
                
                ### 🎯 Chất Lượng
                - **Ngôn Ngữ STT**: Tiếng Việt
                - **Beam Size**: 1
                - **VAD Filter**: Bật
                - **Precision**: float16 (GPU) / int8 (CPU)
                
                ### 📚 Hỗ Trợ
                Nếu gặp lỗi, hãy kiểm tra:
                1. File `.env` có chứa API Key?
                2. GPU có hoạt động? → Chạy `python setup_gpu.py`
                3. Dependencies cài đủ? → Chạy `pip install -r requirements.txt`
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
