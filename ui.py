"""Gradio web interface for the Vietnamese -> English speech-translation app.

This module only deals with the user interface:
  CUSTOM_CSS / build_ui_theme()  -> look & feel
  render_card_header()           -> section header HTML
  build_ui()                     -> complete Blocks layout + event wiring
"""

from typing import Optional

import gradio as gr

import config
from nodes import SpeechTranslationPipeline

CUSTOM_CSS = """
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
def build_ui_theme() -> gr.themes.Base:
    """Builds the custom theme used by the Gradio app."""
    return gr.themes.Base(
        primary_hue=gr.themes.colors.purple,
        secondary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
    )


def render_card_header(icon: str, title: str) -> str:
    """Builds the HTML for a glass-card section header."""
    return f"""
                <div class='glass-card'>
                    <div class='card-title'><span class='card-icon'>{icon}</span> {title}</div>
                </div>
                """


def build_ui() -> gr.Blocks:
    """Builds the complete Gradio Blocks interface."""
    pipeline = SpeechTranslationPipeline()

    def translate_audio(audio_path: Optional[str]):
        """Pipeline handler: called when audio is recorded or uploaded."""
        if not audio_path:
            return "", "", None, "⚠️ Vui lòng ghi âm hoặc tải tệp âm thanh trước!"
        return pipeline.run(audio_path)

    with gr.Blocks(title=config.APP_TITLE, css=CUSTOM_CSS, theme=build_ui_theme()) as demo:
        # Header / hero section
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

        # Row 1: input audio + output audio
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML(render_card_header("🎙️", "Input: Tiếng Việt"))
                audio_input = gr.Audio(
                    label="Ghi âm hoặc tải tệp",
                    sources=["microphone", "upload"],
                    type="filepath",
                    streaming=False,
                    elem_classes=["audio-input"],
                )

            with gr.Column(scale=1):
                gr.HTML(render_card_header("🔊", "Output: Tiếng Anh"))
                audio_output = gr.Audio(
                    label="Phát âm tiếng Anh",
                    type="filepath",
                    interactive=False,
                    autoplay=True,
                    elem_classes=["audio-output"],
                )

        # Status bar
        status_box = gr.Textbox(
            label="📊 Trạng Thái",
            lines=2,
            interactive=False,
            placeholder="🎧 Chờ ghi âm hoặc tải tệp âm thanh — quá trình dịch sẽ chạy tự động...",
            elem_classes=["textbox-output"],
        )

        # Row 2: original Vietnamese text + translated English text
        with gr.Row():
            with gr.Column():
                gr.HTML(render_card_header("📝", "Văn Bản Gốc (Tiếng Việt)"))
                original_text = gr.Textbox(
                    label="Transcript",
                    lines=5,
                    interactive=False,
                    placeholder="Văn bản gốc sẽ hiện ở đây...",
                    elem_classes=["textbox-output"],
                )

            with gr.Column():
                gr.HTML(render_card_header("🌍", "Văn Bản Dịch (Tiếng Anh)"))
                translated_text = gr.Textbox(
                    label="Translation",
                    lines=5,
                    interactive=False,
                    placeholder="Văn bản dịch sẽ hiện ở đây...",
                    elem_classes=["textbox-output"],
                )

        # Auto-run the pipeline when (1) the mic stops recording or (2) a file is uploaded
        translation_outputs = [original_text, translated_text, audio_output, status_box]
        audio_input.stop_recording(fn=translate_audio, inputs=[audio_input], outputs=translation_outputs)
        audio_input.upload(fn=translate_audio, inputs=[audio_input], outputs=translation_outputs)

        # Footer
        gr.HTML("""
        <div class='footer'>
            Made with <span class='heart'>❤</span> · Dịch Giọng Nói Việt - Anh
        </div>
        """)

    return demo