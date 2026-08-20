---
title: Dich Giong Noi Viet - Anh
emoji: 🎙️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.0.0
app_file: app.py
pinned: false
python_version: "3.12"
---

# Speech Translation Pipeline (Vietnamese → English)

Gradio app that:

1. **STT** — transcribes Vietnamese speech with `faster-whisper`
2. **LLM** — translates to English via Groq (`openai/gpt-oss-20b`)
3. **TTS** — synthesizes English audio with `edge-tts`

## Setup (local)

```bash
pip install -r requirements.txt
# Optional GPU torch:
# pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Create a `.env` file:

```
GROQ_API_KEY=your_groq_api_key
```

Run:

```bash
python app.py
```

## Hugging Face Spaces

> **Note (2026):** Creating Gradio Spaces on free CPU requires a **PRO** plan.
> Free accounts in good standing (often ~30 days old) can host up to **2 ZeroGPU** Spaces.
> If create fails with HTTP 402, use the local public share link or retry after eligibility.

### Deploy (when eligible)

1. Login: `hf auth login`
2. Ensure `.env` contains `GROQ_API_KEY=...`
3. Run: `python deploy_space.py`

The script creates `sonb2208/dich-giong-noi-viet-anh`, uploads files, and sets the `GROQ_API_KEY` secret.

Or set the secret in the UI: **Space → Settings → Variables and secrets → New secret** named `GROQ_API_KEY`.

### Local public link (temporary)

```bash
set GRADIO_SHARE=true
python app.py
```

Gradio prints a `*.gradio.live` URL (about 1 week).
