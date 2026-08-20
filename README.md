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

## ⚡ Lượng tử hóa cho Render (RAM 512MB)

Model Whisper mặc định **`base`** chạy với **`compute_type=int8`** (nén ~2–3 lần), tối ưu cho CPU free tier của Render.

| Model size | float32 (gốc) | int8 (lượng tử hóa) | Hợp RAM 512MB? |
|---|---|---|---|
| `tiny` | ~151 MB | ~50 MB | ✅ Rất tốt |
| `base` | ~290 MB | ~100 MB | ✅ Tốt |
| `small` | ~966 MB | ~460 MB | ❌ Vượt ~ khi cộng Python/Gradio |

### Xác minh model int8 hoạt động (tùy chọn)

Chạy script để tải model về cache máy và xác minh lượng tử hóa int8 hoạt động:

```bash
python quantize_model.py base
```

Script này tải model `base` từ Hugging Face, load với `compute_type=int8`,
test transcription một mẫu âm thanh, và báo thời gian load/inference.

> **Lưu ý:** Model được cache trong `~/.cache/huggingface` (ngoài repo) — không đẩy lên git.
> Render sẽ tự tải model qua `faster-whisper` từ Hugging Face ở lần khởi động đầu tiên
> và cache trong đĩa (thư mục `.cache/` đã được thêm vào `.gitignore`).

## Deploy lên Render (Web)

1. **Tạo repository GitHub** và push code:
   ```bash
   git add .
   git commit -m "Optimize for Render: int8 quantized Whisper base"
   git push origin main
   ```

2. **Truy cập [render.com](https://render.com)** → **New +** → **Blueprint**
3. **Chọn repository** — Render sẽ tự nhận diện `render.yaml`.
4. **Cài `GROQ_API_KEY`** tại: Dashboards → Service → Environment:
   - Key: `GROQ_API_KEY`
   - Value: (lấy từ console.groq.com)

5. **Chọn plan Free** (512MB RAM, CPU 0.1) và Deploy.

> Khi mới tạo, model sẽ được tải từ HF Hub trong lần deploy đầu (~1-2 phút quá trình build).
> Sau lần đầu, model được cache lại để khởi động nhanh hơn.

## Chạy local

```bash
pip install -r requirements.txt
```

Tạo file `.env`:

```
GROQ_API_KEY=your_groq_api_key
```

Chạy:

```bash
python app.py
```

## Biến môi trường

| Biến | Mô tả | Giá trị mặc định |
|---|---|---|
| `WHISPER_MODEL` | Tên model Whisper | `base` (an toàn RAM 512MB) |
| `WHISPER_COMPUTE_TYPE` | Kiểu dữ liệu tính toán | `int8` khi CPU |
| `GROQ_API_KEY` | API key dịch tiếng Anh | — (bắt buộc) |
| `PORT` | Cổng server | Render set tự động |

## Hugging Face Spaces (tùy chọn)

> **Note (2026):** Tạo Gradio Spaces miễn phí CPU yêu cầu PRO. Có thể dùng ZeroGPU
> cho tài khoản cũ hơn ~30 ngày. Nếu tạo Spaces thất bại HTTP 402, dùng local share hoặc Render.

1. Login: `hf auth login`
2. Chạy: `python deploy_space.py`

### Local public link (tạm thời)

```bash
set GRADIO_SHARE=true
python app.py
```

Gradio in ra URL `*.gradio.live` ✓