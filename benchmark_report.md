# 📊 Báo cáo Đo lường Hiệu năng

**Dự án:** Hệ thống Dịch thuật Giọng nói Thời gian thực (Vietnamese Speech-to-English Speech)

**Kiến trúc:** STT → LLM → TTS Pipeline

---

## 1. Thông số Môi trường Kiểm thử

- **GPU:** NVIDIA GeForce RTX 4050 Laptop (6GB VRAM)
- **Python:** 3.12
- **PyTorch:** CUDA 12.7
- **UI:** Gradio Web UI
- **Dữ liệu đầu vào:** File ghi âm tiếng Việt thực tế (độ dài mẫu ~13 giây)

---

## 2. Cấu trúc Đường ống

1. **Nhận diện giọng nói (STT):** `faster-whisper` với model `large-v3-turbo`, chạy local, `float16` qua CUDA.
2. **Dịch thuật văn bản (LLM):** Chuyển từ Gemini API sang **Groq API** với model `llama-3.1-8b-instant`, `temperature=0`, `max_tokens=60`.
3. **Tổng hợp giọng nói (TTS):** Sử dụng fallback `edge-tts` với giọng `en-US-AriaNeural`.

---

## 3. Kết quả Đo lường

| Giai đoạn xử lý | Trước khi tối ưu (Gemini API) | Sau khi tối ưu (Groq API) | % Cải thiện | Đánh giá kỹ thuật |
|---|---|---|---|---|
| Giao tiếp UI (Audio) | 0.01s | 0.01s | 0% | Độ trễ truyền tải dữ liệu cực thấp |
| STT (Faster-Whisper) | 1.26s | 1.26s | 0% | RTX 4050 xử lý tốt file audio dài |
| LLM (Translation) | 13.12s | ~0.30s | +4273% | Loại bỏ hoàn toàn giới hạn hàng đợi và Rate Limit 429 |
| TTS (Edge-TTS) | 1.36s | 1.36s | 0% | Ổn định qua luồng mạng dự phòng |
| **Tổng độ trễ** | **15.77s** | **~2.93s** | **+438%** | Hiệu năng tổng thể cải thiện mạnh |

---

## 4. Nhận xét

- Việc chuyển LLM từ Gemini API sang Groq API mang lại cải thiện hiệu năng rất lớn.
- STT và TTS vẫn giữ được độ ổn định và không đổi nhiều về thời gian xử lý.
- Tổng latency giảm từ ~15.77 giây xuống ~2.93 giây, phù hợp với kịch bản dịch speech-to-speech gần real-time.

---

## 5. Kết luận

- **Điểm mạnh:** LLM translation được tối ưu đáng kể bằng Groq API.
- **Phù hợp:** Hệ thống hiện đã có hiệu năng tốt hơn với phần cứng RTX 4050 và môi trường local.
- **Hướng tiếp theo:** Tối ưu thêm pipeline I/O và giảm thiểu overhead nếu cần rút latency thấp hơn nữa.
