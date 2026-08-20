"""Verify faster-whisper model works with int8 quantization (RAM-friendly).

Render free tier chỉ có 512MB RAM. Script này tải model từ Hugging Face Hub,
load bằng CTranslate2 với compute_type="int8" (lượng tử hóa ngay khi load),
rồi test transcription một mẫu âm thanh tổng hợp.

Usage:
    python quantize_model.py [model_size]

Examples:
    python quantize_model.py base
    python quantize_model.py tiny
"""

from __future__ import annotations

import sys
import time

import numpy as np


def main() -> int:
    model_size = sys.argv[1] if len(sys.argv) > 1 else "base"

    print("=" * 60)
    print(f"Verifying faster-whisper '{model_size}' with int8 quantization...")
    print("=" * 60)

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("faster-whisper is not installed. Run: pip install faster-whisper")
        return 1

    start = time.perf_counter()

    # device='cpu' — hoạt động trên Render free tier (CPU-only, 512MB RAM).
    # compute_type='int8' — giảm ~3x dung lượng so với float32, phù hợp RAM giới hạn.
    model = WhisperModel(
        model_size,
        device="cpu",
        compute_type="int8",
        download_root=None,  # dùng cache mặc định của HF
    )

    load_time = time.perf_counter() - start
    print(f"✅ Model loaded in {load_time:.1f}s")
    print(f"   Model: {model_size} (compute_type=int8)")
    print(f"   Device: CPU")

    # Test transcription với audio tổng hợp có âm thanh nhưng không phải tiếng nói.
    # Mục đích: xác nhận model có thể chạy inference không lỗi.
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)  # tone 440Hz 1s

    print("\nTesting transcription (synthetic tone -> no speech expected)...")
    start = time.perf_counter()
    segments, _info = model.transcribe(audio=audio, language="vi", beam_size=1, vad_filter=True)
    # Duyệt generator để đảm bảo inference chạy xong.
    transcripts = [seg.text.strip() for seg in segments if seg.text]
    infer_time = time.perf_counter() - start
    print(f"   Transcribe took {infer_time:.2f}s")
    print(f"   Transcript: {transcripts if transcripts else '(empty - expected cho tone)'}")

    print("\n" + "=" * 60)
    print("✅ Verification PASSED! Model int8 hoạt động tốt.")
    print("=" * 60)
    print("\nỨng dụng sẽ dùng model này tự động qua biến môi trường:")
    print(f"   WHISPER_MODEL={model_size}")
    print("   WHISPER_COMPUTE_TYPE=int8")
    print("\n(lưu trong render.yaml)")
    return 0


if __name__ == "__main__":
    sys.exit(main())