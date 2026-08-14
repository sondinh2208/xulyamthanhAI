# GPU Setup Guide for Speech Translation Pipeline

## ⚡ Quick Start (GPU-Enabled)

### Step 1: Check GPU Setup
```bash
python setup_gpu.py
```

This will tell you:
- ✅ If NVIDIA GPU is detected
- ✅ If PyTorch has CUDA support
- ❌ What's missing (if anything)

### Step 2: Install PyTorch with CUDA Support

**Option A: CUDA 12.1 (Recommended)**
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Option B: CUDA 11.8**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 3: Verify Installation
```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

### Step 4: Install All Dependencies
```bash
pip install -r requirements.txt
pip install python-dotenv
```

### Step 5: Set Groq API Key
```bash
# On Windows PowerShell:
$env:Groq_API_KEY = ""

# On Windows CMD:
set Groq_API_KEY=""

# On Linux/Mac:
export Groq_API_KEY=your-api-key-here
```

### Step 6: Run the Application
```bash
python main.py
```

Then open: **http://127.0.0.1:7860** in your browser

---

## 📋 Prerequisites

### 1. NVIDIA Graphics Card
- Must support CUDA (GeForce RTX, Tesla, Quadro, etc.)
- Compute Capability 3.5 or higher

### 2. NVIDIA CUDA Toolkit
- Download: https://developer.nvidia.com/cuda-downloads
- Version 11.8 or 12.1 recommended
- Add to PATH

### 3. NVIDIA cuDNN (Optional but Recommended)
- Download: https://developer.nvidia.com/cudnn
- Extract to CUDA directory

### 4. Python 3.9+
- Verify: `python --version`

---

## ✅ Performance Expectations

### With GPU (CUDA enabled)
- STT (Speech-to-Text): ~0.3-0.5 seconds ✅
- LLM (Translation): ~1-2 seconds ✅
- TTS (Text-to-Speech): ~1-2 seconds ✅
- **Total: ~3-5 seconds** ✅

### Without GPU (CPU only)
- STT: ~5-10 seconds ❌
- LLM: ~1-2 seconds
- TTS: ~1-2 seconds
- **Total: ~10-15 seconds** ❌

---

## 🔍 Troubleshooting

### GPU Not Detected
```bash
# Check NVIDIA drivers
nvidia-smi

# Check if CUDA drivers are in PATH
where nvcc  # Windows
which nvcc  # Linux/Mac
```

**Solution**: Install latest NVIDIA drivers and CUDA Toolkit

### PyTorch Reports "No CUDA"
```bash
# Verify PyTorch installation
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall with correct URL
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Model Download Issues
First run downloads the Whisper model (~3GB). Ensure:
- Stable internet connection
- Sufficient disk space (~5GB)
- If slow, the app will still work but STT will be slower

### Out of Memory
If you get CUDA OOM errors:
- Close other GPU applications
- Try the smaller model: change `main.py` line to:
  ```python
  def __init__(self, model_name: str = "deepdml/faster-whisper-base-turbo") -> None:
  ```

---

## 📊 GPU Model Selection

| Model | VRAM | Speed | Accuracy |
|-------|------|-------|----------|
| tiny | 1GB | Very Fast | 90% |
| base | 2GB | Fast | 92% |
| small | 3GB | Moderate | 94% |
| medium | 5GB | Slow | 95% |
| large-v3-turbo | 6GB | Fast | 96% |

For best results with <5s target, use **large-v3-turbo** on GPU.

---

## 🚀 Optimization Tips

1. **Use FP16 (Float16)** - Automatically enabled on GPU for 2x speedup
2. **Pre-load Model** - Model loads once at startup (not per translation)
3. **Monitor GPU**: `nvidia-smi -l 1` (updates every second)
4. **Check Temperatures**: Should stay below 80°C

---

## 📞 Support

If you still have issues:
1. Run `python setup_gpu.py` for detailed diagnostics
2. Check NVIDIA driver version: `nvidia-smi`
3. Verify PyTorch CUDA: `python -c "import torch; print(torch.version.cuda)"`

Good luck! 🚀
