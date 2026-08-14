#!/usr/bin/env python3
"""
GPU Setup and Verification Script for Speech Translation Pipeline
Checks CUDA availability and provides installation instructions
"""

import subprocess
import sys


def check_nvidia_gpu():
    """Check if NVIDIA GPU is available."""
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected:")
            print(result.stdout)
            return True
    except FileNotFoundError:
        print("❌ nvidia-smi not found. NVIDIA GPU drivers may not be installed.")
    except subprocess.TimeoutExpired:
        print("❌ nvidia-smi timed out.")
    except Exception as e:
        print(f"❌ Error checking GPU: {e}")
    return False


def check_torch_cuda():
    """Check if PyTorch has CUDA support."""
    try:
        import torch

        if torch.cuda.is_available():
            print(f"✅ PyTorch CUDA is available")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   Device count: {torch.cuda.device_count()}")
            print(f"   Current device: {torch.cuda.get_device_name(0)}")
            print(f"   Compute capability: {torch.cuda.get_device_capability(0)}")
            return True
        else:
            print("❌ PyTorch installed but CUDA support is NOT available")
            print("   Current build: CPU-only")
            return False
    except ImportError:
        print("❌ PyTorch is not installed")
        return False


def check_faster_whisper():
    """Check if faster-whisper is installed and optimized."""
    try:
        from faster_whisper import WhisperModel

        print("✅ faster-whisper is installed")

        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            print(f"   Configured device: {device}")
            print(f"   Configured compute type: {compute_type}")
            return True
        except Exception as e:
            print(f"⚠️  faster-whisper found but error checking config: {e}")
            return False
    except ImportError:
        print("❌ faster-whisper is not installed")
        return False


def print_installation_instructions():
    """Print installation instructions for GPU support."""
    print("\n" + "=" * 70)
    print("INSTALLATION INSTRUCTIONS")
    print("=" * 70)

    print("\n1. Install NVIDIA CUDA Toolkit (if not already installed):")
    print("   - Download from: https://developer.nvidia.com/cuda-downloads")
    print("   - Choose your OS and CUDA version (12.1 or higher recommended)")
    print("   - Install and add CUDA/bin to PATH")

    print("\n2. Install cuDNN (if not already installed):")
    print("   - Download from: https://developer.nvidia.com/cudnn")
    print("   - Extract and add to CUDA path")

    print("\n3. Install PyTorch with CUDA support:")
    print("   pip uninstall torch torchvision torchaudio -y")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")

    print("\n4. Verify GPU setup:")
    print("   python setup_gpu.py")

    print("\n5. Install other dependencies:")
    print("   pip install -r requirements.txt")

    print("\n" + "=" * 70)


def main():
    print("=" * 70)
    print("GPU SETUP VERIFICATION FOR SPEECH TRANSLATION PIPELINE")
    print("=" * 70 + "\n")

    gpu_available = check_nvidia_gpu()
    print()

    torch_cuda = check_torch_cuda()
    print()

    whisper_ready = check_faster_whisper()
    print()

    if gpu_available and torch_cuda and whisper_ready:
        print("✅✅✅ GPU setup is COMPLETE and READY!")
        print("    Your pipeline will run at optimal speed (<5 seconds)")
        return 0
    else:
        print_installation_instructions()
        return 1


if __name__ == "__main__":
    sys.exit(main())
