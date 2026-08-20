"""Deploy this Gradio app to Hugging Face Spaces (when account is eligible).

HF policy (2026): creating Gradio/Docker Spaces requires PRO, OR a free account
older than ~30 days may use ZeroGPU (up to 2 Spaces).

Usage (after: hf auth login):
  python deploy_space.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

SPACE_ID = os.getenv("HF_SPACE_ID", "sonb2208/dich-giong-noi-viet-anh")
FLAVOR = os.getenv("HF_SPACE_FLAVOR", "zero-a10g")  # free-eligible hardware when allowed

UPLOAD_FILES = [
    "app.py",
    "requirements.txt",
    "packages.txt",
    "README.md",
]


def main() -> int:
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("Install huggingface_hub first: pip install huggingface_hub")
        return 1

    api = HfApi()
    try:
        who = api.whoami()
        print(f"Logged in as: {who.get('name') or who}")
    except Exception as exc:
        print(f"Not logged in. Run: hf auth login\n({exc})")
        return 1

    print(f"Creating Space {SPACE_ID} (sdk=gradio, flavor={FLAVOR})...")
    try:
        create_repo(
            repo_id=SPACE_ID,
            repo_type="space",
            space_sdk="gradio",
            space_hardware=FLAVOR,
            private=False,
            exist_ok=True,
        )
    except Exception as exc:
        print(f"Failed to create Space: {exc}")
        print(
            "If you see HTTP 402: subscribe to PRO, wait ~30 days for ZeroGPU, "
            "or request a community grant. Meanwhile run locally with GRADIO_SHARE=true."
        )
        return 1

    print("Uploading app files...")
    for name in UPLOAD_FILES:
        path = ROOT / name
        if not path.exists():
            print(f"  skip missing {name}")
            continue
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=name,
            repo_id=SPACE_ID,
            repo_type="space",
        )
        print(f"  uploaded {name}")

    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        print("Setting Space secret GROQ_API_KEY...")
        api.add_space_secret(SPACE_ID, "GROQ_API_KEY", groq_key)
        print("  secret set (value not printed)")
    else:
        print(
            "GROQ_API_KEY not found in .env — set it manually:\n"
            f"  https://huggingface.co/spaces/{SPACE_ID}/settings"
        )

    url = f"https://huggingface.co/spaces/{SPACE_ID}"
    print(f"\nDone. Space URL: {url}")
    print("First build may take several minutes while dependencies install.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
