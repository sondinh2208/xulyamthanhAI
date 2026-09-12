"""Entry point: assembles the Gradio interface (ui.py), starts the local server
and exposes it to the internet through an ngrok tunnel (pyngrok).

Run the app with:  python main.py

Gradio always binds to 127.0.0.1:7860 (share=True is never used). A public
https://*.ngrok-free.app URL is opened with pyngrok and printed here.
Cloudflare quick tunnels (trycloudflare.com) are no longer used because some
ISPs block their TLS traffic. The ngrok process is killed when the app exits
(Ctrl+C included), so no hidden tunnel is left behind.
"""

import atexit
import os
import signal
import sys

from pyngrok import ngrok

import config
from ui import build_ui


_public_url: str | None = None
_shutdown_done = False


def start_ngrok_tunnel() -> str | None:
    """Opens an ngrok tunnel to the local Gradio port and returns its public URL."""
    global _public_url

    # pyngrok needs an authtoken (free account: https://dashboard.ngrok.com).
    # The token is read from the environment / .env file (see config.py).
    authtoken = os.getenv(config.NGROK_AUTHTOKEN_ENV, "").strip()
    if authtoken:
        ngrok.set_auth_token(authtoken)

    try:
        # https://xxxx.ngrok-free.app -> http://127.0.0.1:7860
        tunnel = ngrok.connect(config.NGROK_PORT)
        _public_url = tunnel.public_url
    except Exception as exc:
        print(f"[TUNNEL] Could not open the ngrok tunnel: {exc}")
        print("[TUNNEL] The app stays local-only. Set NGROK_AUTHTOKEN in the .env file.")
        return None

    print(f"\n>>> PUBLIC URL (NGROK): {_public_url} <<<\n")
    print("=" * 70)
    print("  🌐 Share this link to open the app from anywhere (ngrok tunnel, low latency).")
    print("=" * 70 + "\n")
    return _public_url


def shutdown_tunnel() -> None:
    """Kills the ngrok process so no hidden tunnel survives the app (idempotent)."""
    global _shutdown_done
    if _shutdown_done:
        return
    _shutdown_done = True
    try:
        ngrok.kill()
        if _public_url:
            print("[TUNNEL] ngrok tunnel closed.")
    except Exception as exc:
        print(f"[TUNNEL] Error while closing the ngrok tunnel: {exc}")


def run_with_tunnel(demo) -> None:
    """Launches Gradio locally with a managed ngrok tunnel around it."""
    if config.NGROK_ENABLED:
        atexit.register(shutdown_tunnel)  # runs on every exit path, even unhandled errors
        # Make sure Ctrl+C (SIGINT/SIGTERM) also tears the tunnel down
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, lambda signum, frame: (shutdown_tunnel(), sys.exit(0)))
            except (ValueError, OSError):
                pass

        start_ngrok_tunnel()

    try:
        demo.launch(
            server_name=config.SERVER_NAME,
            server_port=config.DEFAULT_SERVER_PORT,
            share=config.SHARE_LINK,  # False — Gradio share fully replaced by the ngrok tunnel
        )
    except KeyboardInterrupt:
        shutdown_tunnel()  # Ctrl+C inside the Gradio server loop
    finally:
        shutdown_tunnel()


if __name__ == "__main__":
    demo = build_ui()
    demo.queue()  # keep the queue for optimised WebSocket handling
    run_with_tunnel(demo)