"""Entry point: assembles the Gradio interface (ui.py) and starts the server.

Run the app with:  python main.py

The launch port comes from the PORT environment variable (used on Render/Heroku),
then from GRADIO_SERVER_PORT, and finally from DEFAULT_SERVER_PORT in config.py.
"""

import os

import config
from ui import build_ui

if __name__ == "__main__":
    demo = build_ui()
    port = int(os.environ.get("PORT", os.getenv("GRADIO_SERVER_PORT", str(config.DEFAULT_SERVER_PORT))))
    demo.launch(
        server_name=config.SERVER_NAME,
        server_port=port,
        share=config.SHARE_LINK,
    )