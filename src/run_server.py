"""
run_server.py — Convenience entry point for the Day 7 FastAPI server.

Equivalent to running from the project root:
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

Why a separate script instead of just the uvicorn CLI?
  - Keeps the working directory and sys.path consistent without requiring the
    user to remember to set PYTHONPATH or run from a specific directory.
  - Makes it easy to wire up the server as a PyCharm/VS Code launch config.
  - The uvicorn.run() call is programmatic, so you can override host/port/reload
    via environment variables without touching any source file.

Usage:
    # From the project root (recommended):
    python src/run_server.py

    # With overrides:
    HOST=0.0.0.0 PORT=9000 python src/run_server.py
"""

import os
import sys
from pathlib import Path

# ── Bootstrap sys.path ────────────────────────────────────────────────────────
# Ensure the project root is on sys.path so `from src.api.app import app` works
# regardless of which directory the user invokes this script from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import uvicorn  # noqa: E402 — must import after sys.path is patched

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    reload = os.environ.get("RELOAD", "true").lower() in ("1", "true", "yes")

    print(f"\n{'='*60}")
    print("  AI Driving Assistant — Day 7 API Server")
    print(f"  Listening on : http://{host}:{port}")
    print(f"  WebSocket    : ws://{host}:{port}/ws/stream")
    print(f"  Swagger UI   : http://localhost:{port}/docs")
    print(f"  Hot reload   : {'enabled' if reload else 'disabled'}")
    print(f"{'='*60}\n")

    uvicorn.run(
        "src.api.app:app",
        host=host,
        port=port,
        reload=reload,
        # app_dir is NOT set — we rely on the sys.path.insert above.
        # reload=True requires the app to be specified as an import string,
        # which is why we use "src.api.app:app" rather than importing the
        # object directly.
        log_level="info",
    )
