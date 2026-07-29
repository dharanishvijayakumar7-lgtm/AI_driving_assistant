"""
api/ — FastAPI service layer for the AI Driving Assistant.

Day 7: Wraps the existing Days 1-6 pipeline in a FastAPI app that streams
annotated frames (JPEG/base64) and structured JSON metadata over WebSocket.

Package layout:
    app.py               — FastAPI app instance, startup/shutdown lifecycle.
    websocket_handler.py — /ws/stream endpoint: runs the pipeline and broadcasts.
    schemas.py           — Pydantic models defining the per-frame JSON contract.
"""
