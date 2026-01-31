"""
Statement Copilot - API Module
==============================
FastAPI endpoints for the copilot.
"""

from .main import app, run_server

__all__ = ["app", "run_server"]
