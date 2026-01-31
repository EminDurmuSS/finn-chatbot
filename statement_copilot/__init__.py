"""
Statement Copilot
=================
AI-powered financial assistant with LangGraph orchestration.

bunq-aligned, production-ready architecture.

Quick Start:
-----------
    from statement_copilot import chat

    response = chat("How much did I spend this month?")
    print(response["answer"])

Full Usage:
----------
    from statement_copilot import StatementCopilot

    copilot = StatementCopilot()
    response = copilot.chat(
        message="How much did I spend on groceries this month?",
        tenant_id="my-tenant",
    )

API Server:
----------
    from statement_copilot.api import run_server
    run_server(host="0.0.0.0", port=8000)
"""

from .config import settings, get_settings
from .workflow import (
    StatementCopilot,
    get_copilot,
    chat,
    build_workflow,
)

__version__ = "2.0.0"
__author__ = "Statement Copilot Team"

__all__ = [
    "settings",
    "get_settings",
    "StatementCopilot",
    "get_copilot",
    "chat",
    "build_workflow",
    "__version__",
]
