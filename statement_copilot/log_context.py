"""
Statement Copilot - Logging Context Utilities
============================================
Helpers to add trace/session/node context to logs and format debug output.
"""

from __future__ import annotations

import contextvars
import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Optional, TYPE_CHECKING


def is_verbose_mode() -> bool:
    """Check if verbose debug mode is enabled via DEBUG_VERBOSE environment variable."""
    return os.environ.get("DEBUG_VERBOSE", "").lower() in ("true", "1", "yes")

if TYPE_CHECKING:
    from .flow_logger import FlowLogger

_trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="-")
_session_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("session_id", default="-")
_node_var: contextvars.ContextVar[str] = contextvars.ContextVar("node", default="-")
_flow_logger_var: contextvars.ContextVar[Optional["FlowLogger"]] = contextvars.ContextVar("flow_logger", default=None)


class LogContextFilter(logging.Filter):
    """Inject trace/session/node fields into log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = _trace_id_var.get()
        record.session_id = _session_id_var.get()
        record.node = _node_var.get()
        return True


@contextmanager
def log_context(
    *,
    trace_id: Optional[str] = None,
    session_id: Optional[str] = None,
    node: Optional[str] = None,
):
    """Temporarily set log context values for the current execution flow."""
    tokens: list[tuple[contextvars.ContextVar[str], contextvars.Token]] = []
    if trace_id is not None:
        tokens.append((_trace_id_var, _trace_id_var.set(trace_id)))
    if session_id is not None:
        tokens.append((_session_id_var, _session_id_var.set(session_id)))
    if node is not None:
        tokens.append((_node_var, _node_var.set(node)))
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


def get_log_context() -> Dict[str, str]:
    """Return the current logging context."""
    return {
        "trace_id": _trace_id_var.get(),
        "session_id": _session_id_var.get(),
        "node": _node_var.get(),
    }


def clip_text(value: Any, max_len: int = 200) -> str:
    """Stringify and clip text to keep logs readable."""
    if value is None:
        return ""
    text = str(value).replace("\n", " ").replace("\r", " ").strip()
    if len(text) <= max_len:
        return text
    if max_len <= 3:
        return text[:max_len]
    return text[: max_len - 3] + "..."


def format_kv(
    data: Optional[Dict[str, Any]],
    *,
    max_items: int = 6,
    max_value_len: int = 120,
) -> str:
    """Format a dict as compact key=value pairs for logs."""
    if not data:
        return "{}"
    items: list[str] = []
    for idx, (key, value) in enumerate(data.items()):
        if idx >= max_items:
            items.append(f"...+{len(data) - max_items} more")
            break
        items.append(f"{key}={clip_text(value, max_value_len)}")
    return "{" + ", ".join(items) + "}"


def format_list(
    values: Optional[Iterable[Any]],
    *,
    max_items: int = 6,
    max_value_len: int = 80,
) -> str:
    """Format a list/iterable compactly for logs."""
    if not values:
        return "[]"
    items = list(values)
    rendered: list[str] = []
    for idx, value in enumerate(items):
        if idx >= max_items:
            rendered.append(f"...+{len(items) - max_items} more")
            break
        rendered.append(clip_text(value, max_value_len))
    return "[" + ", ".join(rendered) + "]"


# ──────────────────────────────────────────────────────────────────────────────
# FLOW LOGGER CONTEXT
# ──────────────────────────────────────────────────────────────────────────────

@contextmanager
def flow_context(trace_id: str, session_id: Optional[str] = None):
    """
    Initialize flow logger for a request.

    Usage:
        with flow_context(trace_id, session_id) as flow:
            flow.request("User message")
            with flow.node("input_guard"):
                flow.detail("passed", True)
            flow.response("Answer")
    """
    from .flow_logger import FlowLogger

    flow = FlowLogger(trace_id=trace_id, session_id=session_id)
    token = _flow_logger_var.set(flow)
    try:
        yield flow
    finally:
        _flow_logger_var.reset(token)


def get_flow() -> Optional["FlowLogger"]:
    """Get current flow logger if active."""
    return _flow_logger_var.get()
