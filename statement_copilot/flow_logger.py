"""
Statement Copilot - Flow Logger
===============================
Professional workflow visualization logging.

Usage:
    with flow_context(trace_id, session_id) as flow:
        flow.request("User message")

        with flow.node("input_guard"):
            flow.detail("passed", True)

        with flow.node("orchestrator"):
            flow.detail("intent", "ANALYTICS")
            flow.detail("routing", ["finance_analyst"])

        flow.response("Final answer")
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

# Custom log level for flow visualization
FLOW_LEVEL = 25
logging.addLevelName(FLOW_LEVEL, "FLOW")


def _is_verbose() -> bool:
    """Check if verbose mode is enabled via environment variable."""
    return os.environ.get("DEBUG_VERBOSE", "").lower() in ("true", "1", "yes")


def _clip(text: Any, max_len: int = 80, force_clip: bool = False) -> str:
    """Truncate text for display. In verbose mode, shows full text unless force_clip=True."""
    if text is None:
        return ""
    s = str(text).replace("\n", " ").strip()

    # In verbose mode, don't clip unless explicitly requested
    if _is_verbose() and not force_clip:
        return s

    if len(s) <= max_len:
        return s
    return s[:max_len - 3] + "..."


def _format_value(value: Any, verbose: Optional[bool] = None) -> str:
    """Format a value for display. In verbose mode, shows full content."""
    if verbose is None:
        verbose = _is_verbose()

    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, list):
        if len(value) == 0:
            return "[]"
        if verbose:
            return "[" + ", ".join(str(v) for v in value) + "]"
        return "[" + ", ".join(str(v) for v in value[:3]) + (", ..." if len(value) > 3 else "") + "]"
    if isinstance(value, dict):
        if len(value) == 0:
            return "{}"
        if verbose:
            return "{" + ", ".join(f"{k}={v}" for k, v in value.items()) + "}"
        items = list(value.items())[:3]
        return "{" + ", ".join(f"{k}={v}" for k, v in items) + ("..." if len(value) > 3 else "") + "}"
    return str(value)


@dataclass
class FlowStep:
    """Single step in the flow."""
    node: str
    status: str = "running"  # running, done, error
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)
    is_last: bool = False

    @property
    def duration_ms(self) -> Optional[int]:
        if self.end_time:
            return int((self.end_time - self.start_time) * 1000)
        return None


class FlowLogger:
    """
    Professional workflow visualization logger.

    Creates clean, readable logs showing:
    - Incoming request
    - Node transitions with timing
    - Key details at each step
    - Final response with total time
    """

    HEADER_CHAR = "="
    HEADER_WIDTH = 70

    def __init__(
        self,
        trace_id: str,
        session_id: Optional[str] = None,
    ):
        self.trace_id = trace_id[:8] if len(trace_id) > 8 else trace_id
        self.session_id = session_id[:8] if session_id and len(session_id) > 8 else session_id
        self.logger = logging.getLogger("flow")
        self.steps: List[FlowStep] = []
        self.current_step: Optional[FlowStep] = None
        self.start_time = time.time()
        self._pending_details: Dict[str, Any] = {}

    def _header(self) -> str:
        return self.HEADER_CHAR * self.HEADER_WIDTH

    def _log(self, message: str):
        """Log at FLOW level."""
        self.logger.log(FLOW_LEVEL, message)

    def request(self, message: str, **metadata):
        """Log incoming request."""
        lines = [
            self._header(),
            f"[REQUEST] trace={self.trace_id}",
            f'  "{_clip(message, 60)}"',
        ]

        if self.session_id:
            lines[1] += f" | session={self.session_id}"

        for key, value in metadata.items():
            lines.append(f"  {key}: {_format_value(value)}")

        lines.append(self._header())
        self._log("\n".join(lines))

    @contextmanager
    def node(self, name: str):
        """
        Context manager for a workflow node.

        Usage:
            with flow.node("orchestrator"):
                flow.detail("intent", "ANALYTICS")
                # do work
        """
        step = FlowStep(node=name)
        self.steps.append(step)
        self.current_step = step
        self._pending_details = {}

        try:
            yield step
            step.status = "done"
        except Exception as e:
            step.status = "error"
            step.details["error"] = str(e)
            raise
        finally:
            step.end_time = time.time()
            self.current_step = None
            self._output_step(step)

    def detail(self, key: str, value: Any):
        """Add detail to current node."""
        if self.current_step:
            self.current_step.details[key] = value
        else:
            self._pending_details[key] = value

    def sql(self, query: str, params: Optional[list] = None, row_count: Optional[int] = None, duration_ms: Optional[int] = None):
        """
        Log SQL query execution with formatting.
        In verbose mode, shows full query. Otherwise, shows abbreviated.
        """
        if self.current_step:
            verbose = _is_verbose()

            # Format query
            if verbose:
                # Show full query with proper formatting
                query_display = query.strip().replace("\n", "\n|       ")
            else:
                # Show abbreviated query
                query_display = _clip(query.strip().replace("\n", " "), 100)

            # Format params
            if params:
                params_display = _format_value(params, verbose=verbose)
            else:
                params_display = None

            # Build SQL detail string
            sql_info = f"[SQL] {query_display}"
            if params_display:
                sql_info += f"\n|       params: {params_display}"
            if row_count is not None:
                sql_info += f" -> {row_count} rows"
            if duration_ms is not None:
                sql_info += f" ({duration_ms}ms)"

            # Store as a special detail
            sql_details = self.current_step.details.get("_sql", [])
            sql_details.append({
                "query": query_display,
                "params": params_display,
                "row_count": row_count,
                "duration_ms": duration_ms,
            })
            self.current_step.details["_sql"] = sql_details

    def substep(self, name: str, **details):
        """
        Log a substep within the current node.
        Useful for showing stages like "Query Understanding", "Retrieval", etc.
        """
        if self.current_step:
            substeps = self.current_step.details.get("_substeps", [])
            substeps.append({"name": name, **details})
            self.current_step.details["_substeps"] = substeps

    def verbose_detail(self, key: str, value: Any):
        """
        Add detail that only shows in verbose mode.
        In non-verbose mode, this is a no-op.
        """
        if _is_verbose() and self.current_step:
            self.current_step.details[key] = value

    def _output_step(self, step: FlowStep):
        """Output a completed step."""
        # Determine prefix based on position
        is_last = step.is_last
        prefix = "+->" if is_last else "|->"

        # Status indicator
        if step.status == "done":
            status = "ok" if step.duration_ms and step.duration_ms < 100 else ""
        elif step.status == "error":
            status = "ERR"
        else:
            status = ""

        # Build node line
        duration = f"({step.duration_ms}ms)" if step.duration_ms else ""
        status_suffix = f" {status}" if status else ""
        node_line = f"{prefix} {step.node}{status_suffix} {duration}"

        lines = [node_line]

        # Add details with proper indentation
        detail_prefix = "    " if is_last else "|   "
        verbose = _is_verbose()

        for key, value in step.details.items():
            # Handle special keys
            if key == "_sql":
                # Format SQL queries
                for sql_info in value:
                    query = sql_info.get("query", "")
                    params = sql_info.get("params")
                    row_count = sql_info.get("row_count")
                    duration_ms = sql_info.get("duration_ms")

                    if verbose:
                        # Multi-line SQL display
                        lines.append(f"{detail_prefix}[SQL]")
                        for sql_line in query.split("\n"):
                            lines.append(f"{detail_prefix}  {sql_line.strip()}")
                        if params:
                            lines.append(f"{detail_prefix}  params: {params}")
                        result_info = []
                        if row_count is not None:
                            result_info.append(f"{row_count} rows")
                        if duration_ms is not None:
                            result_info.append(f"{duration_ms}ms")
                        if result_info:
                            lines.append(f"{detail_prefix}  -> {', '.join(result_info)}")
                    else:
                        # Compact SQL display
                        compact_query = _clip(query.replace("\n", " "), 60, force_clip=True)
                        result_parts = []
                        if row_count is not None:
                            result_parts.append(f"{row_count} rows")
                        if duration_ms is not None:
                            result_parts.append(f"{duration_ms}ms")
                        result_str = f" -> {', '.join(result_parts)}" if result_parts else ""
                        lines.append(f"{detail_prefix}[SQL] {compact_query}{result_str}")

            elif key == "_substeps":
                # Format substeps
                for substep in value:
                    substep_name = substep.pop("name", "substep")
                    lines.append(f"{detail_prefix}|-- [{substep_name}]")
                    for sub_key, sub_value in substep.items():
                        formatted = _format_value(sub_value, verbose=verbose)
                        lines.append(f"{detail_prefix}|   {sub_key}: {formatted}")

            elif not key.startswith("_"):
                # Regular details
                formatted = _format_value(value, verbose=verbose)
                lines.append(f"{detail_prefix}{key}: {formatted}")

        self._log("\n".join(lines))

    def mark_last_node(self):
        """Mark the next node as the last one."""
        if self.steps:
            self.steps[-1].is_last = True

    def response(self, answer: str, status: str = "success"):
        """Log final response."""
        total_s = time.time() - self.start_time

        status_prefix = "OK" if status == "success" else "ERR"

        lines = [
            self._header(),
            f"[RESPONSE] {status_prefix} | {total_s:.2f}s",
            f'  "{_clip(answer, 60)}"',
            self._header(),
        ]

        self._log("\n".join(lines))

    def error(self, message: str, exception: Optional[Exception] = None):
        """Log error."""
        lines = [f"[ERROR] {message}"]
        if exception:
            lines.append(f"  {type(exception).__name__}: {exception}")
        self._log("\n".join(lines))


class FlowFormatter(logging.Formatter):
    """
    Formatter for flow logs.
    Flow logs are output as-is, other logs get standard format.
    """

    def format(self, record: logging.LogRecord) -> str:
        if record.levelno == FLOW_LEVEL:
            return record.getMessage()

        # Standard format for non-flow logs
        timestamp = time.strftime("%H:%M:%S")
        level = record.levelname[:4]

        # Get context if available
        trace = getattr(record, 'trace_id', '-')
        node = getattr(record, 'node', '-')

        return f"{timestamp} | {level} | {trace} | {node} | {record.getMessage()}"
