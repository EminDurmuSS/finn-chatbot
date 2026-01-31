"""
Statement Copilot - Orchestrator (Wrapper)
==========================================
Compatibility layer for the orchestrator module naming.
"""

from .orchestor import (
    OrchestratorAgent,
    ResponseSynthesizer,
    get_orchestrator,
    get_synthesizer,
)

__all__ = [
    "OrchestratorAgent",
    "ResponseSynthesizer",
    "get_orchestrator",
    "get_synthesizer",
]
