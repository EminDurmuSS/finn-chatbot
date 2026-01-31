"""
Statement Copilot - API
=======================
FastAPI application with REST endpoints.
"""

import logging
import time
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field

from ..config import settings
from ..core import (
    ChatRequest,
    ChatResponse,
    ActionApprovalRequest,
    Evidence,
)
from ..workflow import get_copilot, StatementCopilot

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# LIFESPAN
# ═══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("Starting Statement Copilot API...")
    
    # Initialize copilot
    _ = get_copilot()
    
    logger.info("Statement Copilot API ready")
    
    yield
    
    logger.info("Shutting down Statement Copilot API...")


# ═══════════════════════════════════════════════════════════════════════════════
# APP
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Statement Copilot API",
    description="AI-powered financial assistant with LangGraph orchestration",
    version=settings.app_version,
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════════
# REQUEST/RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class MessageRequest(BaseModel):
    """Chat message request"""
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: Optional[str] = None
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None


class MessageResponse(BaseModel):
    """Chat message response"""
    answer: str
    session_id: str
    trace_id: str
    intent: Optional[str] = None
    confidence: Optional[float] = None
    evidence: Optional[Dict[str, Any]] = None
    suggestions: Optional[List[str]] = None
    needs_confirmation: bool = False
    action_plan: Optional[Dict[str, Any]] = None
    action_result: Optional[Dict[str, Any]] = None
    total_latency_ms: Optional[int] = None
    warnings: Optional[List[str]] = None
    error: Optional[str] = None


class ActionConfirmRequest(BaseModel):
    """Action confirmation request"""
    session_id: str
    action_id: str
    approved: bool
    reason: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    environment: str


# ═══════════════════════════════════════════════════════════════════════════════
# DEPENDENCIES
# ═══════════════════════════════════════════════════════════════════════════════

def get_copilot_dependency() -> StatementCopilot:
    """Dependency for copilot instance"""
    return get_copilot()


# ═══════════════════════════════════════════════════════════════════════════════
# MIDDLEWARE
# ═══════════════════════════════════════════════════════════════════════════════

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests"""
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} "
        f"status={response.status_code} "
        f"duration={duration:.3f}s"
    )
    
    return response


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        environment=settings.environment,
    )


@app.post("/chat", response_model=MessageResponse)
async def chat(
    request: MessageRequest,
    copilot: StatementCopilot = Depends(get_copilot_dependency),
):
    """
    Process a chat message.
    
    Main endpoint for user interactions.
    """
    try:
        result = copilot.chat(
            message=request.message,
            session_id=request.session_id,
            tenant_id=request.tenant_id,
            user_id=request.user_id,
        )
        
        return MessageResponse(**result)
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/confirm", response_model=MessageResponse)
async def confirm_action(
    request: ActionConfirmRequest,
    copilot: StatementCopilot = Depends(get_copilot_dependency),
):
    """
    Confirm or reject a pending action.
    
    Called after user reviews action plan.
    """
    try:
        result = copilot.confirm_action(
            session_id=request.session_id,
            action_id=request.action_id,
            approved=request.approved,
            reason=request.reason,
        )
        
        return MessageResponse(**result)
        
    except Exception as e:
        logger.error(f"Confirm error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/session/{session_id}/state")
async def get_session_state(
    session_id: str,
    copilot: StatementCopilot = Depends(get_copilot_dependency),
):
    """
    Get current state for a session.
    
    Useful for debugging and recovery.
    """
    state = copilot.get_state(session_id)
    
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Return sanitized state (remove sensitive data)
    return {
        "session_id": session_id,
        "intent": state.get("intent"),
        "confidence": state.get("confidence"),
        "needs_confirmation": state.get("needs_confirmation"),
        "pending_action_id": state.get("pending_action_id"),
        "completed_at": state.get("completed_at"),
    }


@app.get("/download/{filename}")
async def download_file(filename: str):
    """
    Download generated file.
    
    Returns exported files (Excel, CSV, PDF).
    """
    filepath = settings.get_outputs_path() / filename
    
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    # Security check: ensure file is in outputs directory
    try:
        filepath.resolve().relative_to(settings.get_outputs_path().resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return FileResponse(
        path=filepath,
        filename=filename,
        media_type="application/octet-stream",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# QUICK ENDPOINTS (for testing)
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/quick/sum")
async def quick_sum_endpoint(
    tenant_id: str = settings.default_tenant_id,
    categories: Optional[str] = None,
):
    """Quick sum calculation without LLM"""
    from ..agents import quick_sum
    from datetime import date, timedelta
    
    today = date.today()
    date_start = today.replace(day=1)
    
    category_list = categories.split(",") if categories else None
    
    result = quick_sum(
        tenant_id=tenant_id,
        date_start=date_start,
        date_end=today,
        categories=category_list,
    )
    
    return {"sum": result, "date_start": date_start, "date_end": today}


@app.get("/quick/search")
async def quick_search_endpoint(
    query: str,
    tenant_id: str = settings.default_tenant_id,
    top_k: int = 10,
):
    """Quick transaction search"""
    from ..agents import quick_search
    
    results = quick_search(
        query=query,
        tenant_id=tenant_id,
        top_k=top_k,
    )
    
    return {"results": results, "count": len(results)}


# ═══════════════════════════════════════════════════════════════════════════════
# ERROR HANDLERS
# ═══════════════════════════════════════════════════════════════════════════════

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.debug else "An error occurred",
        },
    )


# ═══════════════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════════════

def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the API server"""
    import uvicorn
    
    uvicorn.run(
        "statement_copilot.api.main:app",
        host=host,
        port=port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    run_server()