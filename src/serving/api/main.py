"""
FastAPI-based API server for the Automated Code Review System
Provides REST endpoints and WebSocket support for code review inference
"""

import os
import sys
import uuid
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any
from contextlib import asynccontextmanager
import json

from fastapi import FastAPI, BackgroundTasks, HTTPException, WebSocket, WebSocketDisconnect, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator
import uvicorn
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Add parent directories to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Import project modules
from src.models.inference.optimized_inference import (
    OptimizedInferencePipeline,
    InferenceConfig,
    InferenceRequest,
    InferenceResponse
)
from src.models.inference.cache_manager import HybridInferenceCache, CacheConfig
from src.data.collectors.github_collector import GitHubCollector
from src.data.tokenizers.code_aware_tokenizer import CodeAwareTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ====================
# Metrics Configuration
# ====================

request_count = Counter("code_review_requests_total", "Total number of code review requests")
request_latency = Histogram("code_review_latency_seconds", "Request latency in seconds")
token_usage = Counter("token_usage_total", "Total tokens used across all requests")
cache_hits = Counter("cache_hits_total", "Total number of cache hits")
cache_misses = Counter("cache_misses_total", "Total number of cache misses")
active_connections = Gauge("websocket_connections_active", "Number of active WebSocket connections")
queue_size = Gauge("review_queue_size", "Current size of the review queue")
model_inference_time = Histogram("model_inference_seconds", "Model inference time")

# ====================
# Request/Response Models
# ====================

class CodeReviewRequest(BaseModel):
    """Request model for code review"""
    repo: str = Field(..., description="Repository name or URL")
    pr_number: Optional[int] = Field(None, description="Pull request number")
    diff: str = Field(..., description="Code diff to review")
    urgency: str = Field("normal", description="Priority level: low, normal, high, critical")
    context: Optional[str] = Field(None, description="Additional context for the review")
    language: Optional[str] = Field(None, description="Programming language")
    max_tokens: int = Field(500, ge=50, le=2000, description="Maximum tokens for response")
    temperature: float = Field(0.7, ge=0.0, le=1.0, description="Generation temperature")
    
    @field_validator('urgency')
    @classmethod
    def validate_urgency(cls, v):
        valid_urgencies = ['low', 'normal', 'high', 'critical']
        if v.lower() not in valid_urgencies:
            raise ValueError(f"Urgency must be one of {valid_urgencies}")
        return v.lower()

class CodeReviewResponse(BaseModel):
    """Response model for code review"""
    review_id: str
    status: str  # 'completed', 'queued', 'processing', 'failed'
    comments: Optional[List[Dict]] = None
    suggestions: Optional[List[Dict]] = None
    confidence: Optional[float] = None
    tokens_used: Optional[int] = None
    processing_time: Optional[float] = None
    cached: bool = False
    error: Optional[str] = None

class BatchReviewRequest(BaseModel):
    """Request model for batch code reviews"""
    reviews: List[CodeReviewRequest]
    batch_id: Optional[str] = None

class ReviewStatusResponse(BaseModel):
    """Response model for review status check"""
    review_id: str
    status: str
    progress: Optional[float] = None
    estimated_completion: Optional[str] = None
    result: Optional[CodeReviewResponse] = None

class HealthCheckResponse(BaseModel):
    """Response model for health check"""
    status: str
    version: str
    uptime: float
    model_loaded: bool
    cache_available: bool
    queue_size: int
    active_connections: int

# ====================
# Global State Management
# ====================

class AppState:
    """Application state manager"""
    def __init__(self):
        self.inference_pipeline: Optional[OptimizedInferencePipeline] = None
        self.cache: Optional[HybridInferenceCache] = None
        self.tokenizer: Optional[CodeAwareTokenizer] = None
        self.github_collector: Optional[GitHubCollector] = None
        self.pending_reviews: Dict[str, Any] = {}
        self.websocket_connections: Dict[str, WebSocket] = {}
        self.startup_time = datetime.now()
        self.is_ready = False

app_state = AppState()

# ====================
# Rate Limiting
# ====================

limiter = Limiter(key_func=get_remote_address)

# ====================
# Lifespan Management
# ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info("Starting up API server...")
    
    try:
        # Initialize cache
        cache_config = CacheConfig(
            redis_host=os.getenv("REDIS_HOST", "localhost"),
            redis_port=int(os.getenv("REDIS_PORT", 6379)),
            cache_ttl=int(os.getenv("CACHE_TTL", 3600)),
            enable_redis=os.getenv("ENABLE_REDIS", "true").lower() == "true",
            enable_memory=True,
            compression=True
        )
        app_state.cache = HybridInferenceCache(cache_config)
        logger.info("Cache initialized")
        
        # Initialize inference pipeline
        # Use smaller model for Mac development to avoid memory issues
        import torch
        has_cuda = torch.cuda.is_available()
        has_mps = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
        is_production = os.getenv("ENVIRONMENT", "development").lower() == "production"
        
        # Smart model selection based on available resources
        default_model = "codellama/CodeLlama-7b-hf"  # Default for all environments
        model_path = os.getenv("MODEL_PATH", default_model)
        
        # Configure quantization based on environment and device
        quantization_bits = 0  # Default: no quantization
        if has_cuda and is_production:
            # Production with CUDA: Use quantization for efficiency
            quantization_bits = int(os.getenv("QUANTIZATION_BITS", 4))
        elif has_cuda and not is_production:
            # Development with CUDA: Optional quantization
            quantization_bits = int(os.getenv("QUANTIZATION_BITS", 0))
        # Mac/CPU: Always disable quantization (quantization_bits = 0)
        
        # Device configuration
        device = "auto"
        dtype = torch.float16  # Default
        max_batch_size = 4     # Default
        compile_model = False  # Default
        enable_flash_attention = False  # Default
        
        if has_cuda:
            # CUDA available (production or development)
            device = "cuda"
            dtype = torch.float16
            max_batch_size = int(os.getenv("MAX_BATCH_SIZE", 8 if is_production else 4))
            compile_model = os.getenv("COMPILE_MODEL", "true" if is_production else "false").lower() == "true"
            enable_flash_attention = os.getenv("ENABLE_FLASH_ATTENTION", "true").lower() == "true"
        elif has_mps:
            # Apple Silicon Mac with high RAM
            device = "cpu"  # Use CPU instead of MPS to avoid buffer size issues
            dtype = torch.float32  # CPU works better with float32
            max_batch_size = int(os.getenv("MAX_BATCH_SIZE", 1))  # Conservative batch size
            compile_model = False  # Disable compilation on Mac
            logger.info("Mac with MPS detected: Using CPU to avoid MPS memory allocation issues")
        else:
            # CPU fallback
            device = "cpu"
            dtype = torch.float32  # CPU works better with float32
            max_batch_size = int(os.getenv("MAX_BATCH_SIZE", 1))
            compile_model = False  # Disable torch.compile on CPU
        
        # Memory and performance settings based on device
        if has_cuda and is_production:
            # Production CUDA: Optimize for throughput
            cache_size = int(os.getenv("MEMORY_CACHE_SIZE", 500))
            max_sequence_length = int(os.getenv("MAX_SEQUENCE_LENGTH", 4096))
            max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", 1024))
        elif has_cuda:
            # Development CUDA: Balance speed and resources
            cache_size = int(os.getenv("MEMORY_CACHE_SIZE", 200))
            max_sequence_length = int(os.getenv("MAX_SEQUENCE_LENGTH", 2048))
            max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", 512))
        else:
            # Mac/CPU: Conservative settings
            cache_size = int(os.getenv("MEMORY_CACHE_SIZE", 50))
            max_sequence_length = int(os.getenv("MAX_SEQUENCE_LENGTH", 1024))
            max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", 256))
        
        inference_config = InferenceConfig(
            model_path=model_path,
            max_batch_size=max_batch_size,
            cache_size=cache_size,
            quantization_bits=quantization_bits,
            compile_model=compile_model,
            device=device,
            dtype=dtype,
            enable_flash_attention=enable_flash_attention,
            max_sequence_length=max_sequence_length,
            max_new_tokens=max_new_tokens
        )
        
        logger.info(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")
        logger.info(f"Device configuration: CUDA={has_cuda}, MPS={has_mps}, Device={device}")
        logger.info(f"Model settings: Quantization={quantization_bits}bit, Batch={max_batch_size}, Compile={compile_model}")
        logger.info(f"Memory settings: Cache={cache_size}, SeqLen={max_sequence_length}, NewTokens={max_new_tokens}")
        
        app_state.inference_pipeline = OptimizedInferencePipeline(inference_config)
        await app_state.inference_pipeline.initialize()
        
        # Start processing loop
        asyncio.create_task(app_state.inference_pipeline.start_processing())
        logger.info("Inference pipeline initialized")
        
        # Initialize tokenizer
        app_state.tokenizer = CodeAwareTokenizer()
        logger.info("Tokenizer initialized")
        
        # Initialize GitHub collector if token is provided
        github_token = os.getenv("GITHUB_TOKEN")
        if github_token:
            app_state.github_collector = GitHubCollector(github_token)
            logger.info("GitHub collector initialized")
        
        app_state.is_ready = True
        logger.info("API server ready")
        
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down API server...")
    
    if app_state.inference_pipeline:
        await app_state.inference_pipeline.shutdown()
    
    if app_state.cache:
        app_state.cache.clear()
    
    # Close all WebSocket connections
    for ws in app_state.websocket_connections.values():
        await ws.close()
    
    logger.info("API server shutdown complete")

# ====================
# FastAPI Application
# ====================

app = FastAPI(
    title="Automated Code Review API",
    description="AI-powered code review system with real-time inference",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ====================
# Helper Functions
# ====================

def generate_review_id() -> str:
    """Generate unique review ID"""
    return f"review_{uuid.uuid4().hex[:12]}"

def check_app_ready():
    """Dependency to check if app is ready"""
    if not app_state.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is still initializing"
        )
    return True

async def process_code_review(request: CodeReviewRequest, review_id: str) -> CodeReviewResponse:
    """Process a single code review request"""
    try:
        # Check cache first
        cache_key_params = {
            "repo": request.repo,
            "diff": request.diff[:100],  # Use first 100 chars for key
            "language": request.language,
            "temperature": request.temperature
        }
        
        cached_result = await app_state.cache.get_async(request.diff, cache_key_params)
        if cached_result:
            cache_hits.inc()
            return CodeReviewResponse(
                review_id=review_id,
                status="completed",
                comments=json.loads(cached_result) if isinstance(cached_result, str) else cached_result,
                confidence=0.95,
                tokens_used=0,
                processing_time=0.0,
                cached=True
            )
        
        cache_misses.inc()
        
        # Prepare prompt for inference
        prompt = f"""Review the following code diff and provide constructive feedback:

Repository: {request.repo}
Language: {request.language or 'auto-detect'}
Context: {request.context or 'General code review'}

Diff:
{request.diff}

Please provide:
1. Issues found (if any)
2. Suggestions for improvement
3. Security considerations
4. Performance considerations
"""
        
        # Create inference request
        inference_req = InferenceRequest(
            id=review_id,
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=0.95,
            top_k=50,
            stop_sequences=["```", "---", "###"]
        )
        
        # Process with inference pipeline
        with model_inference_time.time():
            inference_resp = await app_state.inference_pipeline.add_request(inference_req)
        
        # Parse response into structured format
        try:
            # Simple parsing - in production, use more sophisticated parsing
            lines = inference_resp.text.split('\n')
            comments = []
            suggestions = []
            
            current_section = None
            for line in lines:
                if 'issue' in line.lower() or 'problem' in line.lower():
                    current_section = 'issues'
                elif 'suggestion' in line.lower() or 'improvement' in line.lower():
                    current_section = 'suggestions'
                elif line.strip() and current_section:
                    if current_section == 'issues':
                        comments.append({"type": "issue", "text": line.strip()})
                    else:
                        suggestions.append({"type": "suggestion", "text": line.strip()})
        except:
            # Fallback to raw response
            comments = [{"type": "general", "text": inference_resp.text}]
            suggestions = []
        
        # Cache the result
        cache_value = json.dumps({"comments": comments, "suggestions": suggestions})
        await app_state.cache.set_async(request.diff, cache_key_params, cache_value)
        
        # Update metrics
        token_usage.inc(inference_resp.tokens_generated)
        
        return CodeReviewResponse(
            review_id=review_id,
            status="completed",
            comments=comments,
            suggestions=suggestions,
            confidence=0.85,
            tokens_used=inference_resp.tokens_generated,
            processing_time=inference_resp.processing_time,
            cached=inference_resp.cached
        )
        
    except Exception as e:
        logger.error(f"Error processing review {review_id}: {e}")
        return CodeReviewResponse(
            review_id=review_id,
            status="failed",
            error=str(e)
        )

async def notify_websocket_clients(review_id: str, result: CodeReviewResponse):
    """Notify WebSocket clients about completed review"""
    message = {
        "type": "review_complete",
        "review_id": review_id,
        "result": result.dict()
    }
    
    disconnected = []
    for client_id, ws in app_state.websocket_connections.items():
        try:
            await ws.send_json(message)
        except:
            disconnected.append(client_id)
    
    # Clean up disconnected clients
    for client_id in disconnected:
        app_state.websocket_connections.pop(client_id, None)
        active_connections.dec()

# ====================
# API Endpoints
# ====================

@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "service": "Automated Code Review API",
        "version": "1.0.0",
        "status": "ready" if app_state.is_ready else "initializing"
    }

@app.get("/health", response_model=HealthCheckResponse, tags=["Monitoring"])
async def health_check():
    """Health check endpoint"""
    uptime = (datetime.now() - app_state.startup_time).total_seconds()
    
    return HealthCheckResponse(
        status="healthy" if app_state.is_ready else "initializing",
        version="1.0.0",
        uptime=uptime,
        model_loaded=app_state.inference_pipeline is not None,
        cache_available=app_state.cache is not None,
        queue_size=len(app_state.pending_reviews),
        active_connections=len(app_state.websocket_connections)
    )

@app.post("/review", response_model=CodeReviewResponse, tags=["Code Review"])
@limiter.limit("10/minute")
async def create_review(
    request: Request,
    review_request: CodeReviewRequest,
    background_tasks: BackgroundTasks,
    _: bool = Depends(check_app_ready)
):
    """
    Create a new code review request
    
    - For 'critical' or 'high' urgency: processed immediately
    - For 'normal' or 'low' urgency: queued for background processing
    """
    request_count.inc()
    
    review_id = generate_review_id()
    
    with request_latency.time():
        if review_request.urgency in ['critical', 'high']:
            # Process immediately
            result = await process_code_review(review_request, review_id)
            
            # Notify WebSocket clients
            await notify_websocket_clients(review_id, result)
            
            return result
        else:
            # Queue for background processing
            app_state.pending_reviews[review_id] = {
                "request": review_request,
                "status": "queued",
                "created_at": datetime.now().isoformat()
            }
            queue_size.set(len(app_state.pending_reviews))
            
            async def background_process():
                try:
                    app_state.pending_reviews[review_id]["status"] = "processing"
                    result = await process_code_review(review_request, review_id)
                    app_state.pending_reviews[review_id] = {
                        "status": "completed",
                        "result": result.dict(),
                        "completed_at": datetime.now().isoformat()
                    }
                    await notify_websocket_clients(review_id, result)
                except Exception as e:
                    app_state.pending_reviews[review_id] = {
                        "status": "failed",
                        "error": str(e),
                        "failed_at": datetime.now().isoformat()
                    }
                finally:
                    queue_size.set(len([r for r in app_state.pending_reviews.values() if r.get("status") == "queued"]))
            
            background_tasks.add_task(background_process)
            
            return CodeReviewResponse(
                review_id=review_id,
                status="queued"
            )

@app.post("/review/batch", response_model=List[CodeReviewResponse], tags=["Code Review"])
@limiter.limit("5/minute")
async def create_batch_review(
    request: Request,
    batch_request: BatchReviewRequest,
    background_tasks: BackgroundTasks,
    _: bool = Depends(check_app_ready)
):
    """
    Create multiple code review requests in a single batch
    
    All requests in the batch are processed with the same urgency level
    """
    batch_id = batch_request.batch_id or f"batch_{uuid.uuid4().hex[:8]}"
    responses = []
    
    for review_req in batch_request.reviews:
        review_id = f"{batch_id}_{generate_review_id()}"
        
        if review_req.urgency in ['critical', 'high']:
            result = await process_code_review(review_req, review_id)
            responses.append(result)
        else:
            # Queue for background processing
            app_state.pending_reviews[review_id] = {
                "request": review_req,
                "status": "queued",
                "batch_id": batch_id,
                "created_at": datetime.now().isoformat()
            }
            
            async def process_batch_item(req, rid):
                try:
                    app_state.pending_reviews[rid]["status"] = "processing"
                    result = await process_code_review(req, rid)
                    app_state.pending_reviews[rid] = {
                        "status": "completed",
                        "result": result.dict(),
                        "completed_at": datetime.now().isoformat()
                    }
                    await notify_websocket_clients(rid, result)
                except Exception as e:
                    app_state.pending_reviews[rid] = {
                        "status": "failed",
                        "error": str(e),
                        "failed_at": datetime.now().isoformat()
                    }
            
            background_tasks.add_task(process_batch_item, review_req, review_id)
            
            responses.append(CodeReviewResponse(
                review_id=review_id,
                status="queued"
            ))
    
    queue_size.set(len([r for r in app_state.pending_reviews.values() if r.get("status") == "queued"]))
    return responses

@app.get("/review/{review_id}", response_model=ReviewStatusResponse, tags=["Code Review"])
async def get_review_status(review_id: str):
    """Check the status of a specific review"""
    if review_id not in app_state.pending_reviews:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Review {review_id} not found"
        )
    
    review_data = app_state.pending_reviews[review_id]
    
    response = ReviewStatusResponse(
        review_id=review_id,
        status=review_data["status"]
    )
    
    if review_data["status"] == "completed":
        response.result = CodeReviewResponse(**review_data["result"])
    elif review_data["status"] == "processing":
        response.progress = 0.5  # Simplified progress
    elif review_data["status"] == "queued":
        # Estimate based on queue position
        queued_before = sum(
            1 for r in app_state.pending_reviews.values()
            if r.get("status") == "queued" and r.get("created_at", "") < review_data.get("created_at", "")
        )
        response.estimated_completion = f"{queued_before * 10} seconds"  # Rough estimate
    
    return response

@app.delete("/review/{review_id}", tags=["Code Review"])
async def cancel_review(review_id: str):
    """Cancel a pending review"""
    if review_id not in app_state.pending_reviews:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Review {review_id} not found"
        )
    
    review_data = app_state.pending_reviews[review_id]
    
    if review_data["status"] in ["completed", "failed"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel review in {review_data['status']} status"
        )
    
    app_state.pending_reviews[review_id]["status"] = "cancelled"
    queue_size.set(len([r for r in app_state.pending_reviews.values() if r.get("status") == "queued"]))
    
    return {"message": f"Review {review_id} cancelled"}

@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint"""
    return StreamingResponse(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

@app.get("/stats", tags=["Monitoring"])
async def get_stats(_: bool = Depends(check_app_ready)):
    """Get detailed statistics about the service"""
    
    cache_stats = app_state.cache.get_stats() if app_state.cache else {}
    pipeline_health = await app_state.inference_pipeline.health_check() if app_state.inference_pipeline else {}
    
    return {
        "uptime_seconds": (datetime.now() - app_state.startup_time).total_seconds(),
        "total_requests": request_count._value.get(),
        "cache_stats": cache_stats,
        "pipeline_health": pipeline_health,
        "pending_reviews": len(app_state.pending_reviews),
        "active_websockets": len(app_state.websocket_connections),
        "queue_breakdown": {
            "queued": sum(1 for r in app_state.pending_reviews.values() if r.get("status") == "queued"),
            "processing": sum(1 for r in app_state.pending_reviews.values() if r.get("status") == "processing"),
            "completed": sum(1 for r in app_state.pending_reviews.values() if r.get("status") == "completed"),
            "failed": sum(1 for r in app_state.pending_reviews.values() if r.get("status") == "failed")
        }
    }

# ====================
# WebSocket Endpoints
# ====================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    client_id = f"ws_{uuid.uuid4().hex[:8]}"
    app_state.websocket_connections[client_id] = websocket
    active_connections.inc()
    
    try:
        await websocket.send_json({
            "type": "connection",
            "client_id": client_id,
            "message": "Connected to code review service"
        })
        
        while True:
            # Keep connection alive and handle client messages
            data = await websocket.receive_text()
            
            # Handle different message types
            try:
                message = json.loads(data)
                
                if message.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                elif message.get("type") == "subscribe":
                    # Could implement subscription to specific review IDs
                    review_id = message.get("review_id")
                    await websocket.send_json({
                        "type": "subscribed",
                        "review_id": review_id
                    })
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Unknown message type: {message.get('type')}"
                    })
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON"
                })
                
    except WebSocketDisconnect:
        app_state.websocket_connections.pop(client_id, None)
        active_connections.dec()
        logger.info(f"Client {client_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        app_state.websocket_connections.pop(client_id, None)
        active_connections.dec()

# ====================
# GitHub Integration Endpoints
# ====================

@app.post("/github/review-pr", tags=["GitHub Integration"])
@limiter.limit("5/minute")
async def review_github_pr(
    request: Request,
    repo_owner: str,
    repo_name: str,
    pr_number: int,
    background_tasks: BackgroundTasks,
    _: bool = Depends(check_app_ready)
):
    """Review a GitHub pull request directly"""
    if not app_state.github_collector:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="GitHub integration not configured. Set GITHUB_TOKEN environment variable."
        )
    
    try:
        # Fetch PR data
        pr_data = app_state.github_collector.get_pr_data(repo_owner, repo_name, pr_number)
        
        # Create review request
        review_request = CodeReviewRequest(
            repo=f"{repo_owner}/{repo_name}",
            pr_number=pr_number,
            diff=pr_data.get("diff", ""),
            urgency="normal",
            context=pr_data.get("description", ""),
            language=pr_data.get("language", "auto-detect")
        )
        
        review_id = generate_review_id()
        
        # Process the review
        async def process_and_post():
            result = await process_code_review(review_request, review_id)
            
            # Post comments back to GitHub if successful
            if result.status == "completed" and result.comments:
                for comment in result.comments:
                    app_state.github_collector.post_pr_comment(
                        repo_owner, repo_name, pr_number,
                        comment.get("text", "")
                    )
        
        background_tasks.add_task(process_and_post)
        
        return {
            "message": f"Review initiated for PR #{pr_number}",
            "review_id": review_id,
            "pr_url": f"https://github.com/{repo_owner}/{repo_name}/pull/{pr_number}"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process GitHub PR: {str(e)}"
        )

# ====================
# Main Entry Point
# ====================

if __name__ == "__main__":
    # Configuration from environment variables
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    workers = int(os.getenv("API_WORKERS", 1))
    
    # Run the application
    if workers > 1:
        # For production with multiple workers
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            workers=workers,
            log_level="info",
            access_log=True
        )
    else:
        # For development or single worker
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info",
            access_log=True
        )