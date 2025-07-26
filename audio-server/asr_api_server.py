#!/usr/bin/env python3
"""
ASR API Server

Provides a REST API interface for the optimized ASR system with Kimi task classification.
Other Python scripts can call this API to get task results in real-time.
"""

import asyncio
import threading
import queue
import time
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import uvicorn
import json

# Import our ASR components
from optimized_asr import (
    AudioConfig, VADConfig, ASRConfig, KimiConfig,
    OptimizedRealTimeASR, KimiTaskClassifier
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TaskResult:
    """Task result from Kimi classification"""
    task: str
    original_text: str
    timestamp: float
    confidence: Optional[float] = None

class ASRAPIService:
    """ASR API Service that manages the ASR system and task queue"""
    
    def __init__(self):
        self.task_queue = queue.Queue()
        self.asr_system = None
        self.is_running = False
        self._setup_asr_system()
    
    def _setup_asr_system(self):
        """Initialize the ASR system with configurations"""
        try:
            # Audio configuration
            audio_config = AudioConfig(
                sample_rate=16000,
                chunk_size=480,
                channels=1
            )
            
            # VAD configuration
            vad_config = VADConfig(
                aggressiveness=0,
                min_speech_duration=0.5,
                max_speech_duration=5.0,
                silence_threshold=1.0
            )
            
            # ASR configuration
            asr_config = ASRConfig(
                model_name="base",  # Use tiny model for faster API response
                language="zh",
                fp16=True,
                beam_size=1,
                temperature=0.0
            )
            
            # Kimi configuration
            kimi_config = KimiConfig(
                api_key="sk-b4pdM1xHqOdY16GIWZxhKIFoR7zdJ3ob3QHo8MU88LY5ONFY",  # Replace with actual API key
                base_url="https://api.moonshot.cn/v1",
                model="moonshot-v1-8k",
                temperature=0.3,
                max_chinese_words=15
            )
            
            # Initialize ASR system
            self.asr_system = OptimizedRealTimeASR(
                audio_config=audio_config,
                vad_config=vad_config,
                asr_config=asr_config,
                kimi_config=kimi_config,
                speech_callback=self._speech_callback,
                task_callback=self._task_callback
            )
            
            logger.info("ASR system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ASR system: {e}")
            raise
    
    def _speech_callback(self, text: str, confidence: float):
        """Callback for speech recognition results"""
        logger.debug(f"Speech recognized: {text} (confidence: {confidence})")
    
    def _task_callback(self, text: str, task: str):
        """Callback for task classification results"""
        logger.info(f"Task detected: {task} from text: {text}")
        
        # Create task result
        task_result = TaskResult(
            task=task,
            original_text=text,
            timestamp=time.time()
        )
        
        # Add to queue for API consumers
        try:
            self.task_queue.put_nowait(task_result)
        except queue.Full:
            logger.warning("Task queue is full, dropping oldest task")
            try:
                self.task_queue.get_nowait()  # Remove oldest
                self.task_queue.put_nowait(task_result)  # Add new
            except queue.Empty:
                pass
    
    def start(self):
        """Start the ASR system"""
        if not self.is_running:
            try:
                self.asr_system.start()
                self.is_running = True
                logger.info("ASR API service started")
            except Exception as e:
                logger.error(f"Failed to start ASR system: {e}")
                raise
    
    def stop(self):
        """Stop the ASR system"""
        if self.is_running:
            try:
                self.asr_system.stop()
                self.is_running = False
                logger.info("ASR API service stopped")
            except Exception as e:
                logger.error(f"Failed to stop ASR system: {e}")
    
    def get_task(self, timeout: float = 1.0) -> Optional[TaskResult]:
        """Get the next task from the queue"""
        try:
            return self.task_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get ASR system statistics"""
        if self.asr_system:
            return self.asr_system.get_stats()
        return {}

# Global ASR service instance
asr_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage ASR service lifecycle"""
    global asr_service
    
    # Startup
    logger.info("Starting ASR API server...")
    asr_service = ASRAPIService()
    asr_service.start()
    
    yield
    
    # Shutdown
    logger.info("Shutting down ASR API server...")
    if asr_service:
        asr_service.stop()

# Create FastAPI app
app = FastAPI(
    title="ASR Task Classification API",
    description="Real-time speech recognition with Kimi task classification",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "ASR Task Classification API",
        "status": "running" if asr_service and asr_service.is_running else "stopped",
        "endpoints": {
            "/task": "Get next task (blocking)",
            "/task/stream": "Stream tasks (Server-Sent Events)",
            "/stats": "Get system statistics",
            "/health": "Health check"
        }
    }

@app.get("/task")
async def get_task(timeout: float = 30.0):
    """Get the next task result (blocking until task is available or timeout)"""
    if not asr_service or not asr_service.is_running:
        raise HTTPException(status_code=503, detail="ASR service not running")
    
    # Run in thread pool to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    task_result = await loop.run_in_executor(None, asr_service.get_task, timeout)
    
    if task_result is None:
        raise HTTPException(status_code=408, detail="Timeout waiting for task")
    
    return {
        "task": task_result.task,
        "original_text": task_result.original_text,
        "timestamp": task_result.timestamp,
        "confidence": task_result.confidence
    }

@app.get("/task/stream")
async def stream_tasks():
    """Stream task results using Server-Sent Events"""
    if not asr_service or not asr_service.is_running:
        raise HTTPException(status_code=503, detail="ASR service not running")
    
    async def generate_tasks():
        """Generate task events"""
        while True:
            try:
                # Get task with short timeout to allow for graceful shutdown
                loop = asyncio.get_event_loop()
                task_result = await loop.run_in_executor(None, asr_service.get_task, 1.0)
                
                if task_result:
                    data = {
                        "task": task_result.task,
                        "original_text": task_result.original_text,
                        "timestamp": task_result.timestamp,
                        "confidence": task_result.confidence
                    }
                    yield f"data: {json.dumps(data)}\n\n"
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in task stream: {e}")
                yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"
                break
    
    return StreamingResponse(
        generate_tasks(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

@app.get("/stats")
async def get_stats():
    """Get ASR system statistics"""
    if not asr_service:
        raise HTTPException(status_code=503, detail="ASR service not available")
    
    stats = asr_service.get_stats()
    stats["api_status"] = "running" if asr_service.is_running else "stopped"
    stats["queue_size"] = asr_service.task_queue.qsize()
    
    return stats

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if asr_service and asr_service.is_running else "unhealthy",
        "timestamp": time.time()
    }

@app.post("/start")
async def start_service():
    """Start the ASR service"""
    if not asr_service:
        raise HTTPException(status_code=503, detail="ASR service not available")
    
    if asr_service.is_running:
        return {"message": "Service already running"}
    
    try:
        asr_service.start()
        return {"message": "Service started successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start service: {e}")

@app.post("/stop")
async def stop_service():
    """Stop the ASR service"""
    if not asr_service:
        raise HTTPException(status_code=503, detail="ASR service not available")
    
    if not asr_service.is_running:
        return {"message": "Service already stopped"}
    
    try:
        asr_service.stop()
        return {"message": "Service stopped successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop service: {e}")

def main():
    """Main function to run the API server"""
    logger.info("Starting ASR Task Classification API Server")
    
    # Run the server
    uvicorn.run(
        "asr_api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    main()