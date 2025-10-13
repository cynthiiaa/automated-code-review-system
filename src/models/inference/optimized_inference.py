import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import List, Dict, Optional, Any, Tuple
import asyncio
from dataclasses import dataclass, field
import logging
from collections import OrderedDict
import time
import psutil
import GPUtil
from contextlib import asynccontextmanager
import hashlib
import json
from pathlib import Path
import os
from concurrent.futures import ThreadPoolExecutor
import warnings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class InferenceConfig:
    """Configuration for inference pipeline"""
    model_path: str
    max_batch_size: int = 8
    min_batch_size: int = 1
    batch_timeout: float = 0.1
    max_sequence_length: int = 2048
    max_new_tokens: int = 512
    default_temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    cache_size: int = 1000
    cache_ttl: int = 3600
    device: str = "auto"
    dtype: torch.dtype = torch.float16
    quantization_bits: int = 4
    compile_model: bool = True
    num_workers: int = 2
    memory_threshold: float = 0.9
    enable_flash_attention: bool = True

@dataclass
class InferenceRequest:
    id: str
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    stop_sequences: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    priority: int = 0
    callback: Optional[asyncio.Future] = None

@dataclass
class InferenceResponse:
    id: str
    text: str
    tokens_generated: int
    processing_time: float
    model_name: str
    cached: bool = False

class CacheManager:
    """LRU cache for inference results"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.ttl = ttl
        self.hits = 0
        self.misses = 0
    
    def _get_cache_key(self, prompt: str, params: Dict[str, Any]) -> str:
        """Generate cache key from prompt and parameters"""
        cache_data = {
            "prompt": prompt,
            "params": params
        }
        return hashlib.md5(json.dumps(cache_data, sort_keys=True).encode()).hexdigest()
    
    def get(self, prompt: str, params: Dict[str, Any]) -> Optional[str]:
        """Get cached result if available and not expired"""
        key = self._get_cache_key(prompt, params)
        if key in self.cache:
            entry, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                self.cache.move_to_end(key)
                self.hits += 1
                return entry
            else:
                del self.cache[key]
        self.misses += 1
        return None
    
    def put(self, prompt: str, params: Dict[str, Any], result: str):
        """Add result to cache"""
        key = self._get_cache_key(prompt, params)
        self.cache[key] = (result, time.time())
        self.cache.move_to_end(key)
        
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
    
    def clear(self):
        """Clear the cache"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

class MemoryManager:
    """Monitor and manage GPU/CPU memory"""
    
    def __init__(self, threshold: float = 0.9):
        self.threshold = threshold
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else None
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        stats = {
            "cpu_percent": psutil.virtual_memory().percent / 100,
            "cpu_available_gb": psutil.virtual_memory().available / (1024**3)
        }
        
        if self.device is not None:
            gpu = GPUtil.getGPUs()[self.device]
            stats.update({
                "gpu_percent": gpu.memoryUtil,
                "gpu_available_gb": gpu.memoryFree / 1024,
                "gpu_temperature": gpu.temperature
            })
        
        return stats
    
    def is_memory_available(self, estimated_size_gb: float = 1.0) -> bool:
        """Check if enough memory is available for operation"""
        stats = self.get_memory_stats()
        
        if self.device is not None:
            return stats["gpu_percent"] < self.threshold and \
                   stats["gpu_available_gb"] > estimated_size_gb
        else:
            return stats["cpu_percent"] < self.threshold and \
                   stats["cpu_available_gb"] > estimated_size_gb
    
    def clear_cache(self):
        """Clear GPU cache to free memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

class OptimizedInferencePipeline:
    """Production-ready inference pipeline with batching and optimization"""
    
    def __init__(self, config: Optional[InferenceConfig] = None):
        self.config = config or InferenceConfig(model_path="codellama/CodeLlama-7b-hf")
        self.model = None
        self.tokenizer = None
        self.request_queue = asyncio.PriorityQueue()
        self.cache = CacheManager(self.config.cache_size, self.config.cache_ttl)
        self.memory_manager = MemoryManager(self.config.memory_threshold)
        self.executor = ThreadPoolExecutor(max_workers=self.config.num_workers)
        self._shutdown = False
        self._initialized = False
        self.stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "avg_latency": 0,
            "errors": 0
        }
    
    async def initialize(self):
        """Initialize model and tokenizer asynchronously"""
        if self._initialized:
            return
        
        logger.info(f"Initializing inference pipeline with model: {self.config.model_path}")
        
        try:
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._load_model_and_tokenizer
            )
            self._initialized = True
            logger.info("Inference pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            raise
    
    def _load_model_and_tokenizer(self):
        """Load model and tokenizer with optimizations"""
        quantization_config = None
        if self.config.quantization_bits in [4, 8]:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=(self.config.quantization_bits == 4),
                load_in_8bit=(self.config.quantization_bits == 8),
                bnb_4bit_compute_dtype=self.config.dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path,
                use_fast=True,
                padding_side="left",
                truncation_side="left"
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            model_kwargs = {
                "torch_dtype": self.config.dtype,
                "device_map": self.config.device,
                "trust_remote_code": True
            }
            
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
            
            if self.config.enable_flash_attention:
                model_kwargs["attn_implementation"] = "flash_attention_2"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                **model_kwargs
            )
            
            if self.config.compile_model and hasattr(torch, 'compile'):
                logger.info("Compiling model with torch.compile...")
                self.model = torch.compile(
                    self.model,
                    mode="reduce-overhead",
                    fullgraph=True
                )
            
            self.model.eval()
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    async def add_request(self, request: InferenceRequest) -> InferenceResponse:
        """Add request to queue and wait for response"""
        if not self._initialized:
            await self.initialize()
        
        params = {
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k
        }
        
        cached_result = self.cache.get(request.prompt, params)
        if cached_result:
            logger.debug(f"Cache hit for request {request.id}")
            return InferenceResponse(
                id=request.id,
                text=cached_result,
                tokens_generated=len(self.tokenizer.encode(cached_result)),
                processing_time=0.0,
                model_name=self.config.model_path,
                cached=True
            )
        
        future = asyncio.Future()
        request.callback = future
        
        priority = -request.priority
        await self.request_queue.put((priority, request.timestamp, request))
        
        try:
            response = await asyncio.wait_for(future, timeout=60.0)
            return response
        except asyncio.TimeoutError:
            logger.error(f"Request {request.id} timed out")
            self.stats["errors"] += 1
            raise
    
    async def start_processing(self):
        """Start the batch processing loop"""
        logger.info("Starting batch processing loop")
        
        while not self._shutdown:
            try:
                await self._process_batch_loop()
            except Exception as e:
                logger.error(f"Error in batch processing: {e}")
                await asyncio.sleep(1)
    
    async def _process_batch_loop(self):
        """Main batch processing loop"""
        batch = []
        batch_start_time = time.time()
        
        while len(batch) < self.config.max_batch_size:
            try:
                timeout = self.config.batch_timeout if batch else None
                _, _, request = await asyncio.wait_for(
                    self.request_queue.get(),
                    timeout=timeout
                )
                batch.append(request)
                
                if not self.memory_manager.is_memory_available():
                    logger.warning("Memory threshold reached, processing current batch")
                    break
                    
            except asyncio.TimeoutError:
                if batch:
                    break
                continue
        
        if batch:
            batch_size = len(batch)
            logger.debug(f"Processing batch of {batch_size} requests")
            
            try:
                await self._process_batch(batch)
                
                latency = time.time() - batch_start_time
                self.stats["total_requests"] += batch_size
                self.stats["avg_latency"] = (
                    (self.stats["avg_latency"] * (self.stats["total_requests"] - batch_size) + latency) /
                    self.stats["total_requests"]
                )
                
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                self.stats["errors"] += 1
                
                for request in batch:
                    if request.callback and not request.callback.done():
                        request.callback.set_exception(e)
    
    async def _process_batch(self, batch: List[InferenceRequest]):
        """Process a batch of requests efficiently"""
        start_time = time.time()
        
        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                self.executor,
                self._run_inference,
                batch
            )
            
            processing_time = time.time() - start_time
            
            for request, generated_text in zip(batch, results):
                response = InferenceResponse(
                    id=request.id,
                    text=generated_text,
                    tokens_generated=len(self.tokenizer.encode(generated_text)),
                    processing_time=processing_time / len(batch),
                    model_name=self.config.model_path,
                    cached=False
                )
                
                params = {
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "top_k": request.top_k
                }
                self.cache.put(request.prompt, params, generated_text)
                
                if request.callback and not request.callback.done():
                    request.callback.set_result(response)
                
                self.stats["total_tokens"] += response.tokens_generated
                
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise
    
    def _run_inference(self, batch: List[InferenceRequest]) -> List[str]:
        """Run inference on CPU/GPU (blocking)"""
        inputs = self._prepare_batch_inputs(batch)
        
        with torch.inference_mode():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=min(
                        max(r.max_tokens for r in batch),
                        self.config.max_new_tokens
                    ),
                    temperature=batch[0].temperature if batch else self.config.default_temperature,
                    top_p=batch[0].top_p if batch else self.config.top_p,
                    top_k=batch[0].top_k if batch else self.config.top_k,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                    return_dict_in_generate=False
                )
        
        results = self._decode_outputs(outputs, batch, inputs["input_ids"])
        return results
    
    def _prepare_batch_inputs(self, batch: List[InferenceRequest]) -> Dict[str, torch.Tensor]:
        """Prepare and tokenize batch inputs"""
        prompts = [req.prompt for req in batch]
        
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_sequence_length,
            return_attention_mask=True
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        return inputs
    
    def _decode_outputs(
        self,
        outputs: torch.Tensor,
        batch: List[InferenceRequest],
        input_ids: torch.Tensor
    ) -> List[str]:
        """Decode model outputs to text"""
        results = []
        
        for i, request in enumerate(batch):
            output_ids = outputs[i]
            
            input_length = input_ids[i].shape[0]
            generated_ids = output_ids[input_length:]
            
            generated_text = self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            for stop_seq in request.stop_sequences:
                if stop_seq in generated_text:
                    generated_text = generated_text.split(stop_seq)[0]
                    break
            
            results.append(generated_text.strip())
        
        return results
    
    async def health_check(self) -> Dict[str, Any]:
        """Check pipeline health status"""
        memory_stats = self.memory_manager.get_memory_stats()
        
        return {
            "status": "healthy" if self._initialized else "initializing",
            "model": self.config.model_path,
            "queue_size": self.request_queue.qsize(),
            "cache_hit_rate": self.cache.hit_rate,
            "memory": memory_stats,
            "stats": self.stats,
            "uptime": time.time() if self._initialized else 0
        }
    
    async def shutdown(self):
        """Gracefully shutdown the pipeline"""
        logger.info("Shutting down inference pipeline...")
        self._shutdown = True
        
        while not self.request_queue.empty():
            try:
                _, _, request = self.request_queue.get_nowait()
                if request.callback and not request.callback.done():
                    request.callback.cancel()
            except asyncio.QueueEmpty:
                break
        
        self.executor.shutdown(wait=True)
        self.cache.clear()
        self.memory_manager.clear_cache()
        
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        logger.info("Inference pipeline shutdown complete")

    @asynccontextmanager
    async def session(self):
        """Context manager for pipeline lifecycle"""
        await self.initialize()
        processing_task = asyncio.create_task(self.start_processing())
        
        try:
            yield self
        finally:
            await self.shutdown()
            processing_task.cancel()
            try:
                await processing_task
            except asyncio.CancelledError:
                pass


async def main():
    """Example usage of the inference pipeline"""
    config = InferenceConfig(
        model_path="codellama/CodeLlama-7b-hf",
        max_batch_size=4,
        cache_size=100,
        quantization_bits=4
    )
    
    async with OptimizedInferencePipeline(config).session() as pipeline:
        request = InferenceRequest(
            id="test-001",
            prompt="def fibonacci(n):",
            max_tokens=150,
            temperature=0.8,
            stop_sequences=["def ", "class "]
        )
        
        response = await pipeline.add_request(request)
        print(f"Generated: {response.text}")
        print(f"Tokens: {response.tokens_generated}, Time: {response.processing_time:.2f}s")
        
        health = await pipeline.health_check()
        print(f"Pipeline health: {json.dumps(health, indent=2)}")


if __name__ == "__main__":
    asyncio.run(main())