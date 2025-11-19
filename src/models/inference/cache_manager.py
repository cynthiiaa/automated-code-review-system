import hashlib
import json
import logging
import time
import asyncio
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from collections import OrderedDict
from contextlib import asynccontextmanager

try:
    import redis
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None
    aioredis = None

import pickle

logger = logging.getLogger(__name__)

@dataclass
class CacheConfig:
    """Configuration for cache system"""
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    cache_ttl: int = 3600  # 1 hour default
    max_memory_items: int = 1000
    enable_redis: bool = True
    enable_memory: bool = True
    compression: bool = True
    prefix: str = "inference:"

@dataclass
class CacheStats:
    """Cache statistics tracking"""
    hits: int = 0
    misses: int = 0
    errors: int = 0
    redis_hits: int = 0
    memory_hits: int = 0
    total_requests: int = 0
    avg_retrieval_time: float = 0.0
    avg_storage_time: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        return self.hits / self.total_requests if self.total_requests > 0 else 0.0
    
    @property
    def redis_hit_rate(self) -> float:
        return self.redis_hits / self.total_requests if self.total_requests > 0 else 0.0
    
    @property
    def memory_hit_rate(self) -> float:
        return self.memory_hits / self.total_requests if self.total_requests > 0 else 0.0

class HybridInferenceCache:
    """Hybrid caching system with Redis persistence and in-memory LRU cache"""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self.stats = CacheStats()
        
        # In-memory LRU cache
        self.memory_cache = OrderedDict() if self.config.enable_memory else None
        
        # Redis clients (sync and async)
        self.redis_client = None
        self.redis_async_client = None
        self._redis_initialized = False
        
        if self.config.enable_redis and REDIS_AVAILABLE:
            self._init_redis_sync()
    
    def _init_redis_sync(self):
        """Initialize synchronous Redis client with error handling"""
        try:
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                password=self.config.redis_password,
                decode_responses=False,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            # Test connection
            self.redis_client.ping()
            self._redis_initialized = True
            logger.info(f"Redis cache initialized at {self.config.redis_host}:{self.config.redis_port}")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}. Falling back to memory-only cache.")
            self.redis_client = None
            self._redis_initialized = False
    
    async def _init_redis_async(self):
        """Initialize asynchronous Redis client"""
        if not REDIS_AVAILABLE or not self.config.enable_redis:
            return
        
        try:
            self.redis_async_client = await aioredis.from_url(
                f"redis://{self.config.redis_host}:{self.config.redis_port}/{self.config.redis_db}",
                password=self.config.redis_password,
                decode_responses=False,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            await self.redis_async_client.ping()
            logger.info("Async Redis client initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize async Redis: {e}")
            self.redis_async_client = None
    
    def get_cache_key(self, prompt: str, params: Dict[str, Any]) -> str:
        """Generate deterministic cache key"""
        # Normalize parameters for consistent key generation
        normalized_params = json.dumps(params, sort_keys=True)
        cache_input = f"{prompt}||{normalized_params}"
        hash_key = hashlib.sha256(cache_input.encode()).hexdigest()
        return f"{self.config.prefix}{hash_key}"
    
    def _serialize(self, data: Any) -> bytes:
        """Serialize data for storage"""
        serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        if self.config.compression:
            import zlib
            return zlib.compress(serialized)
        return serialized
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize data from storage"""
        if self.config.compression:
            import zlib
            data = zlib.decompress(data)
        return pickle.loads(data)
    
    def get(self, prompt: str, params: Dict[str, Any]) -> Optional[str]:
        """Synchronous cache retrieval with fallback chain"""
        start_time = time.time()
        self.stats.total_requests += 1
        
        key = self.get_cache_key(prompt, params)
        
        # Try memory cache first
        if self.memory_cache is not None:
            if key in self.memory_cache:
                # Move to end (LRU)
                self.memory_cache.move_to_end(key)
                value, timestamp = self.memory_cache[key]
                
                # Check TTL
                if time.time() - timestamp < self.config.cache_ttl:
                    self.stats.hits += 1
                    self.stats.memory_hits += 1
                    self._update_retrieval_time(start_time)
                    return value
                else:
                    # Expired
                    del self.memory_cache[key]
        
        # Try Redis cache
        if self._redis_initialized and self.redis_client:
            try:
                cached_data = self.redis_client.get(key)
                if cached_data:
                    value = self._deserialize(cached_data)
                    self.stats.hits += 1
                    self.stats.redis_hits += 1
                    
                    # Populate memory cache
                    if self.memory_cache is not None:
                        self._add_to_memory_cache(key, value)
                    
                    self._update_retrieval_time(start_time)
                    return value
            except Exception as e:
                logger.error(f"Redis get error: {e}")
                self.stats.errors += 1
        
        self.stats.misses += 1
        self._update_retrieval_time(start_time)
        return None
    
    async def get_async(self, prompt: str, params: Dict[str, Any]) -> Optional[str]:
        """Asynchronous cache retrieval"""
        start_time = time.time()
        self.stats.total_requests += 1
        
        key = self.get_cache_key(prompt, params)
        
        # Try memory cache first (synchronous)
        if self.memory_cache is not None:
            if key in self.memory_cache:
                self.memory_cache.move_to_end(key)
                value, timestamp = self.memory_cache[key]
                
                if time.time() - timestamp < self.config.cache_ttl:
                    self.stats.hits += 1
                    self.stats.memory_hits += 1
                    self._update_retrieval_time(start_time)
                    return value
                else:
                    del self.memory_cache[key]
        
        # Try Redis cache (async)
        if self.redis_async_client:
            try:
                cached_data = await self.redis_async_client.get(key)
                if cached_data:
                    value = self._deserialize(cached_data)
                    self.stats.hits += 1
                    self.stats.redis_hits += 1
                    
                    # Populate memory cache
                    if self.memory_cache is not None:
                        self._add_to_memory_cache(key, value)
                    
                    self._update_retrieval_time(start_time)
                    return value
            except Exception as e:
                logger.error(f"Async Redis get error: {e}")
                self.stats.errors += 1
        
        self.stats.misses += 1
        self._update_retrieval_time(start_time)
        return None
    
    def set(self, prompt: str, params: Dict[str, Any], result: str) -> bool:
        """Synchronous cache storage"""
        start_time = time.time()
        key = self.get_cache_key(prompt, params)
        success = False
        
        # Store in memory cache
        if self.memory_cache is not None:
            self._add_to_memory_cache(key, result)
            success = True
        
        # Store in Redis
        if self._redis_initialized and self.redis_client:
            try:
                serialized = self._serialize(result)
                self.redis_client.setex(
                    key,
                    self.config.cache_ttl,
                    serialized
                )
                success = True
            except Exception as e:
                logger.error(f"Redis set error: {e}")
                self.stats.errors += 1
        
        self._update_storage_time(start_time)
        return success
    
    async def set_async(self, prompt: str, params: Dict[str, Any], result: str) -> bool:
        """Asynchronous cache storage"""
        start_time = time.time()
        key = self.get_cache_key(prompt, params)
        success = False
        
        # Store in memory cache (synchronous)
        if self.memory_cache is not None:
            self._add_to_memory_cache(key, result)
            success = True
        
        # Store in Redis (async)
        if self.redis_async_client:
            try:
                serialized = self._serialize(result)
                await self.redis_async_client.setex(
                    key,
                    self.config.cache_ttl,
                    serialized
                )
                success = True
            except Exception as e:
                logger.error(f"Async Redis set error: {e}")
                self.stats.errors += 1
        
        self._update_storage_time(start_time)
        return success
    
    def _add_to_memory_cache(self, key: str, value: str):
        """Add item to memory cache with LRU eviction"""
        if self.memory_cache is None:
            return
        
        self.memory_cache[key] = (value, time.time())
        self.memory_cache.move_to_end(key)
        
        # Evict oldest if over limit
        if len(self.memory_cache) > self.config.max_memory_items:
            self.memory_cache.popitem(last=False)
    
    def _update_retrieval_time(self, start_time: float):
        """Update average retrieval time statistics"""
        elapsed = time.time() - start_time
        n = self.stats.total_requests
        self.stats.avg_retrieval_time = (
            (self.stats.avg_retrieval_time * (n - 1) + elapsed) / n
        )
    
    def _update_storage_time(self, start_time: float):
        """Update average storage time statistics"""
        elapsed = time.time() - start_time
        # Simple exponential moving average
        alpha = 0.1
        self.stats.avg_storage_time = (
            alpha * elapsed + (1 - alpha) * self.stats.avg_storage_time
        )
    
    def clear(self):
        """Clear all caches"""
        if self.memory_cache:
            self.memory_cache.clear()
        
        if self._redis_initialized and self.redis_client:
            try:
                # Clear all keys with our prefix
                pattern = f"{self.config.prefix}*"
                cursor = 0
                while True:
                    cursor, keys = self.redis_client.scan(
                        cursor, match=pattern, count=100
                    )
                    if keys:
                        self.redis_client.delete(*keys)
                    if cursor == 0:
                        break
            except Exception as e:
                logger.error(f"Error clearing Redis cache: {e}")
    
    async def clear_async(self):
        """Asynchronously clear all caches"""
        if self.memory_cache:
            self.memory_cache.clear()
        
        if self.redis_async_client:
            try:
                pattern = f"{self.config.prefix}*"
                cursor = 0
                while True:
                    cursor, keys = await self.redis_async_client.scan(
                        cursor, match=pattern, count=100
                    )
                    if keys:
                        await self.redis_async_client.delete(*keys)
                    if cursor == 0:
                        break
            except Exception as e:
                logger.error(f"Error clearing async Redis cache: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "hit_rate": f"{self.stats.hit_rate:.2%}",
            "total_requests": self.stats.total_requests,
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "errors": self.stats.errors,
            "redis_hits": self.stats.redis_hits,
            "memory_hits": self.stats.memory_hits,
            "memory_cache_size": len(self.memory_cache) if self.memory_cache else 0,
            "avg_retrieval_ms": f"{self.stats.avg_retrieval_time * 1000:.2f}",
            "avg_storage_ms": f"{self.stats.avg_storage_time * 1000:.2f}",
            "redis_connected": self._redis_initialized,
        }
    
    def warm_cache(self, items: Dict[Tuple[str, Dict], str]):
        """Pre-populate cache with known prompt/response pairs"""
        for (prompt, params), result in items.items():
            self.set(prompt, params, result)
    
    async def warm_cache_async(self, items: Dict[Tuple[str, Dict], str]):
        """Asynchronously pre-populate cache"""
        tasks = [
            self.set_async(prompt, params, result)
            for (prompt, params), result in items.items()
        ]
        await asyncio.gather(*tasks)
    
    @asynccontextmanager
    async def session(self):
        """Async context manager for cache lifecycle"""
        await self._init_redis_async()
        try:
            yield self
        finally:
            if self.redis_async_client:
                await self.redis_async_client.close()


# Backward compatibility
class InferenceCache(HybridInferenceCache):
    """Backward compatible interface"""
    
    def __init__(self, redis_host: str = "localhost"):
        config = CacheConfig(redis_host=redis_host)
        super().__init__(config)