# Configuration Guide

This directory contains environment-specific configuration files for the Automated Code Review System.

## Quick Start

### For Mac Development (Your Current Setup)
```bash
# Copy development config
cp config/development.env .env

# Run the server
python src/serving/api/main.py
```

### For Production (CUDA Server)
```bash
# Copy production config
cp config/production.env .env

# Edit with your specific values
nano .env

# Run with multiple workers
ENVIRONMENT=production python src/serving/api/main.py
```

## Environment Configurations

### Development Environment (`development.env`)
**Optimized for Mac laptops and local testing:**
- ✅ **No quantization** (avoids bitsandbytes CUDA requirement)
- ✅ **MPS support** for Apple Silicon Macs
- ✅ **CPU fallback** for Intel Macs
- ✅ **Conservative memory usage** (50 cache items, 1024 seq length)
- ✅ **Memory-only cache** (no Redis required)
- ✅ **Single worker** for debugging

**Expected behavior on Mac:**
- Model loads without CUDA errors
- Uses CPU (Apple Silicon MPS has memory allocation issues)
- Slower inference but full CodeLlama-7B quality
- ~15-20GB memory usage (requires 64GB system RAM for comfortable usage)

### Production Environment (`production.env`)
**Optimized for CUDA GPU servers:**
- 🚀 **4-bit quantization** for memory efficiency
- 🚀 **Flash attention** for speed
- 🚀 **Torch compilation** for optimization
- 🚀 **Large batch sizes** (8 requests)
- 🚀 **Redis caching** for persistence
- 🚀 **Multiple workers** for throughput

**Expected behavior on CUDA server:**
- ~4-6GB VRAM usage (with quantization)
- 5-10x faster inference than CPU
- Handles 50+ concurrent requests
- Sub-second response times

## Key Configuration Variables

### Model Settings
| Variable | Development | Production | Description |
|----------|-------------|------------|-------------|
| `QUANTIZATION_BITS` | `0` | `4` | Quantization level (0=disabled) |
| `MAX_BATCH_SIZE` | `2` | `8` | Requests processed together |
| `COMPILE_MODEL` | `false` | `true` | Enable torch.compile |
| `MAX_SEQUENCE_LENGTH` | `1024` | `4096` | Max input tokens |
| `MAX_NEW_TOKENS` | `256` | `1024` | Max generated tokens |

### Cache Settings
| Variable | Development | Production | Description |
|----------|-------------|------------|-------------|
| `ENABLE_REDIS` | `false` | `true` | Use Redis for persistence |
| `MEMORY_CACHE_SIZE` | `50` | `500` | In-memory cache items |
| `CACHE_TTL` | `1800` | `7200` | Cache expiration (seconds) |

### API Settings
| Variable | Development | Production | Description |
|----------|-------------|------------|-------------|
| `API_WORKERS` | `1` | `4` | Uvicorn worker processes |
| `API_HOST` | `127.0.0.1` | `0.0.0.0` | Bind address |
| `CORS_ORIGINS` | `*` | `domain.com` | Allowed origins |

## Device-Specific Behavior

### Mac (Apple Silicon - 64GB RAM)
```bash
# Detected automatically
Device: CPU (MPS avoided for memory allocation)
Quantization: Disabled
Memory: ~15-20GB
Speed: Medium-slow (30-60 seconds per review)
```

### Mac (Intel or lower RAM)
```bash
# Detected automatically  
Device: CPU
Quantization: Disabled
Memory: ~2-4GB (smaller models recommended)
Speed: Slow
```

### Linux (CUDA)
```bash
# Detected automatically
Device: CUDA
Quantization: 4-bit (if ENVIRONMENT=production)
Memory: ~4-6GB VRAM
Speed: Fast
```

## Testing Your Setup

### 1. Check Configuration
```bash
# View detected settings (without loading model)
python -c "
import os
os.environ['MODEL_PATH'] = 'microsoft/DialoGPT-medium'  # Small test model
exec(open('src/serving/api/main.py').read())
"
```

### 2. Test with Mock Server
```bash
# Fast testing without model loading
python src/serving/api/mock_main.py
curl http://localhost:8001/health
```

### 3. Test with Real CodeLlama
```bash
# Full test with actual model
ENVIRONMENT=development python src/serving/api/main.py
# Wait for model to load (2-5 minutes on Mac)
curl http://localhost:8000/health
```

## Troubleshooting

### Mac Issues
- **"No module named torch"**: Install with `pip install torch`
- **CUDA error**: Make sure `QUANTIZATION_BITS=0` in development
- **Memory error**: Reduce `MAX_SEQUENCE_LENGTH` to 512
- **Slow loading**: Use smaller model like `microsoft/DialoGPT-medium` for testing

### Production Issues
- **OOM (Out of Memory)**: Reduce `MAX_BATCH_SIZE` or enable quantization
- **Slow startup**: Use model caching or pre-loaded containers
- **Redis connection**: Check `REDIS_HOST` and networking
- **Rate limiting**: Adjust worker count and rate limits

## Performance Expectations

### Mac Development (64GB RAM)
```
Model Load Time: 2-5 minutes
First Request: 30-60 seconds  
Cached Request: <1 second
Throughput: 1-2 requests/minute
Memory Usage: 15-20GB
```

### Production CUDA
```
Model Load Time: 30-60 seconds
First Request: 1-3 seconds
Cached Request: <100ms
Throughput: 30-60 requests/minute
VRAM Usage: 4-6GB
```

## Deployment Scripts

### Development
```bash
#!/bin/bash
# dev-start.sh
cp config/development.env .env
export PYTHONPATH=$PWD
python src/serving/api/main.py
```

### Production
```bash
#!/bin/bash
# prod-start.sh
cp config/production.env .env
export ENVIRONMENT=production
export PYTHONPATH=$PWD
uvicorn src.serving.api.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker
```