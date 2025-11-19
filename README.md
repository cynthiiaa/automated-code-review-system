# Automated Code Review System

I'm building this to understand how LLMs work with code by creating a system that reviews pull requests automatically. Still figuring things out as I go.

## What I have so far

**Data Collection**: `github_collector.py` pulls PR data from repos - diffs, comments, file changes. It's how I'm getting training data.

**Tokenization**: `code_aware_tokenizer.py` handles the tokenization challenges with code. Compares how different tokenizers (CodeLlama vs GPT-4) handle code structures, whitespace, and diffs. Also does smart truncation to fit context windows.

**Model Testing**: `model_evaluator.py` benchmarks different models (CodeLlama, CodeBERT, CodeT5, StarCoder) on latency, throughput, and quality. Still working on the evaluation metrics.

**Caching System**: `cache_manager.py` provides hybrid Redis + memory caching for faster responses. Falls back to memory-only if Redis isn't available.

**API Server**: `main.py` is a FastAPI server that handles code review requests. Includes WebSocket support for real-time updates, rate limiting, and monitoring.

## Quick Start

### For Mac Development (Local Testing)
```bash
# Setup
python3.10 -m venv code-review
source code-review/bin/activate
pip install -r requirements.txt

# Option 1: Use the startup script (recommended)
./scripts/dev-start.sh

# Option 2: Run directly with environment variables
ENVIRONMENT=development MODEL_PATH=codellama/CodeLlama-7b-hf python src/serving/api/main.py

# Option 3: Copy config and run
cp config/development.env .env
python src/serving/api/main.py
```

### For Production (CUDA Server)
```bash
# Setup (same as above)

# Option 1: Use the startup script (recommended)
./scripts/prod-start.sh

# Option 2: Run directly with environment variables
ENVIRONMENT=production MODEL_PATH=codellama/CodeLlama-7b-hf QUANTIZATION_BITS=4 python src/serving/api/main.py

# Option 3: Copy config and run
cp config/production.env .env
python src/serving/api/main.py
```

## Environment Differences

This system works on both Mac (for development) and CUDA servers (for production), but they're configured differently:

### Mac Development
- **Device**: CPU (avoids MPS memory allocation issues)
- **Model**: Full CodeLlama-7B (if 32GB+ RAM) or smaller models (if 16GB RAM)
- **Memory**: ~15-20GB RAM usage for CodeLlama-7B
- **Speed**: ~30-60 seconds per review (CPU inference)
- **Cache**: Memory-only (no Redis needed)
- **Purpose**: Testing, development, experimentation with production-quality models

### CUDA Production  
- **Device**: NVIDIA GPU with CUDA
- **Model**: CodeLlama-7B with 4-bit quantization
- **Memory**: ~4-6GB VRAM usage
- **Speed**: ~1-3 seconds per review
- **Cache**: Redis + memory hybrid caching
- **Purpose**: High-throughput production usage

The same code automatically detects your environment and adjusts settings accordingly.

## API Endpoints

Once running, the API provides:

- `POST /review` - Submit code for review
- `GET /review/{id}` - Check review status
- `POST /review/batch` - Submit multiple reviews
- `GET /health` - Server health check
- `GET /metrics` - Prometheus metrics
- `WS /ws` - WebSocket for real-time updates

Example usage:
```bash
curl -X POST "http://localhost:8000/review" \
  -H "Content-Type: application/json" \
  -d '{
    "repo": "myproject/repo",
    "diff": "def hello(): print(\"Hello World\")",
    "urgency": "high",
    "language": "python"
  }'
```

## Testing

Run the test suites to verify everything works:

```bash
# Test cache functionality
python tests/test_cache_manager.py

# Test tokenization
python tests/test_tokenizer.py  

# Test model evaluation
python tests/test_model_evaluator.py
```


### Successful Startup Example (Mac with 64GB RAM)

When everything works correctly, you'll see logs like this:
```
INFO:     Started server process [83319]
INFO:__main__:Starting up API server...
WARNING:src.models.inference.cache_manager:Failed to connect to Redis: ... Falling back to memory-only cache.
INFO:__main__:Cache initialized
INFO:__main__:Mac with MPS detected: Using CPU to avoid MPS memory allocation issues
INFO:__main__:Environment: development
INFO:__main__:Device configuration: CUDA=False, MPS=True, Device=cpu
INFO:__main__:Model settings: Quantization=0bit, Batch=1, Compile=False
INFO:__main__:Memory settings: Cache=50, SeqLen=1024, NewTokens=256
INFO:src.models.inference.optimized_inference:Initializing inference pipeline with model: codellama/CodeLlama-7b-hf
Loading checkpoint shards: 100%|████████████████| 2/2 [00:25<00:00, 12.51s/it]
INFO:src.models.inference.optimized_inference:Inference pipeline initialized successfully
INFO:__main__:API server ready
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### Environment Variables Reference

Common variables you can set when running main.py:

```bash
# Model configuration
MODEL_PATH=codellama/CodeLlama-7b-hf          # Which model to use
QUANTIZATION_BITS=0                           # 0=disabled, 4=4-bit (CUDA only)
MAX_BATCH_SIZE=1                              # Requests processed together
COMPILE_MODEL=false                           # Enable torch.compile (CUDA only)

# Memory settings
MAX_SEQUENCE_LENGTH=1024                      # Max input tokens
MAX_NEW_TOKENS=256                            # Max generated tokens
MEMORY_CACHE_SIZE=50                          # Cache items in memory

# API settings
API_HOST=127.0.0.1                            # Bind address (0.0.0.0 for production)
API_PORT=8000                                 # Port number
API_WORKERS=1                                 # Number of worker processes

# Cache settings
ENABLE_REDIS=false                            # Use Redis for caching
REDIS_HOST=localhost                          # Redis server address
CACHE_TTL=1800                                # Cache expiration (seconds)

# Environment detection
ENVIRONMENT=development                       # development or production
```

## Configuration

Configuration files are in the `config/` directory:

- `development.env` - Mac-friendly settings
- `production.env` - CUDA server optimized settings
- `README.md` - Detailed configuration guide

The system automatically copies the right config based on your environment, but you can manually override settings with environment variables.

## Current Focus

The main model I'm testing is [CodeLlama-7B](https://huggingface.co/codellama/CodeLlama-7b-hf) but I'm comparing it against others to see what works best for code review tasks.

Working on:
- Better evaluation metrics for code review quality
- Fine-tuning approaches with LoRA
- Scaling to handle more concurrent requests
- Improving review accuracy and usefulness

## Documentation

- **[Mac vs CUDA Development](learning_materials/mac_vs_cuda_guide.md)** - Detailed comparison of development environments
- **[Issues and Workarounds](docs/issues_and_workarounds.md)** - Common problems and solutions encountered during development
- **[Configuration Guide](config/README.md)** - Complete configuration reference