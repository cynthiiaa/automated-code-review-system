# Automated Code Review System

I'm building this to understand how LLMs work with code by creating a system that reviews pull requests automatically. Still figuring things out as I go.

## What I have so far

**Data Collection**: `github_collector.py` pulls PR data from repos - diffs, comments, file changes. It's how I'm getting training data.

**Tokenization**: `code_aware_tokenizer.py` handles the tokenization challenges with code. Compares how different tokenizers (CodeLlama vs GPT-4) handle code structures, whitespace, and diffs. Also does smart truncation to fit context windows.

**Model Testing**: `model_evaluator.py` benchmarks different models (CodeLlama, CodeBERT, CodeT5, StarCoder) on latency, throughput, and quality. Still working on the evaluation metrics.

## Setup

```bash
python3.10 -m venv code-review
source code-review/bin/activate
pip install -r requirements.txt
```

The main model I'm testing is [CodeLlama-7B](https://huggingface.co/codellama/CodeLlama-7b-hf) but I'm comparing it against others to see what works best for code review tasks.
