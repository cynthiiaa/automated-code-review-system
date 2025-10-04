from typing import Dict, List, Any, Optional
from transformers import AutoTokenizer
import tiktoken
import re

class CodeAwareTokenizer:
    """Demonstrates tokenization challenges with code"""

    def __init__(self, model_name: str = "codellama/CodeLlama-7b-hf"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.gpt4_tokenizer = tiktoken.encoding_for_model("gpt-4")

    def analyze_tokenization(self, code: str) -> Dict[str, Any]:
        """Compare tokenization across different tokenizers"""

        if not code:
            return {
                "transformers_tokens": 0,
                "gpt4_tokens": 0,
                "efficiency_ratio": 0,
                "special_tokens": [],
                "whitespace_handling": {}
            }

        # standard tokenization
        tokens_transformers = self.tokenizer.encode(code)

        # GPT-4 tokenization
        tokens_gpt4 = self.gpt4_tokenizer.encode(code)

        # analyze differences
        analysis = {
            "transformers_tokens": len(tokens_transformers),
            "gpt4_tokens": len(tokens_gpt4),
            "efficiency_ratio": len(tokens_transformers) / len(tokens_gpt4) if len(tokens_gpt4) > 0 else 0,
            "special_tokens": self._identify_special_tokens(code),
            "whitespace_handling": self._analyze_whitespace(code)
        }

        return analysis
    
    def _identify_special_tokens(self, code: str) -> List[str]:
        """Identify special tokens and code-specific patterns"""
        special_patterns = [
            (r'\b(class|def|import|from|return|if|else|elif|for|while|try|except|with|as)\b', 'keyword'),
            (r'[\+\-\*/=%<>!&|^~]', 'operator'),
            (r'[{}\[\]()]', 'bracket'),
            (r'["\'].*?["\']', 'string'),
            (r'#.*$', 'comment'),
            (r'\b\d+\b', 'number'),
            (r'\b(None|True|False)\b', 'constant') 
        ]

        found_tokens = []
        for pattern, token_type in special_patterns:
            matches = re.findall(pattern, code, re.MULTILINE)
            if matches:
                found_tokens.extend([(match, token_type) for match in matches[:5]])

        return found_tokens
    
    def _analyze_whitespace(self, code: str) -> Dict[str, Any]:
        """Analyze whitespace handling in tokenization"""
        analysis = {
            "total_whitespace": len(re.findall(r'\s', code)),
            "indentation_levels": len(set(re.findall(r'^[ \t]+', code, re.MULTILINE))),
            "empty_lines": len(re.findall(r'^\s*$', code, re.MULTILINE)),
            "tabs_vs_spaces": {
                "tabs": code.count('\t'),
                "spaces": code.count(' ')
            }
        }

        tokens_with_ws = self.tokenizer.encode(code)
        tokens_no_ws = self.tokenizer.encode(re.sub(r'\s+', ' ', code))

        analysis["whitespace_token_impact"] = len(tokens_with_ws) - len(tokens_no_ws)

        return analysis
    
    def optimize_for_context_window(self, diff: str, max_tokens: int = 2400):
        """Smart truncation preserving code structure"""
        if not diff:
            return ""
        
        lines = diff.split('\n')

        changed_lines = []
        context_lines = []

        for i, line in enumerate(lines):
            if line.startswith(('+', '-')) and not line.startswith(('+++', '---')):
                changed_lines.append((i, line))
            else:
                context_lines.append((i, line))
        
        result_lines = []
        current_tokens = 0

        for idx, line in changed_lines:
            tokens = len(self.tokenizer.encode(line))
            if current_tokens + tokens <= max_tokens * 0.7:
                result_lines.append((idx, line))
                current_tokens += tokens
        
        remaining_tokens = max_tokens - current_tokens
        context_budget = remaining_tokens // 2

        for idx, line in context_lines:
            if current_tokens >= max_tokens:
                break
            tokens = len(self.tokenizer.encode(line))
            if tokens <= context_budget:
                result_lines.append((idx, line))
                current_tokens += tokens
                context_budget -= tokens

        result_lines.sort(key=lambda x: x[0])

        return "\n".join(line for _, line in result_lines)
