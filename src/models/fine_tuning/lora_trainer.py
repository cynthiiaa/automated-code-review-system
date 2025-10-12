from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    TrainerCallback
)
import torch
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import json
from dataclasses import dataclass, asdict
import numpy as np
from evaluate import load
import logging

logging.basicConfig(level=logging.INFO)  
logger = logging.getLogger(__name__)

@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning"""
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: list = None
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    def save(self, path: str):
        """Save configuration to JSON file"""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str):
        """Load configuration from JSON file"""
        with open(path, 'r') as f:
            return cls(**json.load(f))

@dataclass
class TrainerConfig:
    """Configuration for training"""
    output_dir: str = "./models/code-review-lora"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    warmup_ratio: float = 0.1
    learning_rate: float = 2e-4
    logging_steps: int = 25
    save_strategy: str = "epoch"
    evaluation_strategy: str = "epoch"
    save_total_limit: int = 3
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True
    report_to: str = "wandb"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    push_to_hub: bool = False
    resume_from_checkpoint: Optional[str] = None
    max_grad_norm: float = 0.3
    optim: str = "paged_adamw_8bit"
    lr_scheduler_type: str = "cosine"

    def to_training_args(self) -> TrainingArguments:
        """Convert to HuggingFace TrainingArguments"""
        return TrainingArguments(**asdict(self))
    
class MetricsCallback(TrainerCallback):
    """Custom callback for tracking additional metrics"""

    def __init__(self):
        self.metrics_history = []
        self.perplexity_metric = load("perplexity", module_type="metric")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Track metrics during evaluation"""
        if metrics:
            # calculate perplexity
            if "eval_loss" in metrics:
                metrics["eval_perplexity"] = np.exp(metrics["eval_loss"])
            
            self.metrics_history.append(metrics)
            logger.info(f"Evaluation metrics: {metrics}")
    
    def save_metrics(self, path: str):
        """Save metrics history to file"""
        with open(path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)

class CodeReviewLoRATrainer:
    def __init__(
            self,
            base_model: str,
            lora_config: Optional[LoRAConfig] = None,
            trainer_config: Optional[TrainerConfig] = None,
            device: Optional[str] = None
    ):
        """
        Initialize LoRA trainer for code review models

        Args:
            base_model: HuggingFace model name or path
            lora_config: LoRA configuration
            trainer_config: Training configuration
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.base_model = base_model
        self.lora_config = lora_config or LoRAConfig()
        self.trainer_config = trainer_config or TrainerConfig()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.model = None

        logger.info(f"Initialized trainer with device: {self.device}")
        
    def prepare_lora_model(self) -> Tuple[Any, Any]:
        """ Configure LoRA for efficient fine-tuning
            Returns:
                Tuple of (model, tokenizer)
        """
        try:
            # load tokenizer
            logger.info(f"Loading tokenizer for {self.base_model}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model,
                trust_remote_code=True
            )

            # set padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # configure LoRA
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.lora_config.r,
                lora_alpha=self.lora_config.lora_alpha,
                lora_dropout=self.lora_config.lora_dropout,
                target_modules=self.lora_config.target_modules,
                bias=self.lora_config.bias
            )

            # load model with optimizations
            logger.info(f"Loading model {self.base_model}")
            model_kwargs = {
                "trust_remote_code": True,  
                "device_map": "auto" if self.device == "cuda" else None
            }

            # use appropriate precision based on hardware
            if self.device == "cuda":
                if torch.cuda.is_bf16_supported():
                    model_kwargs["torch_dtype"] = torch.bfloat16
                else:
                    model_kwargs["torch_dtype"] = torch.float16
                
                # optional: use 8-bit quantization for large models
                if self.trainer_config.optim == "paged_adamw_8bit":
                    model_kwargs["load_in_8bit"] = True
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                **model_kwargs
            )

            # prepare model for training
            self.model.config.use_cache = False # this is needed for gradient checkpointing

            # apply LoRA
            logger.info("Applying LoRA configuration")
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()

            # enable gradient checkpointing if specified
            if self.trainer_config.gradient_checkpointing:
                self.model.enable_input_require_grads()
                self.model.gradient_checkpointing_enable()
            
            return self.model, self.tokenizer
        
        except Exception as e:
            logger.error(f"Error preparing LoRA model: {e}")
            raise

    def get_data_collator(self):
        """Create data collator for efficient batching"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized. Call prepare_lora_model first.")
        
        return DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
    
    def train(
            self,
            train_dataset,
            eval_dataset=None,
            callbacks=None,
            compute_metrics=None
    ):
        """ Train the model with LoRA
            Args:
                train_dataset: Training dataset
                eval_dataset: Evaluation dataset
                callbacks: Additional training callbacks
                compute_metrics: Function to compute metrics
            
            Returns:
                Trainer object with results
        """
        if self.model is None or self.tokenizer is None:
            logger.info("Model not initialized, preparing LoRA model...")
            self.prepare_lora_model()

        # prepare callbacks
        all_callbacks = []

        # add early stopping if evaluation dataset provided
        if eval_dataset is not None:
            all_callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=3,
                    early_stopping_threshold=0.001
                )
            )

        # add metrics callback
        metrics_callback = MetricsCallback()
        all_callbacks.append(metrics_callback)

        # add user callbacks
        if callbacks:
            all_callbacks.extend(callbacks)
        
        # get training arguments
        training_args = self.trainer_config.to_training_args()

        # create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.get_data_collator(),
            compute_metrics=compute_metrics,
            callbacks=all_callbacks
        )

        # train
        logger.info("Starting training...")
        try:
            if self.trainer_config.resume_from_checkpoint:
                logger.info(f"Resuming from checkpoint: {self.trainer_config.resume_from_checkpoint}")
            result = trainer.train(
                resume_from_checkpoint=self.trainer_config.resume_from_checkpoint
            )

            logger.info(f"Training completed. Results: {result}")

            # save metrics
            metrics_path = Path(self.trainer_config.output_dir) / "metrics_history.json"
            metrics_callback.save_metrics(str(metrics_path))

            return trainer
        
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    def save_model(self, output_dir: str):
        """ Save the trained LoRA model
            Args:
                output_dir: Directory to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # save model
            logger.info(f"Saving model to {output_dir}")
            self.model.save_pretrained(output_dir)

            # save tokenizer
            if self.tokenizer:
                self.tokenizer.save_pretrained(output_dir)

            # Save configurations
            lora_config_path = output_path / "lora_config.json"
            self.lora_config.save(str(lora_config_path))

            trainer_config_path = output_path / "trainer_config.json"
            with open(trainer_config_path, 'w') as f:
                json.dump(asdict(self.trainer_config), f, indent=2)
            
            logger.info(f"Model saved successfully to {output_dir}")

        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    @classmethod
    def load_model(cls, model_dir: str, device: Optional[str] = None):
        """ Load a saved LoRA model
            Args:
                model_dir: Directory containing the saved model
                device: Device to load the model on
            Returns:
                Loaded trainer instance
        """

        try:
            model_path = Path(model_dir)

            # load configs
            lora_config = LoRAConfig.load(str(model_path / "lora_config.json"))

            trainer_config_path = model_path / "trainer_config.json"
            if trainer_config_path.exists():
                with open(trainer_config_path, 'r') as f:
                    trainer_config = TrainerConfig(**json.load(f))
            else:
                trainer_config = TrainerConfig()

            # create trainer instance
            trainer = cls(
                base_model=str(model_dir),
                lora_config=lora_config,
                trainer_config=trainer_config,
                device=device
            )

            # load model and tokenizer
            from peft import PeftModel

            trainer.tokenizer = AutoTokenizer.from_pretrained(model_dir)

            # load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                device_map = "auto" if device == "cuda" else None,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )

            # load LoRA weights
            trainer.model = PeftModel.from_pretrained(base_model, model_dir)

            logger.info(f"Model loaded from {model_dir}")
            return trainer
        
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def inference(self, text: str, max_length: int = 512, temperature: float = 0.7):
        """ Run inference with the trained model

            Args:
                text: input text
                max_length: maximum generation length
                temperature: sampling temperature
            
            Returns:
                Generated text
        """

        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not initialized")
        
        try:
            # tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length
            ).to(self.device)

            # generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.95,
                    num_return_sequences=1
                )
            
            # decode
            generated_text = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )

            return generated_text
        
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise
 