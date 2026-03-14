"""
Stage I: Execution Semantics Warm-up

This module implements the first stage of SemTrans, which trains the model
to understand execution semantics through multi-task learning using PyX dataset:
1. NL-to-code generation (intent alignment)
2. Forward execution simulation
3. Backward abstraction reasoning
"""

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExecutionSemanticsWarmup:
    """
    Stage I: Execution Semantics Warm-up
    
    Trains the model to understand program execution through multi-task learning
    using the PyX dataset directly.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
        lambda_weight: float = 1.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the warmup stage.
        
        Args:
            model_name: Base model to use
            lambda_weight: Weight for execution reasoning tasks (λ in paper)
            device: Device to use for training
        """
        self.model_name = model_name
        self.lambda_weight = lambda_weight
        self.device = device
        
        logger.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def prepare_pyx_data(self, pyx_samples: List[Dict]) -> List[Dict]:
        """
        Prepare PyX dataset for multi-task training.
        
        PyX dataset format:
        {
            "task_id": str,
            "prompt": str,  # Natural language description
            "code": str,    # Python code
            "test_list": List[str],  # Test cases
            "monologue": str  # Execution trace/reasoning
        }
        
        Args:
            pyx_samples: List of PyX dataset samples
            
        Returns:
            Formatted training samples for all three tasks
        """
        formatted_samples = []
        
        for sample in pyx_samples:
            prompt = sample.get('prompt', '')
            code = sample.get('code', '')
            monologue = sample.get('monologue', '')
            test_list = sample.get('test_list', [])
            
            # Task 1: NL-to-code generation (intent alignment)
            if prompt and code:
                intent_input = f"# Task: {prompt}\n# Generate Python code:\n"
                formatted_samples.append({
                    "input": intent_input,
                    "output": code,
                    "task_type": "intent",
                    "weight": 1.0
                })
            
            # Task 2: Forward execution simulation
            if code and monologue:
                forward_input = f"# Simulate the execution of this code:\n{code}\n\n# Execution trace:\n"
                formatted_samples.append({
                    "input": forward_input,
                    "output": monologue,
                    "task_type": "forward",
                    "weight": self.lambda_weight
                })
            
            # Task 3: Backward abstraction reasoning
            # Extract test cases as examples of input-output behavior
            if code and test_list and len(test_list) > 0:
                test_example = test_list[0] if test_list else ""
                backward_input = f"# Given this code:\n{code}\n\n# Example test: {test_example}\n# Infer the preconditions and constraints:\n"
                backward_output = f"The function should handle: {prompt}" if prompt else "Standard input validation required."
                formatted_samples.append({
                    "input": backward_input,
                    "output": backward_output,
                    "task_type": "backward",
                    "weight": self.lambda_weight
                })
        
        return formatted_samples
    
    def train(
        self,
        pyx_data: List[Dict],
        output_dir: str = "./models/stage1_warmup",
        num_epochs: int = 2,
        batch_size: int = 4,
        learning_rate: float = 2e-5,
        save_steps: int = 500,
        max_length: int = 2048
    ):
        """
        Train the model on PyX dataset with multi-task learning.
        
        Args:
            pyx_data: PyX dataset samples
            output_dir: Directory to save the trained model
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            save_steps: Save checkpoint every N steps
            max_length: Maximum sequence length
        """
        logger.info("Preparing PyX training data...")
        
        # Prepare all data from PyX dataset
        all_samples = self.prepare_pyx_data(pyx_data)
        
        # Count samples by task type
        intent_count = sum(1 for s in all_samples if s['task_type'] == 'intent')
        forward_count = sum(1 for s in all_samples if s['task_type'] == 'forward')
        backward_count = sum(1 for s in all_samples if s['task_type'] == 'backward')
        
        logger.info(f"Total training samples: {len(all_samples)}")
        logger.info(f"  - Intent (NL-to-code): {intent_count}")
        logger.info(f"  - Forward (execution simulation): {forward_count}")
        logger.info(f"  - Backward (abstraction reasoning): {backward_count}")
        
        # Create dataset
        class PyXWarmupDataset(Dataset):
            """Dataset for PyX multi-task training."""
            
            def __init__(self, samples, tokenizer, max_length=2048):
                self.samples = samples
                self.tokenizer = tokenizer
                self.max_length = max_length
            
            def __len__(self):
                return len(self.samples)
            
            def __getitem__(self, idx):
                sample = self.samples[idx]
                
                # Combine input and output
                full_text = sample["input"] + sample["output"]
                
                # Tokenize
                tokenized = self.tokenizer(
                    full_text,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                
                # Create labels (same as input_ids for causal LM)
                labels = tokenized["input_ids"].clone()
                
                # Mask the input part (only compute loss on output)
                input_text = sample["input"]
                input_tokens = self.tokenizer(
                    input_text,
                    truncation=True,
                    max_length=self.max_length
                )
                input_length = len(input_tokens["input_ids"])
                labels[0, :input_length] = -100  # Ignore loss for input tokens
                
                return {
                    "input_ids": tokenized["input_ids"].squeeze(),
                    "attention_mask": tokenized["attention_mask"].squeeze(),
                    "labels": labels.squeeze()
                }
        
        dataset = PyXWarmupDataset(all_samples, self.tokenizer, max_length)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            save_steps=save_steps,
            save_total_limit=3,
            logging_steps=100,
            logging_dir=f"{output_dir}/logs",
            fp16=torch.cuda.is_available(),
            gradient_accumulation_steps=2,
            warmup_steps=100,
            weight_decay=0.01,
            report_to="none",
            save_strategy="steps",
            evaluation_strategy="no"
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.tokenizer
        )
        
        logger.info("Starting Stage I training with PyX dataset...")
        logger.info(f"Training configuration:")
        logger.info(f"  - Epochs: {num_epochs}")
        logger.info(f"  - Batch size: {batch_size}")
        logger.info(f"  - Learning rate: {learning_rate}")
        logger.info(f"  - Lambda weight: {self.lambda_weight}")
        
        trainer.train()
        
        logger.info(f"Saving model to {output_dir}")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info("Stage I training completed!")
        logger.info(f"Model saved to: {output_dir}")
    
    def save_model(self, output_dir: str):
        """Save the trained model."""
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Model saved to {output_dir}")
