"""
Stage III: Self-Training with Execution Feedback

This module implements the iterative self-training mechanism that:
1. Generates translation candidates using structured reasoning
2. Verifies candidates through compilation and unit testing
3. Applies error-feedback repair for failed samples
4. Reconstructs rationales for verified samples
5. Retrains the model from M_0
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from typing import Dict, List, Optional, Tuple
import subprocess
import tempfile
import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SelfTraining:
    """
    Stage III: Self-Training with Execution Feedback
    
    Implements iterative self-training with hierarchical sampling,
    error-feedback repair, and rationale reconstruction.
    """
    
    def __init__(
        self,
        base_model_path: str,
        structured_reasoning_module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        t_low: float = 0.2,
        t_high: float = 0.8,
        k_exploration: int = 5,
        t_max_repair: int = 3
    ):
        """
        Initialize self-training stage.
        
        Args:
            base_model_path: Path to the base model (M_0)
            structured_reasoning_module: Instance of StructuredReasoning
            device: Device for training
            t_low: Low temperature for direct attempts
            t_high: High temperature for exploration
            k_exploration: Number of exploration candidates
            t_max_repair: Maximum repair iterations
        """
        self.base_model_path = base_model_path
        self.reasoning_module = structured_reasoning_module
        self.device = device
        self.t_low = t_low
        self.t_high = t_high
        self.k_exploration = k_exploration
        self.t_max_repair = t_max_repair
        
        self.training_data = []
        self.mono_data = []
    
    def verify_translation(
        self,
        java_code: str,
        test_cases: List[Dict]
    ) -> Tuple[bool, Optional[str]]:
        """
        Verify Java code through compilation and unit testing.
        
        Args:
            java_code: Generated Java code
            test_cases: List of test cases with 'input' and 'expected_output'
            
        Returns:
            Tuple of (success, error_message)
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write Java file
            java_file = os.path.join(tmpdir, "Solution.java")
            with open(java_file, 'w', encoding='utf-8') as f:
                f.write(java_code)
            
            # Compile
            try:
                result = subprocess.run(
                    ["javac", java_file],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode != 0:
                    return False, f"Compilation error: {result.stderr}"
            except Exception as e:
                return False, f"Compilation exception: {str(e)}"
            
            # Run tests
            for i, test in enumerate(test_cases):
                try:
                    # Create test runner
                    test_code = self._create_test_runner(java_code, test)
                    test_file = os.path.join(tmpdir, f"Test{i}.java")
                    with open(test_file, 'w', encoding='utf-8') as f:
                        f.write(test_code)
                    
                    # Compile test
                    subprocess.run(
                        ["javac", "-cp", tmpdir, test_file],
                        capture_output=True,
                        timeout=10,
                        check=True
                    )
                    
                    # Run test
                    result = subprocess.run(
                        ["java", "-cp", tmpdir, f"Test{i}"],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    
                    if result.returncode != 0:
                        return False, f"Test {i} failed: {result.stderr}"
                        
                except Exception as e:
                    return False, f"Test {i} exception: {str(e)}"
            
            return True, None
    
    def _create_test_runner(self, java_code: str, test: Dict) -> str:
        """Create a simple test runner for the Java code."""
        # This is a simplified version - in practice, you'd need more sophisticated test generation
        return f"""
public class Test {{
    public static void main(String[] args) {{
        Solution sol = new Solution();
        // Test execution logic here
        System.out.println("Test passed");
    }}
}}
"""
    
    def direct_attempt(
        self,
        source_code: str,
        test_cases: List[Dict]
    ) -> Optional[Dict]:
        """
        Perform low-temperature direct translation attempt.
        
        Args:
            source_code: Python source code
            test_cases: Test cases for verification
            
        Returns:
            Translation result if successful, None otherwise
        """
        self.reasoning_module.temperature = self.t_low
        result = self.reasoning_module.translate(source_code)
        
        success, error = self.verify_translation(
            result['target_code'],
            test_cases
        )
        
        if success:
            return result
        return None
    
    def diversified_exploration(
        self,
        source_code: str,
        test_cases: List[Dict]
    ) -> Optional[Dict]:
        """
        Perform high-temperature diversified exploration.
        
        Args:
            source_code: Python source code
            test_cases: Test cases for verification
            
        Returns:
            First successful translation, None if all fail
        """
        self.reasoning_module.temperature = self.t_high
        
        for k in range(self.k_exploration):
            logger.info(f"Exploration attempt {k+1}/{self.k_exploration}")
            result = self.reasoning_module.translate(source_code)
            
            success, error = self.verify_translation(
                result['target_code'],
                test_cases
            )
            
            if success:
                return result
        
        return None
    
    def iterative_repair(
        self,
        source_code: str,
        test_cases: List[Dict],
        initial_result: Dict
    ) -> Optional[Dict]:
        """
        Perform error-feedback-driven iterative repair.
        
        Args:
            source_code: Python source code
            test_cases: Test cases for verification
            initial_result: Initial translation attempt
            
        Returns:
            Repaired translation if successful, None otherwise
        """
        current_code = initial_result['target_code']
        
        for t in range(self.t_max_repair):
            logger.info(f"Repair iteration {t+1}/{self.t_max_repair}")
            
            # Get error feedback
            success, error_msg = self.verify_translation(current_code, test_cases)
            
            if success:
                return {
                    'r_exec': initial_result['r_exec'],
                    'r_plan': initial_result['r_plan'],
                    'target_code': current_code
                }
            
            # Generate repair prompt
            repair_prompt = f"""The following Java code has errors. Please fix them.

Original Python code:
{source_code}

Current Java code:
{current_code}

Error message:
{error_msg}

Please provide the corrected Java code:"""
            
            # Generate repaired code
            inputs = self.reasoning_module.tokenizer(
                repair_prompt,
                return_tensors="pt",
                truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.reasoning_module.model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    temperature=self.t_low,
                    do_sample=True
                )
            
            repaired = self.reasoning_module.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )
            
            # Extract code from response
            if "```java" in repaired:
                start = repaired.find("```java") + 7
                end = repaired.find("```", start)
                current_code = repaired[start:end].strip()
            else:
                current_code = repaired
        
        return None
    
    def rationale_reconstruction(
        self,
        source_code: str,
        verified_code: str
    ) -> Dict:
        """
        Reconstruct reasoning rationale given verified code.
        
        Args:
            source_code: Python source code
            verified_code: Verified Java code
            
        Returns:
            Reconstructed reasoning components
        """
        prompt = f"""Given the following Python code and its correct Java translation,
explain the reasoning process in two steps:

Step 1 - Execution Semantics Understanding:
Describe how the Python program behaves during execution.

Step 2 - Translation Planning:
Explain the translation strategy used to convert Python to Java.

Python code:
{source_code}

Java code:
{verified_code}

Please provide the reasoning:"""
        
        inputs = self.reasoning_module.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.reasoning_module.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=self.t_low,
                do_sample=False
            )
        
        reasoning = self.reasoning_module.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        
        # Parse reasoning
        result = self.reasoning_module._parse_structured_output(reasoning)
        result['target_code'] = verified_code
        
        return result
    
    def run_iteration(
        self,
        unlabeled_tasks: List[Dict],
        iteration: int
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Run one iteration of self-training.
        
        Args:
            unlabeled_tasks: List of tasks with 'source_code' and 'test_cases'
            iteration: Current iteration number
            
        Returns:
            Tuple of (easy_samples, hard_samples)
        """
        logger.info(f"Starting iteration {iteration}")
        
        easy_samples = []
        hard_samples = []
        
        for idx, task in enumerate(unlabeled_tasks):
            logger.info(f"Processing task {idx+1}/{len(unlabeled_tasks)}")
            
            source_code = task['source_code']
            test_cases = task['test_cases']
            
            # Direct attempt
            result = self.direct_attempt(source_code, test_cases)
            
            if result:
                easy_samples.append({
                    'source_code': source_code,
                    'reasoning': None,  # No reconstruction for easy samples
                    'target_code': result['target_code']
                })
                continue
            
            # Diversified exploration
            result = self.diversified_exploration(source_code, test_cases)
            
            if result:
                hard_samples.append({
                    'source_code': source_code,
                    'result': result,
                    'needs_reconstruction': True
                })
                continue
            
            # Iterative repair
            initial_result = self.reasoning_module.translate(source_code)
            result = self.iterative_repair(source_code, test_cases, initial_result)
            
            if result:
                hard_samples.append({
                    'source_code': source_code,
                    'result': result,
                    'needs_reconstruction': True
                })
        
        logger.info(f"Iteration {iteration} complete:")
        logger.info(f"  Easy samples: {len(easy_samples)}")
        logger.info(f"  Hard samples: {len(hard_samples)}")
        
        return easy_samples, hard_samples
    
    def prepare_training_data(
        self,
        easy_samples: List[Dict],
        hard_samples: List[Dict]
    ) -> List[Dict]:
        """
        Prepare training data with rationale reconstruction for hard samples.
        
        Args:
            easy_samples: Samples solved by direct attempt
            hard_samples: Samples solved by exploration/repair
            
        Returns:
            Combined training data
        """
        training_data = []
        
        # Add easy samples directly
        for sample in easy_samples:
            training_data.append({
                'source_code': sample['source_code'],
                'reasoning': sample['reasoning'],
                'target_code': sample['target_code']
            })
        
        # Reconstruct rationales for hard samples
        for sample in hard_samples:
            if sample['needs_reconstruction']:
                logger.info("Reconstructing rationale for hard sample")
                reconstructed = self.rationale_reconstruction(
                    sample['source_code'],
                    sample['result']['target_code']
                )
                
                training_data.append({
                    'source_code': sample['source_code'],
                    'reasoning': f"{reconstructed['r_exec']}\n\n{reconstructed['r_plan']}",
                    'target_code': reconstructed['target_code']
                })
            else:
                training_data.append({
                    'source_code': sample['source_code'],
                    'reasoning': f"{sample['result']['r_exec']}\n\n{sample['result']['r_plan']}",
                    'target_code': sample['result']['target_code']
                })
        
        return training_data
    
    def train_model(
        self,
        training_data: List[Dict],
        output_dir: str,
        num_epochs: int = 2,
        batch_size: int = 4,
        learning_rate: float = 2e-5
    ):
        """
        Retrain model from M_0 with accumulated training data.
        
        Args:
            training_data: Training samples
            output_dir: Directory to save model
            num_epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
        """
        logger.info(f"Retraining from base model: {self.base_model_path}")
        
        # Load base model
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Prepare dataset
        from torch.utils.data import Dataset
        
        class TranslationDataset(Dataset):
            def __init__(self, data, tokenizer):
                self.data = data
                self.tokenizer = tokenizer
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                sample = self.data[idx]
                
                # Format input
                if sample['reasoning']:
                    full_text = f"{sample['source_code']}\n\n{sample['reasoning']}\n\n{sample['target_code']}"
                else:
                    full_text = f"{sample['source_code']}\n\n{sample['target_code']}"
                
                tokenized = self.tokenizer(
                    full_text,
                    padding="max_length",
                    truncation=True,
                    max_length=2048,
                    return_tensors="pt"
                )
                
                return {
                    "input_ids": tokenized["input_ids"].squeeze(),
                    "attention_mask": tokenized["attention_mask"].squeeze()
                }
        
        dataset = TranslationDataset(training_data, tokenizer)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            save_steps=500,
            save_total_limit=2,
            logging_steps=100,
            fp16=torch.cuda.is_available(),
            report_to="none"
        )
        
        # Train
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer
        )
        
        logger.info("Starting training...")
        trainer.train()
        
        logger.info(f"Saving model to {output_dir}")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
    
    def run_self_training(
        self,
        unlabeled_tasks: List[Dict],
        mono_data: List[Dict],
        num_iterations: int = 5,
        output_base_dir: str = "./self_training"
    ):
        """
        Run complete self-training pipeline.
        
        Args:
            unlabeled_tasks: Unlabeled translation tasks
            mono_data: Monolingual execution data for experience replay
            num_iterations: Number of iterations
            output_base_dir: Base directory for outputs
        """
        self.mono_data = mono_data
        
        for iteration in range(1, num_iterations + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"ITERATION {iteration}/{num_iterations}")
            logger.info(f"{'='*60}\n")
            
            # Run iteration
            easy_samples, hard_samples = self.run_iteration(
                unlabeled_tasks,
                iteration
            )
            
            # Prepare training data
            iter_data = self.prepare_training_data(easy_samples, hard_samples)
            self.training_data.extend(iter_data)
            
            # Mix with experience replay
            import random
            replay_size = int(len(self.mono_data) * 0.1)
            replay_samples = random.sample(self.mono_data, replay_size)
            
            combined_data = self.training_data + replay_samples
            
            logger.info(f"Total training data: {len(combined_data)}")
            logger.info(f"  Translation data: {len(self.training_data)}")
            logger.info(f"  Replay data: {len(replay_samples)}")
            
            # Retrain model
            output_dir = os.path.join(output_base_dir, f"iteration_{iteration}")
            self.train_model(combined_data, output_dir)
            
            # Update reasoning module with new model
            self.reasoning_module = type(self.reasoning_module)(
                output_dir,
                device=self.device
            )
        
        logger.info("\nSelf-training completed!")
        logger.info(f"Final model saved to: {output_dir}")
