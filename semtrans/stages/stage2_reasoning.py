"""
Stage II: Structured Reasoning for Translation

This module implements the structured reasoning paradigm that decomposes
translation into three explicit steps:
1. Execution semantics understanding (R_exec)
2. Translation planning (R_plan)
3. Target code generation (Y)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StructuredReasoning:
    """
    Stage II: Structured Reasoning for Translation
    
    Implements the three-step reasoning paradigm for code translation.
    """
    
    TRANSLATION_PROMPT = """Please translate the following Python code into Java.

Your goal is to produce Java code that is **semantically equivalent** to the original Python program.

Constraints:
- Do NOT add a main method.
- The Java implementation must be placed inside a class named 'Solution'.
- Preserve the original algorithm and semantics.

Before generating the Java code, you must perform structured reasoning in the following three steps:

Step 1 — Execution Semantics Understanding (R_exec)
Analyze how the Python program behaves during execution.
Describe:
- The overall purpose of the program
- Control-flow structure (loops, conditionals)
- Roles of key variables
- How variable values evolve during execution
- Important intermediate states and outputs

Step 2 — Translation Planning (R_plan)
Explain how the Python program should be implemented in Java.
Your plan should include:
- Type mapping (dynamic typing → static typing)
- Data structure mapping
- API mapping (Python libraries → Java equivalents)
- Control-flow transformation if necessary
- Any language-specific differences that must be handled

Step 3 — Java Code Generation
Generate the final Java implementation based on the reasoning above.

Python source code:
{source_code}
"""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        temperature: float = 0.2
    ):
        """
        Initialize the structured reasoning module.
        
        Args:
            model_path: Path to the trained model from Stage I
            device: Device to use for inference
            temperature: Sampling temperature
        """
        self.device = device
        self.temperature = temperature
        
        logger.info(f"Loading model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto"
        )
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def translate(
        self,
        source_code: str,
        max_length: int = 2048,
        return_reasoning: bool = True
    ) -> Dict[str, str]:
        """
        Translate Python code to Java using structured reasoning.
        
        Args:
            source_code: Python source code to translate
            max_length: Maximum generation length
            return_reasoning: Whether to return intermediate reasoning steps
            
        Returns:
            Dictionary containing:
                - 'r_exec': Execution semantics understanding
                - 'r_plan': Translation planning
                - 'target_code': Generated Java code
                - 'full_output': Complete model output
        """
        # Format prompt
        prompt = self.TRANSLATION_PROMPT.format(source_code=source_code)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract reasoning components
        result = {
            "full_output": full_output,
            "r_exec": "",
            "r_plan": "",
            "target_code": ""
        }
        
        if return_reasoning:
            result = self._parse_structured_output(full_output)
        
        return result
    
    def _parse_structured_output(self, output: str) -> Dict[str, str]:
        """
        Parse the structured output into components.
        
        Args:
            output: Full model output
            
        Returns:
            Dictionary with parsed components
        """
        result = {
            "full_output": output,
            "r_exec": "",
            "r_plan": "",
            "target_code": ""
        }
        
        # Extract R_exec
        if "Step 1" in output:
            start = output.find("Step 1")
            end = output.find("Step 2") if "Step 2" in output else len(output)
            result["r_exec"] = output[start:end].strip()
        
        # Extract R_plan
        if "Step 2" in output:
            start = output.find("Step 2")
            end = output.find("Step 3") if "Step 3" in output else len(output)
            result["r_plan"] = output[start:end].strip()
        
        # Extract target code
        if "Step 3" in output:
            start = output.find("Step 3")
            code_start = output.find("```java", start)
            if code_start != -1:
                code_start = output.find("\n", code_start) + 1
                code_end = output.find("```", code_start)
                if code_end != -1:
                    result["target_code"] = output[code_start:code_end].strip()
            else:
                # Try to extract code without markdown
                result["target_code"] = output[start:].strip()
        
        return result
    
    def batch_translate(
        self,
        source_codes: list,
        batch_size: int = 4,
        max_length: int = 2048
    ) -> list:
        """
        Translate multiple Python programs in batches.
        
        Args:
            source_codes: List of Python source codes
            batch_size: Batch size for processing
            max_length: Maximum generation length
            
        Returns:
            List of translation results
        """
        results = []
        
        for i in range(0, len(source_codes), batch_size):
            batch = source_codes[i:i + batch_size]
            logger.info(f"Processing batch {i // batch_size + 1}/{(len(source_codes) + batch_size - 1) // batch_size}")
            
            for source_code in batch:
                result = self.translate(source_code, max_length=max_length)
                results.append(result)
        
        return results
