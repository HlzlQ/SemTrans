"""
SemTrans: Execution-Aware Code Translation with Structured Reasoning and Iterative Self-Training

A three-stage framework for improving LLM-based code translation through:
1. Execution semantics warm-up
2. Structured reasoning for translation
3. Self-training with execution feedback
"""

__version__ = "1.0.0"
__author__ = "Qi Han, Yan Guang, Hui Shu, Fei Kang"

from semtrans.stages.stage1_warmup import ExecutionSemanticsWarmup
from semtrans.stages.stage2_reasoning import StructuredReasoning
from semtrans.stages.stage3_selftraining import SelfTraining

__all__ = [
    "ExecutionSemanticsWarmup",
    "StructuredReasoning",
    "SelfTraining",
]
