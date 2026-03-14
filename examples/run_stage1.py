"""
Example: Running Stage I - Execution Semantics Warm-up

This script trains the model using the PyX dataset directly.
The PyX dataset contains Python code with execution traces and natural language descriptions.
"""

from semtrans.stages import ExecutionSemanticsWarmup
from semtrans.utils import load_pyx_dataset

print("="*60)
print("STAGE I: EXECUTION SEMANTICS WARM-UP")
print("="*60)

# Initialize Stage I
print("\nInitializing model...")
warmup = ExecutionSemanticsWarmup(
    model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
    lambda_weight=1.0  # Weight for execution reasoning tasks
)

# Load PyX dataset
print("\nLoading PyX dataset...")
pyx_data = load_pyx_dataset("data/pyx_dataset.jsonl")
print(f"Loaded {len(pyx_data)} samples from PyX dataset")

# Train the model directly on PyX data
# The prepare_pyx_data method will automatically create three types of tasks:
# 1. NL-to-code generation (intent alignment)
# 2. Forward execution simulation
# 3. Backward abstraction reasoning
print("\nStarting training...")
warmup.train(
    pyx_data=pyx_data,
    output_dir="./models/stage1_warmup",
    num_epochs=2,
    batch_size=4,
    learning_rate=2e-5,
    save_steps=500,
    max_length=2048
)

print("\n" + "="*60)
print("STAGE I COMPLETED!")
print("="*60)
print("Model saved to: ./models/stage1_warmup")
print("\nNext step: Run Stage II for structured reasoning translation")
print("  python examples/run_stage2.py")
