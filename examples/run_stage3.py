"""
Example: Running Stage III - Self-Training with Execution Feedback
"""

from semtrans.stages import StructuredReasoning, SelfTraining
from semtrans.utils import load_translation_dataset, load_pyx_dataset

# Initialize structured reasoning module
reasoning = StructuredReasoning(
    model_path="./models/stage1_warmup",
    temperature=0.2
)

# Initialize self-training
self_training = SelfTraining(
    base_model_path="./models/stage1_warmup",
    structured_reasoning_module=reasoning,
    t_low=0.2,
    t_high=0.8,
    k_exploration=5,
    t_max_repair=3
)

# Load unlabeled translation tasks
unlabeled_tasks = load_translation_dataset("data/translation_tasks.jsonl")

# Load monolingual data for experience replay
mono_data = load_pyx_dataset("data/pyx_dataset.jsonl")

# Run self-training for 5 iterations
self_training.run_self_training(
    unlabeled_tasks=unlabeled_tasks,
    mono_data=mono_data,
    num_iterations=5,
    output_base_dir="./models/self_training"
)

print("\n" + "="*60)
print("SELF-TRAINING COMPLETED!")
print("="*60)
print("Final model saved to: ./models/self_training/iteration_5")
print("\nYou can now use this model for translation:")
print("  from semtrans.stages import StructuredReasoning")
print("  reasoning = StructuredReasoning('./models/self_training/iteration_5')")
