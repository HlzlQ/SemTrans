"""
Example: Complete SemTrans Pipeline
"""

from semtrans.stages import ExecutionSemanticsWarmup, StructuredReasoning, SelfTraining
from semtrans.utils import (
    load_pyx_dataset,
    load_translation_dataset,
    evaluate_translation,
    print_evaluation_report
)

def main():
    print("="*60)
    print("SEMTRANS: Complete Pipeline")
    print("="*60)
    
    # ========== STAGE I: Execution Semantics Warm-up ==========
    print("\n[Stage I] Execution Semantics Warm-up")
    print("-"*60)
    
    warmup = ExecutionSemanticsWarmup(
        model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
        lambda_weight=1.0
    )
    
    # Load PyX dataset
    pyx_data = load_pyx_dataset("data/pyx_dataset.jsonl")
    print(f"Loaded {len(pyx_data)} samples from PyX dataset")
    
    # Train directly on PyX data (no need to format separately)
    warmup.train(
        pyx_data=pyx_data,
        output_dir="./models/stage1_warmup",
        num_epochs=2,
        batch_size=4,
        learning_rate=2e-5
    )
    
    print("✓ Stage I completed!")
    
    # ========== STAGE II: Structured Reasoning ==========
    print("\n[Stage II] Structured Reasoning for Translation")
    print("-"*60)
    
    reasoning = StructuredReasoning(
        model_path="./models/stage1_warmup",
        temperature=0.2
    )
    
    print("✓ Stage II initialized!")
    
    # ========== STAGE III: Self-Training ==========
    print("\n[Stage III] Self-Training with Execution Feedback")
    print("-"*60)
    
    self_training = SelfTraining(
        base_model_path="./models/stage1_warmup",
        structured_reasoning_module=reasoning,
        t_low=0.2,
        t_high=0.8,
        k_exploration=5,
        t_max_repair=3
    )
    
    unlabeled_tasks = load_translation_dataset("data/translation_tasks.jsonl")
    
    self_training.run_self_training(
        unlabeled_tasks=unlabeled_tasks,
        mono_data=pyx_data,
        num_iterations=5,
        output_base_dir="./models/self_training"
    )
    
    print("✓ Stage III completed!")
    
    # ========== EVALUATION ==========
    print("\n[Evaluation] Testing Final Model")
    print("-"*60)
    
    final_model = StructuredReasoning(
        model_path="./models/self_training/iteration_5",
        temperature=0.2
    )
    
    test_data = load_translation_dataset("data/test_set.jsonl")
    
    predictions = []
    test_cases_list = []
    
    for task in test_data:
        result = final_model.translate(task['source_code'])
        predictions.append(result['target_code'])
        test_cases_list.append(task['test_cases'])
    
    metrics = evaluate_translation(predictions, test_cases_list)
    print_evaluation_report(metrics)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    main()
