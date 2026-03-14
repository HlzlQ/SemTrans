#!/usr/bin/env python3
"""
Evaluation script for SemTrans
"""

import argparse
import json
from pathlib import Path

from semtrans.stages import StructuredReasoning
from semtrans.utils import (
    load_translation_dataset,
    evaluate_translation,
    print_evaluation_report,
    save_results
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate SemTrans model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        required=True,
        help="Path to test dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory to save results"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for inference"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("SEMTRANS EVALUATION")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Test data: {args.test_data}")
    print("="*60 + "\n")
    
    # Load model
    print("Loading model...")
    reasoning = StructuredReasoning(
        model_path=args.model_path,
        temperature=args.temperature
    )
    
    # Load test data
    print("Loading test data...")
    test_data = load_translation_dataset(args.test_data)
    print(f"Loaded {len(test_data)} test samples\n")
    
    # Run translation
    print("Running translation...")
    predictions = []
    test_cases_list = []
    all_results = []
    
    for i, task in enumerate(test_data):
        if (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/{len(test_data)}")
        
        result = reasoning.translate(task['source_code'])
        predictions.append(result['target_code'])
        test_cases_list.append(task.get('test_cases', []))
        
        all_results.append({
            'source_code': task['source_code'],
            'r_exec': result['r_exec'],
            'r_plan': result['r_plan'],
            'target_code': result['target_code']
        })
    
    # Evaluate
    print("\nEvaluating translations...")
    metrics = evaluate_translation(predictions, test_cases_list)
    
    # Print report
    print_evaluation_report(metrics)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / "translation_results.jsonl"
    save_results(all_results, str(results_file))
    
    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Translations: {results_file}")
    print(f"  - Metrics: {metrics_file}")


if __name__ == "__main__":
    main()
