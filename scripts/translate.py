#!/usr/bin/env python3
"""
Inference script for SemTrans
"""

import argparse
from semtrans.stages import StructuredReasoning


def main():
    parser = argparse.ArgumentParser(description="Translate code using SemTrans")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model"
    )
    parser.add_argument(
        "--source_file",
        type=str,
        help="Path to source code file"
    )
    parser.add_argument(
        "--source_code",
        type=str,
        help="Source code as string"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--show_reasoning",
        action="store_true",
        help="Show intermediate reasoning steps"
    )
    
    args = parser.parse_args()
    
    # Get source code
    if args.source_file:
        with open(args.source_file, 'r', encoding='utf-8') as f:
            source_code = f.read()
    elif args.source_code:
        source_code = args.source_code
    else:
        print("Error: Must provide either --source_file or --source_code")
        return
    
    # Load model
    print("Loading model...")
    reasoning = StructuredReasoning(
        model_path=args.model_path,
        temperature=args.temperature
    )
    
    # Translate
    print("\nTranslating...\n")
    result = reasoning.translate(source_code)
    
    # Display results
    print("="*60)
    print("TRANSLATION RESULT")
    print("="*60)
    
    if args.show_reasoning:
        print("\n[Step 1] Execution Semantics Understanding:")
        print("-"*60)
        print(result['r_exec'])
        
        print("\n[Step 2] Translation Planning:")
        print("-"*60)
        print(result['r_plan'])
        
        print("\n[Step 3] Generated Code:")
        print("-"*60)
    
    print(result['target_code'])
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
