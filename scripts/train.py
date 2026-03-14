#!/usr/bin/env python3
"""
Training script for SemTrans
"""

import argparse
import yaml
from pathlib import Path

from semtrans.stages import ExecutionSemanticsWarmup, StructuredReasoning, SelfTraining
from semtrans.utils import (
    load_pyx_dataset,
    load_translation_dataset
)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_stage1(config: dict):
    """Train Stage I: Execution Semantics Warm-up."""
    print("\n" + "="*60)
    print("STAGE I: EXECUTION SEMANTICS WARM-UP")
    print("="*60 + "\n")
    
    warmup = ExecutionSemanticsWarmup(
        model_name=config['model']['base_model'],
        lambda_weight=config['stage1']['lambda_weight']
    )
    
    # Load PyX dataset
    print("Loading PyX dataset...")
    pyx_data = load_pyx_dataset(config['data']['pyx_dataset'])
    print(f"Loaded {len(pyx_data)} samples from PyX dataset")
    
    # Train directly on PyX data
    # The model will automatically create three types of tasks:
    # 1. NL-to-code generation (intent alignment)
    # 2. Forward execution simulation
    # 3. Backward abstraction reasoning
    warmup.train(
        pyx_data=pyx_data,
        output_dir=config['stage1']['output_dir'],
        num_epochs=config['stage1']['num_epochs'],
        batch_size=config['stage1']['batch_size'],
        learning_rate=config['stage1']['learning_rate'],
        save_steps=config['stage1']['save_steps']
    )
    
    print("\n✓ Stage I completed!")


def train_stage3(config: dict):
    """Train Stage III: Self-Training with Execution Feedback."""
    print("\n" + "="*60)
    print("STAGE III: SELF-TRAINING WITH EXECUTION FEEDBACK")
    print("="*60 + "\n")
    
    # Initialize Stage II
    reasoning = StructuredReasoning(
        model_path=config['stage1']['output_dir'],
        temperature=config['stage2']['temperature']
    )
    
    # Initialize Stage III
    self_training = SelfTraining(
        base_model_path=config['stage1']['output_dir'],
        structured_reasoning_module=reasoning,
        t_low=config['stage3']['t_low'],
        t_high=config['stage3']['t_high'],
        k_exploration=config['stage3']['k_exploration'],
        t_max_repair=config['stage3']['t_max_repair']
    )
    
    # Load data
    unlabeled_tasks = load_translation_dataset(config['data']['translation_tasks'])
    mono_data = load_pyx_dataset(config['data']['pyx_dataset'])
    
    # Run self-training
    self_training.run_self_training(
        unlabeled_tasks=unlabeled_tasks,
        mono_data=mono_data,
        num_iterations=config['stage3']['num_iterations'],
        output_base_dir=config['stage3']['output_dir']
    )
    
    print("\n✓ Stage III completed!")


def main():
    parser = argparse.ArgumentParser(description="Train SemTrans model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["1", "3", "all"],
        default="all",
        help="Which stage to train (1, 3, or all)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Train specified stages
    if args.stage in ["1", "all"]:
        train_stage1(config)
    
    if args.stage in ["3", "all"]:
        train_stage3(config)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)


if __name__ == "__main__":
    main()
