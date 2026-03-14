"""
Utility functions for data processing
"""

import json
import os
from typing import Dict, List, Optional
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_pyx_dataset(data_path: str) -> List[Dict]:
    """
    Load PyX dataset for execution semantics warm-up.
    
    PyX dataset format:
    {
        "task_id": str,
        "prompt": str,  # Natural language description
        "code": str,    # Python code
        "test_list": List[str],  # Test cases
        "monologue": str  # Execution trace/reasoning
    }
    
    Args:
        data_path: Path to PyX dataset (JSONL format)
        
    Returns:
        List of PyX samples
    """
    logger.info(f"Loading PyX dataset from: {data_path}")
    
    samples = []
    
    if not os.path.exists(data_path):
        logger.warning(f"Dataset file not found: {data_path}")
        logger.info("Please download PyX dataset from: https://huggingface.co/datasets/semcoder/PyX")
        return samples
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                sample = json.loads(line)
                samples.append(sample)
    
    logger.info(f"Loaded {len(samples)} samples from PyX dataset")
    return samples


def load_translation_dataset(data_path: str) -> List[Dict]:
    """
    Load translation dataset with source code and test cases.
    
    Args:
        data_path: Path to translation dataset
        
    Returns:
        List of translation tasks
    """
    logger.info(f"Loading translation dataset from: {data_path}")
    
    tasks = []
    
    if not os.path.exists(data_path):
        logger.warning(f"Dataset file not found: {data_path}")
        return tasks
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                task = json.loads(line)
                tasks.append(task)
    
    logger.info(f"Loaded {len(tasks)} translation tasks")
    return tasks


def save_results(results: List[Dict], output_path: str):
    """
    Save translation results to file.
    
    Args:
        results: List of translation results
        output_path: Output file path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    logger.info(f"Saved {len(results)} results to: {output_path}")


def filter_by_test_coverage(
    samples: List[Dict],
    min_tests: int = 3
) -> List[Dict]:
    """
    Filter samples by minimum test coverage.
    
    Args:
        samples: List of samples
        min_tests: Minimum number of test cases required
        
    Returns:
        Filtered samples
    """
    filtered = [s for s in samples if len(s.get('test_cases', [])) >= min_tests]
    logger.info(f"Filtered {len(samples)} -> {len(filtered)} samples (min_tests={min_tests})")
    return filtered


def decontaminate_dataset(
    train_samples: List[Dict],
    test_samples: List[Dict],
    similarity_threshold: float = 0.8
) -> List[Dict]:
    """
    Remove samples from training set that are too similar to test set.
    
    Args:
        train_samples: Training samples
        test_samples: Test samples
        similarity_threshold: Similarity threshold for filtering
        
    Returns:
        Decontaminated training samples
    """
    logger.info("Decontaminating dataset...")
    
    # Simple implementation - in practice, use MinHash LSH and CodeBERT embeddings
    test_codes = {s.get('source_code', s.get('code', '')) for s in test_samples}
    
    decontaminated = []
    for sample in train_samples:
        sample_code = sample.get('source_code', sample.get('code', ''))
        if sample_code not in test_codes:
            decontaminated.append(sample)
    
    logger.info(f"Decontaminated {len(train_samples)} -> {len(decontaminated)} samples")
    return decontaminated


def split_dataset(
    samples: List[Dict],
    train_ratio: float = 0.9
) -> tuple:
    """
    Split dataset into train and validation sets.
    
    Args:
        samples: List of samples
        train_ratio: Ratio of training samples
        
    Returns:
        Tuple of (train_samples, val_samples)
    """
    import random
    random.shuffle(samples)
    
    split_idx = int(len(samples) * train_ratio)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]
    
    logger.info(f"Split dataset: {len(train_samples)} train, {len(val_samples)} val")
    return train_samples, val_samples
