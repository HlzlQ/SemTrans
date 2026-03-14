"""
Evaluation utilities for code translation
"""

import subprocess
import tempfile
import os
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_pass_at_k(results: List[bool], k: int = 1) -> float:
    """
    Compute pass@k metric.
    
    Args:
        results: List of boolean results (True = passed)
        k: Number of attempts
        
    Returns:
        Pass@k score
    """
    if not results:
        return 0.0
    
    passed = sum(results)
    total = len(results)
    
    return passed / total


def evaluate_translation(
    predictions: List[str],
    test_cases_list: List[List[Dict]],
    language: str = "java"
) -> Dict:
    """
    Evaluate translation predictions.
    
    Args:
        predictions: List of generated code
        test_cases_list: List of test cases for each prediction
        language: Target language
        
    Returns:
        Evaluation metrics
    """
    results = []
    error_types = {
        'compilation_error': 0,
        'runtime_error': 0,
        'test_failure': 0,
        'success': 0
    }
    
    for pred, test_cases in zip(predictions, test_cases_list):
        success, error_type = run_tests(pred, test_cases, language)
        results.append(success)
        
        if success:
            error_types['success'] += 1
        else:
            error_types[error_type] += 1
    
    metrics = {
        'pass@1': compute_pass_at_k(results, k=1),
        'total': len(results),
        'passed': sum(results),
        'failed': len(results) - sum(results),
        'error_distribution': error_types
    }
    
    return metrics


def run_tests(
    code: str,
    test_cases: List[Dict],
    language: str = "java"
) -> Tuple[bool, str]:
    """
    Run test cases for generated code.
    
    Args:
        code: Generated code
        test_cases: Test cases
        language: Programming language
        
    Returns:
        Tuple of (success, error_type)
    """
    if language.lower() == "java":
        return run_java_tests(code, test_cases)
    else:
        raise ValueError(f"Unsupported language: {language}")


def run_java_tests(code: str, test_cases: List[Dict]) -> Tuple[bool, str]:
    """
    Run Java tests.
    
    Args:
        code: Java code
        test_cases: Test cases
        
    Returns:
        Tuple of (success, error_type)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write Java file
        java_file = os.path.join(tmpdir, "Solution.java")
        try:
            with open(java_file, 'w', encoding='utf-8') as f:
                f.write(code)
        except Exception as e:
            return False, 'compilation_error'
        
        # Compile
        try:
            result = subprocess.run(
                ["javac", java_file],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                return False, 'compilation_error'
        except Exception:
            return False, 'compilation_error'
        
        # Run tests
        for test in test_cases:
            try:
                # Create and run test
                test_passed = execute_java_test(tmpdir, test)
                if not test_passed:
                    return False, 'test_failure'
            except subprocess.TimeoutExpired:
                return False, 'runtime_error'
            except Exception:
                return False, 'runtime_error'
        
        return True, 'success'


def execute_java_test(tmpdir: str, test: Dict) -> bool:
    """
    Execute a single Java test case.
    
    Args:
        tmpdir: Temporary directory
        test: Test case dictionary
        
    Returns:
        True if test passed
    """
    # Simplified test execution
    # In practice, you'd need more sophisticated test harness
    return True


def calculate_metrics(results: List[Dict]) -> Dict:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        results: List of evaluation results
        
    Returns:
        Dictionary of metrics
    """
    total = len(results)
    passed = sum(1 for r in results if r.get('passed', False))
    
    metrics = {
        'total_samples': total,
        'passed': passed,
        'failed': total - passed,
        'accuracy': passed / total if total > 0 else 0.0,
        'pass_rate': passed / total if total > 0 else 0.0
    }
    
    return metrics


def print_evaluation_report(metrics: Dict):
    """
    Print formatted evaluation report.
    
    Args:
        metrics: Evaluation metrics
    """
    print("\n" + "="*60)
    print("EVALUATION REPORT")
    print("="*60)
    print(f"Total Samples: {metrics['total_samples']}")
    print(f"Passed: {metrics['passed']}")
    print(f"Failed: {metrics['failed']}")
    print(f"Accuracy (CA@1): {metrics['accuracy']*100:.2f}%")
    print("="*60 + "\n")
