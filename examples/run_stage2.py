"""
Example: Running Stage II - Structured Reasoning Translation
"""

from semtrans.stages import StructuredReasoning
from semtrans.utils import save_results

# Initialize Stage II with trained model from Stage I
reasoning = StructuredReasoning(
    model_path="./models/stage1_warmup",
    temperature=0.2
)

# Example Python code to translate
python_code = """
def process_list(items):
    result = []
    for item in items:
        if isinstance(item, int):
            result.append(item * 2)
        else:
            result.append(item.upper())
    return result
"""

# Translate with structured reasoning
result = reasoning.translate(python_code)

print("="*60)
print("STRUCTURED TRANSLATION RESULT")
print("="*60)

print("\nStep 1 - Execution Semantics Understanding:")
print(result['r_exec'])

print("\nStep 2 - Translation Planning:")
print(result['r_plan'])

print("\nStep 3 - Generated Java Code:")
print(result['target_code'])

# Batch translation
python_codes = [
    "def add(a, b): return a + b",
    "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
    "def reverse_string(s): return s[::-1]"
]

results = reasoning.batch_translate(python_codes, batch_size=2)

# Save results
save_results(results, "outputs/stage2_translations.jsonl")

print("\nBatch translation completed!")
print(f"Results saved to: outputs/stage2_translations.jsonl")
