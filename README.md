# SemTrans: Execution-Aware Code Translation with Structured Reasoning and Iterative Self-Training

Official implementation of **SemTrans**, a three-stage framework for improving LLM-based code translation through execution semantics understanding, structured reasoning, and iterative self-training with execution feedback.

## 📖 Overview

SemTrans addresses three major challenges in cross-language code translation:
1. **Insufficient modeling of execution semantics** - Most models capture surface-level patterns rather than runtime behavior
2. **Lack of structured reasoning** - End-to-end generation lacks interpretability and explicit semantic alignment
3. **Scarcity of high-quality training data** - Limited parallel corpora with reasoning supervision

Our framework achieves **78.54%** average accuracy on Python→Java translation benchmarks, outperforming models with 10× more parameters.

## 🏗️ Architecture

SemTrans consists of three progressive stages:

### Stage I: Execution Semantics Warm-up
Trains the model to understand program execution through multi-task learning:
- **NL-to-code generation**: Aligns functional intent with implementations
- **Forward execution simulation**: Generates step-by-step execution traces
- **Backward abstraction reasoning**: Infers preconditions from outputs

### Stage II: Structured Reasoning for Translation
Decomposes translation into three explicit steps:
1. **Execution Semantics Understanding (R_exec)**: Summarizes runtime behavior
2. **Translation Planning (R_plan)**: Determines cross-language mapping strategy
3. **Target Code Generation (Y)**: Produces code following the plan

### Stage III: Self-Training with Execution Feedback
Iteratively improves through a generation-verification-learning loop:
- **Hierarchical sampling**: Low-temperature direct attempts + high-temperature exploration
- **Error-feedback repair**: Iterative correction using compilation/test feedback
- **Rationale reconstruction**: Regenerates reasoning for hard samples
- **Experience replay**: Prevents catastrophic forgetting

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/HlzlQ/SemTrans.git
cd SemTrans

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from semtrans.stages import StructuredReasoning

# Load trained model
reasoning = StructuredReasoning(
    model_path="./models/semtrans-7b",
    temperature=0.2
)

# Translate Python to Java
python_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""

result = reasoning.translate(python_code)

print("Execution Semantics:", result['r_exec'])
print("Translation Plan:", result['r_plan'])
print("Java Code:", result['target_code'])
```

## 📊 Training Pipeline

### Stage I: Execution Semantics Warm-up

```python
from semtrans.stages import ExecutionSemanticsWarmup
from semtrans.utils import load_pyx_dataset, format_for_training

# Initialize
warmup = ExecutionSemanticsWarmup(
    model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
    lambda_weight=1.0
)

# Load and prepare data
pyx_data = load_pyx_dataset("data/pyx_dataset.jsonl")
intent_data = format_for_training(pyx_data, task_type='intent')
forward_data = format_for_training(pyx_data, task_type='forward')
backward_data = format_for_training(pyx_data, task_type='backward')

# Train
warmup.train(
    intent_data=intent_data,
    forward_data=forward_data,
    backward_data=backward_data,
    output_dir="./models/stage1_warmup",
    num_epochs=2,
    batch_size=4
)
```

### Stage II: Structured Reasoning

```python
from semtrans.stages import StructuredReasoning

# Initialize with Stage I model
reasoning = StructuredReasoning(
    model_path="./models/stage1_warmup",
    temperature=0.2
)

# Translate with structured reasoning
result = reasoning.translate(python_code)
```

### Stage III: Self-Training

```python
from semtrans.stages import SelfTraining
from semtrans.utils import load_translation_dataset

# Initialize
self_training = SelfTraining(
    base_model_path="./models/stage1_warmup",
    structured_reasoning_module=reasoning,
    t_low=0.2,
    t_high=0.8,
    k_exploration=5,
    t_max_repair=3
)

# Load unlabeled tasks
unlabeled_tasks = load_translation_dataset("data/translation_tasks.jsonl")

# Run iterative self-training
self_training.run_self_training(
    unlabeled_tasks=unlabeled_tasks,
    mono_data=pyx_data,
    num_iterations=5,
    output_base_dir="./models/self_training"
)
```

## 📁 Project Structure

```
SemTrans/
├── semtrans/
│   ├── __init__.py
│   ├── stages/
│   │   ├── __init__.py
│   │   ├── stage1_warmup.py          # Execution semantics warm-up
│   │   ├── stage2_reasoning.py       # Structured reasoning
│   │   └── stage3_selftraining.py    # Self-training with feedback
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data_utils.py             # Data loading and processing
│   │   └── evaluation.py             # Evaluation metrics
│   └── models/                        # Model implementations
├── examples/
│   ├── run_stage1.py                 # Stage I example
│   ├── run_stage2.py                 # Stage II example
│   ├── run_stage3.py                 # Stage III example
│   └── complete_pipeline.py          # Full pipeline
├── configs/
│   └── default_config.yaml           # Configuration file
├── data/                              # Dataset directory
├── scripts/                           # Training scripts
├── requirements.txt
└── README.md
```

## 📈 Results

Performance on Python→Java translation benchmarks:

| Model | Size | HumanEval-X | TransCoder | CodeNet | Avg. |
|-------|------|-------------|------------|---------|------|
| Qwen2.5-Coder-7B (Base) | 7B | 64.02% | 72.61% | 56.50% | 64.38% |
| Qwen2.5-Coder-32B-Instruct | 32B    | 73.17%      | 80.50%     | 68.00%     | 73.89%     |
| GLM-4.7-Flash              | 30B    | 76.22%      | 81.74%     | 72.00%     | 76.65%     |
| Kimi-Dev-72B               | 72B    | 77.44%      | **85.89%** | 72.00%     | 78.44%     |
| **SemTrans (Ours)** | **7B** | **78.05%** | 85.06% | **72.50%** | **78.54%** |
| *Improvement over Base* | - | *+14.03%* | *+12.45%* | *+16.00%* | *+14.16%* |

## 🔬 Key Features

- **Execution-Aware**: Models runtime behavior, not just syntax
- **Interpretable**: Explicit reasoning steps for debugging and verification
- **Data-Efficient**: Automatically synthesizes training data from unlabeled code
- **Robust**: Hierarchical sampling + error-feedback repair for hard samples
- **Scalable**: Works with 7B models, outperforms 70B baselines

## 📦 Datasets

We provide two datasets:

1. **D1 (Execution Semantics Warm-up)**: 32,000 Python samples with execution traces
   - Download: [PyX Dataset](https://huggingface.co/datasets/semcoder/PyX)

2. **D2 (Self-Training)**: 7,447 filtered Python-Java pairs with unit tests
   - Download: [SemTrans-D2](https://huggingface.co/datasets/HlzlQ/SemTrans-D2)

## 🛠️ Configuration

Edit `configs/default_config.yaml` to customize:

```yaml
model:
  base_model: "Qwen/Qwen2.5-Coder-7B-Instruct"
  device: "cuda"

stage1:
  lambda_weight: 1.0
  num_epochs: 2
  batch_size: 4

stage3:
  num_iterations: 5
  t_low: 0.2
  t_high: 0.8
  k_exploration: 5
  t_max_repair: 3
```

## 📝 Citation

If you use SemTrans in your research, please cite:

```bibtex
@article{han2026semtrans,
  title={SemTrans: Execution-Aware Code Translation with Structured Reasoning and Iterative Self-Training},
  author={Han, Qi and Guang, Yan and Shu, Hui and Kang, Fei},
  journal={Journal Title},
  year={2026}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Based on the [Qwen2.5-Coder](https://github.com/QwenLM/Qwen2.5-Coder) model family
- Execution trace data from [PyX Dataset](https://huggingface.co/datasets/semcoder/PyX)
- Evaluation benchmarks: HumanEval-X, TransCoder, CodeNet

## 📧 Contact

For questions or issues, please:
- Open an issue on GitHub

## 🔗 Links

- **Dataset**: [HuggingFace](https://huggingface.co/datasets/HlzlQ/SemTrans-Dataset)
- **Model**: [ModelScope](https://www.modelscope.cn/models/JacksonQi/SemTrans)
- **GitHub**: [https://github.com/HlzlQ/SemTrans](https://github.com/HlzlQ/SemTrans)

---

**Note**: This implementation requires GPU resources for training. 
