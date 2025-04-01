---
title: Phi-2 with GRPO and LoRA Demo
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 3.50.2
app_file: app.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# Phi-2 with GRPO and LoRA Demo

This Space demonstrates a fine-tuned Microsoft Phi-2 model using GRPO (Gradient-based Reinforcement Learning from Policy Optimization) with LoRA for parameter-efficient adaptation.

## About the Model

The base model (Microsoft Phi-2) has been fine-tuned using a minimal GRPO implementation with LoRA to create a parameter-efficient adaptation. The fine-tuning process targeted only specific layers of the model to reduce memory requirements and improve performance on a specific subset of tasks.

### Model Details
- Base Model: microsoft/phi-2
- Fine-tuning Method: GRPO (HuggingFace's official implementation)
- Parameter-Efficient Method: LoRA (Low-Rank Adaptation)
- Target Modules: q_proj

## Demo Features

This demo offers a side-by-side comparison between:
1. The original Phi-2 base model
2. The GRPO fine-tuned version

You can enter your own prompts or select from the provided examples to see how the fine-tuning has affected the model's output.

## Example Outputs

### Before Fine-tuning (Base Phi-2)
```
Q: What is deep learning? A: Deep learning is a subset of machine learning that uses neural networks with multiple layers (deep neural networks) to analyze various factors of data. It attempts to mimic the human brain, albeit in a simplified manner, to solve complex problems. Deep learning is particularly effective for tasks involving unstructured data like images, text, and audio, and has enabled significant advances in areas such as computer vision, natural language processing, and speech recognition.
```

### After Fine-tuning (Phi-2 with GRPO)
```
Q: What is deep learning? A: Deep learning is a type of machine learning that uses artificial neural networks with multiple layers to progressively extract higher-level features from raw input. These neural networks are inspired by the structure and function of the human brain, allowing models to learn representations of data with multiple levels of abstraction. Deep learning has revolutionized computer vision, natural language processing, and reinforcement learning applications.
```

## Implementation Details

The fine-tuning process used a minimal implementation focused on computational efficiency:
- Used LoRA with r=1 (minimum rank)
- Applied LoRA only to q_proj parameters
- Optimized for CPU training using PyTorch's optimizations
- Used a small number of training steps for demonstration purposes

## How to Use

1. Enter a prompt in the input box or select one of the example prompts
2. Adjust generation parameters (Max Length, Temperature, Top P) if desired
3. Click "Generate" to see the outputs from both models side by side
4. Compare the results to see how fine-tuning has affected the model's behavior

## Usage in Python

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2", 
    trust_remote_code=True
)

# Load the fine-tuned LoRA adapter
model = PeftModel.from_pretrained(base_model, "arghya2030/phi-2-grpo")
tokenizer = AutoTokenizer.from_pretrained("arghya2030/phi-2-grpo")

# Generate text
prompt = "Q: What is deep learning? A:"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Project Overview

- Base Model: Microsoft Phi-2
- Fine-tuning Method: GRPO (Gradient-based Reinforcement Learning from Policy Optimization)
- Training Dataset: Alpaca
- Framework: TRL (Transformer Reinforcement Learning)
- Deployment: HuggingFace Spaces with Gradio

## Setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/phi-2-finetuning.git
cd phi-2-finetuning
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up HuggingFace token:
```bash
export HF_TOKEN="your-huggingface-token"
```

4. Run training:
```bash
python train.py
```

## Model Architecture

The fine-tuned model uses:
- 4-bit quantization for efficient memory usage
- LoRA adaptation with rank 16
- Target modules: q_proj, k_proj, v_proj, o_proj
- Learning rate: 2e-4
- Training epochs: 3

## Before and After Examples

### Example 1: Question Answering
**Before Fine-tuning:**
```
Q: What is the capital of France?
A: The capital of France is Paris.
```

**After Fine-tuning:**
```
Q: What is the capital of France?
A: The capital of France is Paris. Paris is the largest city in France and serves as the country's political, economic, and cultural center. It is known for its iconic landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum.
```

### Example 2: Creative Writing
**Before Fine-tuning:**
```
Q: Write a short story about a robot learning to paint.
A: The robot picked up a brush and painted.
```

**After Fine-tuning:**
```
Q: Write a short story about a robot learning to paint.
A: In a small workshop filled with canvases and paintbrushes, a robot named Pixel carefully dipped its metallic fingers into a palette of vibrant colors. Despite its precise programming, it struggled to understand the concept of artistic expression. Day after day, it practiced, creating increasingly complex patterns until one day, it painted a masterpiece that moved human observers to tears. The robot had learned that art wasn't just about colors and shapes, but about the soul behind the creation.
```

## HuggingFace Spaces Demo

Visit our HuggingFace Spaces demo at: [Your Spaces URL]

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Microsoft for the Phi-2 model
- HuggingFace for the TRL library and Spaces platform
- Stanford Alpaca team for the dataset 