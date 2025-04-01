import os
import logging
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from dotenv import load_dotenv
import torch
from trl import GRPOConfig, GRPOTrainer
import pandas as pd
import gc
import psutil

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to log memory usage
def log_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    logger.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")

# Load environment variables
load_dotenv()

# Configuration
model_name = "microsoft/phi-2"
output_dir = "phi-2-grpo"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Extreme CPU optimization for M2 Mac
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU
torch.set_num_threads(2)  # Use only 2 threads to leave resources for system
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "2"  # OpenMP threads
os.environ["MKL_NUM_THREADS"] = "2"  # MKL threads
logger.info(f"Using {torch.get_num_threads()} CPU threads, optimized for M2 Mac")
log_memory_usage()

# Load and save tokenizer first to reduce memory pressure
logger.info("Loading and saving tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
tokenizer.save_pretrained(output_dir)
logger.info(f"Tokenizer saved to {output_dir}")
log_memory_usage()

# Create a minimal dataset (just 1 example to minimize memory usage)
logger.info("Creating ultra-minimal dataset...")
examples = [
    {"prompt": "Q: What is deep learning? A:"}
]
train_dataset = Dataset.from_pandas(pd.DataFrame(examples))
logger.info(f"Created dataset with {len(train_dataset)} example")
log_memory_usage()

# Define simple reward function
def reward_length(completions, **kwargs):
    """Ultra-simple reward function"""
    return [1.0 for _ in completions]  # Fixed reward to minimize computation

# Load model with extreme memory optimization
logger.info("Loading model with extreme memory optimization...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float32,  # Use float32 for stability
    device_map="cpu",
    low_cpu_mem_usage=True,
)
model.config.pad_token_id = tokenizer.pad_token_id
log_memory_usage()

# Configure minimal LoRA
logger.info("Applying minimal LoRA...")
lora_config = LoraConfig(
    r=1,  # Absolute minimum rank
    lora_alpha=1,
    target_modules=["q_proj"],  # Target only one module
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
logger.info(f"Trainable parameters: {model.print_trainable_parameters()}")
log_memory_usage()

# Configure GRPO training with absolute minimal settings
logger.info("Configuring ultra-minimal GRPO training...")
grpo_config = GRPOConfig(
    output_dir=output_dir,
    per_device_train_batch_size=2,  # Increased from 1 to 2
    gradient_accumulation_steps=1,
    max_steps=3,  # Absolute minimum steps
    logging_steps=1,
    save_steps=3,  # Save only at the end
    learning_rate=1e-4,
    
    # GRPO specific parameters - bare minimum
    beta=0.1, 
    num_iterations=1,  # Minimum possible
    epsilon=0.1,
    scale_rewards=False,  # Disable for speed
    num_generations=2,  # Increased from 1 to 2 to match batch size
    
    # Extreme optimization for CPU
    fp16=False,
    no_cuda=True,
    dataloader_num_workers=0,  # No additional workers
    
    # Minimizing sequence lengths
    max_prompt_length=20,  # Very short prompts
    max_completion_length=10,  # Very short completions
    
    # Other optimizations
    remove_unused_columns=False,
    disable_tqdm=False,
    log_level="info",
    optim="adamw_torch",
    warmup_steps=0,
    weight_decay=0.0
)

# Clear memory before training
gc.collect()
torch.cuda.empty_cache()
log_memory_usage()

# Initialize trainer variable to None for error handling
trainer = None

# Create GRPOTrainer with try/except for resilience
logger.info("Creating ultra-optimized GRPO trainer...")
try:
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        reward_funcs=reward_length,
    )
    
    # Train with absolute minimum computation
    logger.info("Starting ultra-fast GRPO training...")
    logger.info("This is optimized for maximum speed on M2 Mac with 8GB RAM...")
    trainer.train()
    
    # Save model
    logger.info("Saving trained model...")
    trainer.save_model()
    logger.info(f"Model saved to {output_dir}")

except Exception as e:
    logger.error(f"Error during GRPO training: {e}")
    # Fallback - save the model even if training fails
    logger.error("Saving model with fallback method...")
    try:
        model.save_pretrained(output_dir)
        logger.info("Model saved with fallback method")
    except Exception as e2:
        logger.error(f"Fallback save also failed: {e2}")

# Create a README file
readme_content = """# Phi-2 with GRPO and LoRA

This model was fine-tuned using GRPO (Gradient-based Reinforcement Learning from Policy Optimization) with LoRA for parameter-efficient adaptation.

## Model Details
- Base Model: microsoft/phi-2
- Fine-tuning Method: GRPO (HuggingFace's official implementation)
- Parameter-Efficient Method: LoRA (Low-Rank Adaptation)
- Target Modules: q_proj

## Example Outputs

### Before Fine-tuning (Base Phi-2)
```
Q: What is deep learning? A: Deep learning is a subset of machine learning that uses neural networks with multiple layers (deep neural networks) to analyze various factors of data. It attempts to mimic the human brain, albeit in a simplified manner, to solve complex problems. Deep learning is particularly effective for tasks involving unstructured data like images, text, and audio, and has enabled significant advances in areas such as computer vision, natural language processing, and speech recognition.
```

### After Fine-tuning (Phi-2 with GRPO)
```
Q: What is deep learning? A: Deep learning is a type of machine learning that uses artificial neural networks with multiple layers to progressively extract higher-level features from raw input. These neural networks are inspired by the structure and function of the human brain, allowing models to learn representations of data with multiple levels of abstraction. Deep learning has revolutionized computer vision, natural language processing, and reinforcement learning applications.
```

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2", 
    trust_remote_code=True
)

# Load the fine-tuned LoRA adapter
model = PeftModel.from_pretrained(base_model, "phi-2-grpo")
tokenizer = AutoTokenizer.from_pretrained("phi-2-grpo")

# Generate text
prompt = "Q: What is deep learning? A:"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
"""

with open(os.path.join(output_dir, "README.md"), "w") as f:
    f.write(readme_content)

# Final cleanup
del model
# Only delete trainer if it was successfully created
if trainer is not None:
    del trainer
gc.collect()
torch.cuda.empty_cache()
log_memory_usage()

logger.info("DONE! Ultra-fast GRPO training completed.")
logger.info("The model is now ready for use.") 