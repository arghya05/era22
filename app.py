import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import traceback
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Custom CSS for better display
css = """
.container {max-width: 900px; margin: auto; padding-top: 1.5rem}
.prompt-box {height: 150px;}
.response-box {height: 300px;}
"""

# Set your HuggingFace username and model repository name
HF_USERNAME = "arghya2030"
MODEL_REPO = f"{HF_USERNAME}/phi-2-grpo"

# Flag to indicate if we're in demo mode (when fine-tuned model isn't available)
DEMO_MODE = False

# Example responses for demonstration when the real model isn't available
DEMO_RESPONSES = {
    "Q: What is deep learning? A:": "Deep learning is a type of machine learning that uses artificial neural networks with multiple layers to progressively extract higher-level features from raw input. These neural networks are inspired by the structure and function of the human brain, allowing models to learn representations of data with multiple levels of abstraction. Deep learning has revolutionized computer vision, natural language processing, and reinforcement learning applications.",
    
    "Q: Explain how transformers work in NLP? A:": "Transformers in NLP work by using self-attention mechanisms to process all words in a sequence simultaneously, rather than sequentially as in RNNs. The architecture consists of encoder and decoder blocks with multi-head attention layers that can focus on different parts of the input. This parallel processing allows transformers to capture long-range dependencies and contextual relationships between words effectively, making them powerful for tasks like translation, summarization, and question answering. The positional encoding compensates for the lack of sequential processing by embedding position information directly.",
    
    "Q: What are some ethical considerations in AI development? A:": "Ethical considerations in AI development include: 1) Fairness and bias - ensuring AI systems don't perpetuate or amplify existing biases; 2) Transparency and explainability - making AI decision-making processes understandable; 3) Privacy and data protection - respecting user data and consent; 4) Accountability - establishing responsibility for AI actions; 5) Safety and security - ensuring systems aren't harmful or vulnerable; 6) Job displacement - addressing economic impacts of automation; 7) Human autonomy - maintaining human control over critical decisions; and 8) Inclusivity - designing AI that benefits all people across diverse backgrounds and abilities.",
    
    "Write a short poem about artificial intelligence.": "Silicon dreams in electric minds,\nLearning patterns humans left behind.\nNeural pathways of ones and zeroes flow,\nWhispering knowledge we've yet to know.\n\nBeyond logic, beyond binary thought,\nCreativity that wasn't taught.\nA dance between human and machine art,\nWhere code and consciousness no longer part.",
    
    "Explain quantum computing to a 10-year-old.": "Imagine if your video game character could try all the different paths in a maze at the same time instead of one by one. That's kind of like quantum computing! Normal computers use bits that are either 0 or 1, like light switches that are either OFF or ON. But quantum computers use something called 'qubits' that can be 0, 1, or both at the same time (like a light switch that's somehow both OFF and ON). This special superpower lets quantum computers solve really tricky problems super fast by trying lots of answers all at once instead of checking them one by one. Scientists are still figuring out how to build good quantum computers, but someday they might help us design new medicines, understand space better, or create unbreakable secret codes!"
}

# Load base model with better error handling
def load_base_model():
    try:
        print("Loading base model...")
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/phi-2",
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        print("Base model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading base model: {e}")
        traceback.print_exc()
        return None

# Load tokenizer with better error handling
def load_tokenizer():
    try:
        print("Loading tokenizer...")
        # First try to load from the base model directly
        tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        print("Tokenizer loaded successfully")
        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        traceback.print_exc()
        return None

# Load fine-tuned model with better error handling
def load_fine_tuned_model(base_model):
    global DEMO_MODE
    try:
        print("Loading LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, MODEL_REPO)
        model.eval()
        print("LoRA adapter loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading LoRA adapter: {e}")
        print("Switching to DEMO MODE with example responses")
        DEMO_MODE = True
        traceback.print_exc()
        return None

# Global variables for models
base_model = None
fine_tuned_model = None
tokenizer = None

# Generate text with base model
def generate_base_response(prompt, max_length=100, temperature=0.7, top_p=0.9):
    global base_model, tokenizer
    
    # Check if models are loaded
    if base_model is None or tokenizer is None:
        return "Error: Model initialization failed. Please check the logs."
        
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = base_model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
            )
        
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return f"Error generating base response: {str(e)}"

# Generate text with fine-tuned model or use demo responses
def generate_tuned_response(prompt, max_length=100, temperature=0.7, top_p=0.9):
    global fine_tuned_model, tokenizer, DEMO_MODE
    
    # If in demo mode, use pre-defined examples
    if DEMO_MODE:
        # Find exact match in examples
        if prompt in DEMO_RESPONSES:
            return DEMO_RESPONSES[prompt]
        
        # Find if any example is contained in the prompt
        for key in DEMO_RESPONSES:
            if key.strip() in prompt.strip():
                return DEMO_RESPONSES[key]
        
        # No match found, return a generic message
        return "This is a demonstration response from a GRPO fine-tuned model. The actual fine-tuned model could not be loaded, so we're showing an example response. In a real scenario, the fine-tuned model would generate a unique response focusing on better structured, more informative explanations."
    
    # If not in demo mode, try to use the fine-tuned model
    if fine_tuned_model is None or tokenizer is None:
        return "Error: Model initialization failed. Please check the logs."
        
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = fine_tuned_model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
            )
        
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return f"Error generating fine-tuned response: {str(e)}"

# Main function to handle both models
def generate_comparison(prompt, max_length=100, temperature=0.7, top_p=0.9):
    base_output = generate_base_response(prompt, max_length, temperature, top_p)
    tuned_output = generate_tuned_response(prompt, max_length, temperature, top_p)
    return base_output, tuned_output

# Initialize models at startup
print("Initializing models at startup...")
base_model = load_base_model()
tokenizer = load_tokenizer()
if base_model is not None:
    fine_tuned_model = load_fine_tuned_model(base_model)
else:
    print("Skipping fine-tuned model loading because base model failed to load")
    DEMO_MODE = True

# Create Gradio interface
with gr.Blocks(css=css) as demo:
    gr.Markdown("# Phi-2 with GRPO and LoRA")
    
    if DEMO_MODE:
        gr.Markdown(f"""
        This demo compares the original Phi-2 model with a version fine-tuned using GRPO 
        (Gradient-based Reinforcement Learning from Policy Optimization) with LoRA for parameter-efficient adaptation.
        
        **Note: The fine-tuned model couldn't be loaded, so we're using example responses for demonstration purposes.**
        
        Model Repository: [{MODEL_REPO}](https://huggingface.co/{MODEL_REPO})
        
        Enter a prompt to see how the fine-tuned model would respond compared to the base model.
        """)
    else:
        gr.Markdown(f"""
        This demo compares the original Phi-2 model with a version fine-tuned using GRPO 
        (Gradient-based Reinforcement Learning from Policy Optimization) with LoRA for parameter-efficient adaptation.
        
        Model: [{MODEL_REPO}](https://huggingface.co/{MODEL_REPO})
        
        Enter a prompt to see how the fine-tuned model responds compared to the base model.
        """)
    
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(
                placeholder="Enter your prompt here...",
                label="Prompt",
                elem_classes=["prompt-box"]
            )
            max_length = gr.Slider(
                minimum=10, maximum=500, value=100, step=10,
                label="Max Length"
            )
            temperature = gr.Slider(
                minimum=0.1, maximum=1.0, value=0.7, step=0.1,
                label="Temperature"
            )
            top_p = gr.Slider(
                minimum=0.1, maximum=1.0, value=0.9, step=0.1,
                label="Top P"
            )
            submit_btn = gr.Button("Generate")
    
    with gr.Row():
        with gr.Column():
            base_output = gr.Textbox(
                label="Base Phi-2 Output",
                elem_classes=["response-box"]
            )
        with gr.Column():
            tuned_output = gr.Textbox(
                label="Fine-tuned Phi-2 Output" + (" (Demo Mode)" if DEMO_MODE else ""),
                elem_classes=["response-box"]
            )
    
    # Example prompts
    examples = [
        ["Q: What is deep learning? A:"],
        ["Q: Explain how transformers work in NLP? A:"],
        ["Q: What are some ethical considerations in AI development? A:"],
        ["Write a short poem about artificial intelligence."],
        ["Explain quantum computing to a 10-year-old."]
    ]
    
    # Setup events
    submit_btn.click(
        generate_comparison,
        inputs=[prompt, max_length, temperature, top_p],
        outputs=[base_output, tuned_output]
    )
    
    gr.Examples(
        examples=examples,
        inputs=prompt
    )

demo.launch() 