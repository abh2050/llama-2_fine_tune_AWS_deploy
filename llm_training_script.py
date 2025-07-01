#!/usr/bin/env python
# coding: utf-8

# In[20]:


# # LLaMA-2 Fine-tuning with LoRA using Claude-style Instruction Data

# This notebook demonstrates how to fine-tune a LLaMA-2 model using LoRA (Low-Rank Adaptation) with Claude-4 style instruction datasets. We'll cover:

# 1. **Dataset Preparation**: Converting Claude-style conversations to LLaMA format
# 2. **Model Setup**: Loading LLaMA-2 with LoRA configuration
# 3. **Training**: Fine-tuning with optimized parameters
# 4. **Evaluation**: Testing the fine-tuned model

# ## Prerequisites
# - CUDA-compatible GPU (recommended: 16GB+ VRAM)
# - Hugging Face account with LLaMA-2 access
# - Claude-style instruction dataset


# In[21]:


# Install required packages
get_ipython().system('pip install -q transformers datasets peft trl accelerate bitsandbytes wandb')
get_ipython().system('pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118')


# ## 📚 About the OpenAssistant-Guanaco Dataset
# 
# We're using a **500-example subset** of the **timdettmers/openassistant-guanaco** dataset, which is a high-quality instruction-following dataset:
# 
# ### Dataset Features:
# - **Original Size**: ~10,000 high-quality conversations
# - **Used Size**: 500 examples (for optimal training speed)
# - **Format**: Human-Assistant dialogue pairs
# - **Quality**: Curated and filtered for instruction-following tasks
# - **Source**: Based on OpenAssistant conversations
# - **Preprocessing**: Already formatted with `### Human:` and `### Assistant:` markers
# 
# ### Why This Dataset Size:
# 1. **Training Speed**: 500 examples provide fast training iterations
# 2. **Quality over Quantity**: Still maintains high-quality instruction examples
# 3. **Demonstration**: Perfect size for learning and experimentation
# 4. **Resource Efficient**: Works well on limited computational resources
# 5. **Good Performance**: Sufficient for effective fine-tuning
# 
# ### Training Benefits:
# - **Faster Iterations**: Complete training in minutes rather than hours
# - **Easy Experimentation**: Quick to test different hyperparameters
# - **Resource Friendly**: Lower memory and compute requirements
# - **Still Effective**: 500 high-quality examples can significantly improve model performance
# 
# This subset provides an excellent balance between training speed and model quality improvement.

# In[22]:


import json
import torch
import pandas as pd
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
import wandb
import os
from typing import List, Dict, Any

# Check if CUDA is available
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")


# In[23]:


# ✅ Step 1: Load OpenAssistant-Guanaco Dataset

print("Loading timdettmers/openassistant-guanaco dataset...")
print("This is a high-quality instruction-following dataset with ~10k conversations")

# Load the OpenAssistant-Guanaco dataset
try:
    # Load the dataset from Hugging Face Hub
    guanaco_dataset = load_dataset("timdettmers/openassistant-guanaco")
    
    print(f"✅ Dataset loaded successfully!")
    print(f"Dataset structure: {guanaco_dataset}")
    
    # The dataset has a 'train' split
    raw_dataset = guanaco_dataset['train']
    print(f"Total examples: {len(raw_dataset)}")
    
    # Show sample data structure
    print("\nSample data structure:")
    sample = raw_dataset[0]
    print(f"Keys: {list(sample.keys())}")
    print(f"Sample text length: {len(sample['text'])}")
    print(f"First 200 characters: {sample['text'][:200]}...")
    
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    print("Falling back to sample data...")
    
    # Fallback sample data
    sample_data = [
        {"text": "### Human: What is quantum entanglement?\n\n### Assistant: Quantum entanglement is a phenomenon in quantum mechanics where two or more particles become correlated in such a way that the quantum state of each particle cannot be described independently, even when separated by large distances."},
        {"text": "### Human: How do I write a Python function to calculate factorial?\n\n### Assistant: Here's a simple recursive approach:\n\n```python\ndef factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    return n * factorial(n - 1)\n```"},
        {"text": "### Human: Explain machine learning in simple terms.\n\n### Assistant: Machine learning is like teaching a computer to recognize patterns and make predictions, similar to how humans learn from experience."}
    ]
    
    raw_dataset = Dataset.from_list(sample_data)
    print(f"Using fallback dataset with {len(raw_dataset)} examples")


# In[24]:


# ✅ Step 2: Dataset Preparation for OpenAssistant-Guanaco

def clean_and_filter_dataset(dataset, max_length=2048, min_length=50):
    """Clean and filter the dataset for better training quality"""
    
    def is_valid_example(example):
        text = example['text']
        # Filter out very short or very long examples
        if len(text) < min_length or len(text) > max_length:
            return False
        # Ensure it has the expected format
        if "### Human:" not in text or "### Assistant:" not in text:
            return False
        return True
    
    # Filter the dataset
    filtered_dataset = dataset.filter(is_valid_example)
    
    print(f"Original dataset size: {len(dataset)}")
    print(f"Filtered dataset size: {len(filtered_dataset)}")
    
    return filtered_dataset

def preprocess_guanaco_text(example):
    """Preprocess the text to ensure consistent formatting"""
    text = example['text']
    
    # Ensure proper spacing and formatting
    text = text.replace("###Human:", "### Human:")
    text = text.replace("###Assistant:", "### Assistant:")
    text = text.replace("### Human :", "### Human:")
    text = text.replace("### Assistant :", "### Assistant:")
    
    # Add proper ending if missing
    if not text.endswith('\n'):
        text += '\n'
    
    return {'text': text}

print("Processing OpenAssistant-Guanaco dataset...")

# Clean and filter the dataset
cleaned_dataset = clean_and_filter_dataset(raw_dataset)

# Apply preprocessing
processed_dataset = cleaned_dataset.map(preprocess_guanaco_text)

print(f"Final processed dataset size: {len(processed_dataset)}")

# Show examples of the processed data
print("\n=== Sample Processed Examples ===")
for i in range(min(3, len(processed_dataset))):
    print(f"\nExample {i+1}:")
    text = processed_dataset[i]['text']
    print(text[:300] + ("..." if len(text) > 300 else ""))
    print("-" * 50)


# In[25]:


# Load and process the OpenAssistant-Guanaco dataset
print("Setting up training dataset...")

# Use the processed dataset
hf_dataset = processed_dataset

print(f"Full dataset size: {len(hf_dataset)}")

# Limit dataset to 500 examples for faster training and demonstration
MAX_EXAMPLES = 500
if len(hf_dataset) > MAX_EXAMPLES:
    print(f"Limiting dataset to {MAX_EXAMPLES} examples for faster training...")
    hf_dataset = hf_dataset.select(range(MAX_EXAMPLES))
    print(f"✅ Dataset reduced from {len(processed_dataset)} to {len(hf_dataset)} examples")
else:
    print(f"Dataset size ({len(hf_dataset)}) is already within the limit")

print(f"\nFinal dataset size: {len(hf_dataset)} examples")
print("\nSample formatted text:")
print(hf_dataset[0]["text"][:400] + "...")

print(f"\n📊 Training with {len(hf_dataset)} examples for optimal training speed")


# In[26]:


# ✅ Create Train/Validation Split for 500-example Dataset

print("Creating train/validation split for limited dataset...")

# For the 500-example dataset, use 80/20 split
test_size = 0.2
print(f"Using {test_size:.0%} split (80% train, 20% validation)")

# Split the dataset into train and validation
train_test_split = hf_dataset.train_test_split(test_size=test_size, seed=42)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

print(f"Training dataset size: {len(train_dataset)} examples")
print(f"Evaluation dataset size: {len(eval_dataset)} examples")

# Show statistics about the datasets
def analyze_dataset(dataset, name):
    lengths = [len(example['text']) for example in dataset]
    avg_length = sum(lengths) / len(lengths)
    max_length = max(lengths)
    min_length = min(lengths)
    
    print(f"\n{name} Dataset Statistics:")
    print(f"  Examples: {len(dataset)}")
    print(f"  Avg text length: {avg_length:.0f} characters")
    print(f"  Min text length: {min_length} characters")
    print(f"  Max text length: {max_length} characters")

analyze_dataset(train_dataset, "Training")
analyze_dataset(eval_dataset, "Evaluation")

print(f"\n✅ Dataset split complete:")
print(f"  • Training: {len(train_dataset)} examples")
print(f"  • Evaluation: {len(eval_dataset)} examples")
print(f"  • Total: {len(hf_dataset)} examples")

print(f"\nSample evaluation text:")
print(eval_dataset[0]["text"][:300] + "...")


# In[27]:


# ✅ Step 3: Model Configuration

# Model configuration
MODEL_NAME = "meta-llama/Llama-2-7b-hf"  # Change to "meta-llama/Llama-2-7b-chat-hf" for chat version
OUTPUT_DIR = "./llama2-claude-lora"

# Quantization config for memory efficiency
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# LoRA configuration
lora_config = LoraConfig(
    r=16,  # Rank of adaptation
    lora_alpha=32,  # LoRA scaling parameter
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

print("Configuration:")
print(f"Model: {MODEL_NAME}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"LoRA rank: {lora_config.r}")
print(f"LoRA alpha: {lora_config.lora_alpha}")
print(f"Target modules: {lora_config.target_modules}")


# In[28]:


# ✅ CPU-Friendly Configuration (for environments without CUDA)

# Check if CUDA is available and adjust configuration accordingly
if not torch.cuda.is_available():
    print("⚠️  CUDA not available - using CPU-friendly configuration")
    print("Note: For production training, use a GPU environment")
    
    # Use a smaller model for CPU demonstration
    MODEL_NAME = "microsoft/DialoGPT-small"  # Much smaller model for CPU
    OUTPUT_DIR = "./dialogpt-claude-lora-cpu"
    
    # No quantization for CPU
    bnb_config = None
    
    # Simpler LoRA config for smaller model
    lora_config = LoraConfig(
        r=8,  # Smaller rank for CPU
        lora_alpha=16,
        target_modules=["c_attn"],  # DialoGPT uses different module names
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    print("CPU Configuration:")
    print(f"Model: {MODEL_NAME}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"LoRA rank: {lora_config.r}")
    print("Quantization: Disabled (CPU mode)")

else:
    print("✅ CUDA available - using original GPU configuration")
    # Keep original configuration from previous cell


# In[29]:


# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Add padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Tokenizer loaded. Vocab size: {len(tokenizer)}")
print(f"Pad token: {tokenizer.pad_token}")
print(f"EOS token: {tokenizer.eos_token}")


# In[30]:


# Tokenize the dataset for training
def tokenize_function(examples):
    """Tokenize the text data for training"""
    # Tokenize the text
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding=False,  # We'll pad dynamically during training
        max_length=512,
        return_overflowing_tokens=False,
    )
    
    # For causal language modeling, labels are the same as input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

print("Tokenizing datasets...")

# Tokenize training dataset
tokenized_train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=train_dataset.column_names,
    desc="Tokenizing training dataset"
)

# Tokenize evaluation dataset
tokenized_eval_dataset = eval_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=eval_dataset.column_names,
    desc="Tokenizing evaluation dataset"
)

print(f"Tokenized training dataset size: {len(tokenized_train_dataset)}")
print(f"Tokenized evaluation dataset size: {len(tokenized_eval_dataset)}")
print(f"Sample tokenized data keys: {tokenized_train_dataset[0].keys()}")
print(f"Sample input_ids length: {len(tokenized_train_dataset[0]['input_ids'])}")


# In[31]:


# Load base model
print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

# Apply LoRA
print("Applying LoRA configuration...")
model = get_peft_model(model, lora_config)

# Print model info
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())

print(f"Trainable parameters: {trainable_params:,}")
print(f"Total parameters: {total_params:,}")
print(f"Percentage of trainable parameters: {100 * trainable_params / total_params:.2f}%")

model.print_trainable_parameters()


# In[32]:


# ✅ Step 4: Training Configuration

# Adjust training arguments based on available hardware
if torch.cuda.is_available():
    # GPU configuration
    batch_size = 1
    fp16_enabled = True
    device_pin_memory = True
else:
    # CPU configuration - more conservative settings
    batch_size = 1
    fp16_enabled = False  # FP16 not supported on CPU
    device_pin_memory = False

# Training arguments optimized for 500-example dataset
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,  # More epochs for smaller dataset
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=4,  # Smaller accumulation for small dataset
    learning_rate=2e-4,  # Standard learning rate for instruction tuning
    warmup_ratio=0.1,  # 10% warmup
    logging_steps=5,  # More frequent logging for small dataset
    eval_steps=20,  # Evaluate every 20 steps (good for ~400 training examples)
    save_steps=40,  # Save every 40 steps (multiple of eval_steps=20)
    save_total_limit=3,  # Keep 3 best checkpoints
    eval_strategy="steps",  # Enable evaluation during training
    dataloader_pin_memory=device_pin_memory,
    group_by_length=True,  # Group samples by length for efficiency
    report_to="none",  # Change to "wandb" for experiment tracking
    run_name="llama2-guanaco-lora-500",
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    fp16=fp16_enabled,
    remove_unused_columns=False,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    # Additional optimizations
    max_grad_norm=1.0,  # Gradient clipping
    weight_decay=0.01,  # L2 regularization
)

# Initialize trainer
# Note: Newer versions of SFTTrainer may have different parameter names
try:
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,  # Use train split
        eval_dataset=eval_dataset,    # Use eval split
        args=training_args,
        max_seq_length=512,
        dataset_text_field="text",
        packing=False,
        tokenizer=tokenizer,
    )
    print("✅ Using SFTTrainer with evaluation dataset")
except TypeError:
    # Fallback for different SFTTrainer API versions or use standard Trainer
    from transformers import Trainer, DataCollatorForLanguageModeling
    
    # Use data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal language modeling
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,  # Use tokenized train dataset
        eval_dataset=tokenized_eval_dataset,    # Use tokenized eval dataset
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    print("✅ Using standard Trainer with evaluation dataset")

print("Training configuration:")
print(f"Epochs: {training_args.num_train_epochs}")
print(f"Batch size: {training_args.per_device_train_batch_size}")
print(f"Gradient accumulation: {training_args.gradient_accumulation_steps}")
print(f"Learning rate: {training_args.learning_rate}")
print(f"Max sequence length: 512")


# In[33]:


# ✅ Evaluation Metrics and Monitoring

import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def compute_metrics(eval_pred):
    """Compute custom metrics for evaluation"""
    predictions, labels = eval_pred
    
    # Convert logits to predictions (get the token with highest probability)
    predictions = np.argmax(predictions, axis=-1)
    
    # Flatten arrays for comparison (ignoring padding tokens)
    predictions = predictions.flatten()
    labels = labels.flatten()
    
    # Only consider non-padding tokens (assuming -100 is used for padding in labels)
    mask = labels != -100
    predictions = predictions[mask]
    labels = labels[mask]
    
    # Calculate token-level accuracy
    accuracy = accuracy_score(labels, predictions)
    
    return {
        "accuracy": accuracy,
        "token_count": len(labels)
    }

def plot_training_history(trainer):
    """Plot training and evaluation loss"""
    if hasattr(trainer, 'state') and trainer.state.log_history:
        logs = trainer.state.log_history
        
        train_loss = [log['train_loss'] for log in logs if 'train_loss' in log]
        eval_loss = [log['eval_loss'] for log in logs if 'eval_loss' in log]
        
        if train_loss or eval_loss:
            plt.figure(figsize=(10, 5))
            
            if train_loss:
                plt.subplot(1, 2, 1)
                plt.plot(train_loss, label='Training Loss')
                plt.title('Training Loss')
                plt.xlabel('Step')
                plt.ylabel('Loss')
                plt.legend()
            
            if eval_loss:
                plt.subplot(1, 2, 2)
                plt.plot(eval_loss, label='Evaluation Loss', color='orange')
                plt.title('Evaluation Loss')
                plt.xlabel('Step')
                plt.ylabel('Loss')
                plt.legend()
            
            plt.tight_layout()
            plt.show()
        else:
            print("No training history available for plotting")

print("Evaluation metrics and monitoring functions ready!")
print("- compute_metrics: Custom metric computation for evaluation")
print("- plot_training_history: Visualize training progress")


# In[ ]:


# ✅ Step 5: Start Training with 500-example Dataset

print("🚀 Starting training with 500-example subset of OpenAssistant-Guanaco...")
print("This optimized dataset size provides:")
print("  • Faster training iterations")
print("  • Reasonable training time")
print("  • Good model performance for demonstration")
print("Evaluation will run every 20 steps to monitor progress.")

# Start training
training_result = trainer.train()

# Print training summary
print(f"\n✅ Training completed!")
print(f"Dataset used: OpenAssistant-Guanaco subset ({len(train_dataset)} training examples)")
print(f"Total training time: {training_result.metrics.get('train_runtime', 'N/A'):.2f} seconds" if 'train_runtime' in training_result.metrics else "")
print(f"Final training loss: {training_result.metrics.get('train_loss', 'N/A'):.4f}" if 'train_loss' in training_result.metrics else "")

# Run final evaluation
print("\n📊 Running final evaluation...")
eval_results = trainer.evaluate()
print("Final evaluation results:")
for key, value in eval_results.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.4f}")
    else:
        print(f"  {key}: {value}")

# Save the final model
trainer.save_model()
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\n💾 Model saved to {OUTPUT_DIR}")
print("Your model is now fine-tuned on a curated subset of high-quality instruction-following data!")

# Plot training history if available
try:
    plot_training_history(trainer)
except Exception as e:
    print(f"Could not plot training history: {e}")


# In[ ]:


# ✅ Comprehensive Model Evaluation

def evaluate_model_on_dataset(model, tokenizer, eval_dataset, num_samples=10):
    """Evaluate model performance on evaluation dataset"""
    
    if num_samples and num_samples < len(eval_dataset):
        # Sample random examples for evaluation
        import random
        indices = random.sample(range(len(eval_dataset)), num_samples)
        eval_subset = eval_dataset.select(indices)
    else:
        eval_subset = eval_dataset
    
    print(f"Evaluating on {len(eval_subset)} samples...")
    
    total_loss = 0
    valid_examples = 0
    
    model.eval()
    
    for i, example in enumerate(eval_subset):
        try:
            # Tokenize the text with proper length handling
            inputs = tokenizer(
                example["text"], 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            ).to(model.device)
            
            # Skip if the sequence is too short
            if inputs["input_ids"].shape[1] < 10:
                continue
            
            # Get model predictions
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                total_loss += loss.item()
                valid_examples += 1
            
            if i < 3:  # Show first 3 examples
                print(f"\nExample {i+1}:")
                # Show the human question part
                text = example['text']
                if "### Human:" in text:
                    human_part = text.split("### Human:")[1].split("### Assistant:")[0].strip()
                    print(f"Human: {human_part[:150]}...")
                print(f"Loss: {loss.item():.4f}")
        
        except Exception as e:
            print(f"Error processing example {i}: {e}")
            continue
    
    if valid_examples == 0:
        print("No valid examples processed!")
        return None
    
    avg_loss = total_loss / valid_examples
    perplexity = torch.exp(torch.tensor(avg_loss))
    
    print(f"\n📊 Evaluation Results:")
    print(f"Valid examples processed: {valid_examples}")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")
    
    return {
        "avg_loss": avg_loss,
        "perplexity": perplexity.item(),
        "valid_examples": valid_examples
    }

# Run comprehensive evaluation
print("🔍 Running comprehensive evaluation...")
eval_results = evaluate_model_on_dataset(model, tokenizer, eval_dataset, num_samples=5)

print(f"\n✅ Evaluation complete!")
print(f"This gives us a baseline for model performance on unseen data.")


# In[ ]:


# ✅ Step 6: Test the Fine-tuned Model

def generate_response(prompt: str, max_length: int = 256) -> str:
    """Generate response using the fine-tuned model"""
    
    # Format the prompt
    formatted_prompt = f"### Human: {prompt}\n\n### Assistant:"
    
    # Tokenize
    inputs = tokenizer(
        formatted_prompt, 
        return_tensors="pt", 
        truncation=True, 
        max_length=512
    ).to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=inputs["input_ids"].shape[1] + max_length,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's response
    if "### Assistant:" in response:
        response = response.split("### Assistant:")[-1].strip()
    
    return response

# Test the model
test_prompts = [
    "What is machine learning?",
    "How do neural networks work?",
    "Explain the concept of recursion in programming."
]

print("Testing the fine-tuned model:")
print("=" * 50)

for prompt in test_prompts:
    print(f"\n🔹 Human: {prompt}")
    response = generate_response(prompt)
    print(f"🤖 Assistant: {response}")
    print("-" * 50)


# In[ ]:


# ✅ Step 7: Model Loading and Deployment Utilities

def load_fine_tuned_model(model_path: str):
    """Load the fine-tuned model for inference"""
    from peft import PeftModel
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    
    # Load LoRA weights
    model = PeftModel.from_pretrained(base_model, model_path)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    return model, tokenizer

def save_model_for_deployment(output_dir: str):
    """Save model in a format suitable for deployment"""
    
    # Merge LoRA weights with base model for deployment
    merged_model = model.merge_and_unload()
    
    # Save merged model
    deployment_dir = f"{output_dir}_merged"
    merged_model.save_pretrained(deployment_dir)
    tokenizer.save_pretrained(deployment_dir)
    
    print(f"Merged model saved to {deployment_dir}")
    print("This merged model can be used without PEFT dependencies.")
    
    return deployment_dir

# Example usage (uncomment to use)
# deployment_path = save_model_for_deployment(OUTPUT_DIR)

print("Utilities for model loading and deployment are ready!")
print("\nTo load the fine-tuned model later:")
print("model, tokenizer = load_fine_tuned_model('./llama2-claude-lora')")
print("\nTo create a deployment-ready model:")
print("deployment_path = save_model_for_deployment('./llama2-claude-lora')")


# ## 🔧 Advanced Configuration & Tips
# 
# ### Memory Optimization
# - **4-bit Quantization**: Reduces memory usage by ~75%
# - **Gradient Checkpointing**: Enable with `gradient_checkpointing=True`
# - **DeepSpeed**: For multi-GPU training, add DeepSpeed configuration
# 
# ### LoRA Parameters Tuning
# - **Rank (r)**: Higher rank = more parameters but better adaptation (8-64)
# - **Alpha**: Usually 2×rank for optimal performance
# - **Target Modules**: Include attention and MLP layers for best results
# 
# ### Dataset Scaling
# For production use:
# 1. Use datasets with 10K+ examples
# 2. Implement proper train/validation splits
# 3. Add data filtering and quality checks
# 4. Consider multi-turn conversation handling
# 
# ### Training Optimizations
# ```python
# # Example production training arguments
# training_args = TrainingArguments(
#     output_dir="./llama2-claude-production",
#     num_train_epochs=1,  # Usually 1-3 epochs for instruction tuning
#     per_device_train_batch_size=4,
#     gradient_accumulation_steps=4,
#     learning_rate=2e-4,
#     max_grad_norm=1.0,
#     warmup_ratio=0.1,
#     lr_scheduler_type="cosine",
#     logging_steps=25,
#     save_strategy="epoch",
#     evaluation_strategy="epoch",
#     fp16=True,
#     dataloader_num_workers=4,
#     remove_unused_columns=False,
#     report_to="wandb",  # For experiment tracking
#     load_best_model_at_end=True,
#     metric_for_best_model="eval_loss",
# )
# ```

# ## 📊 Evaluation Setup Summary
# 
# This notebook now includes comprehensive evaluation capabilities:
# 
# ### 🔄 Training Evaluation
# - **Train/Validation Split**: 80/20 split with additional evaluation samples
# - **Evaluation Strategy**: Evaluation runs every few training steps
# - **Metrics Tracked**: Loss, perplexity, and token-level accuracy
# - **Best Model Selection**: Automatically saves the best model based on evaluation loss
# 
# ### 📈 Evaluation Metrics
# - **Perplexity**: Measures how well the model predicts the next token (lower is better)
# - **Loss**: Cross-entropy loss on the evaluation set
# - **Token Accuracy**: Percentage of correctly predicted tokens
# 
# ### 🎯 Evaluation Best Practices
# 1. **Diverse Evaluation Data**: Include various types of questions and responses
# 2. **Regular Monitoring**: Evaluate frequently to catch overfitting early
# 3. **Multiple Metrics**: Don't rely on loss alone - use task-specific metrics
# 4. **Human Evaluation**: Supplement automated metrics with human assessment
# 
# ### 🏭 Production Evaluation
# For production systems, consider:
# - **Larger Evaluation Sets**: 1000+ examples covering edge cases
# - **Domain-Specific Metrics**: BLEU, ROUGE, or custom task metrics
# - **A/B Testing**: Compare model versions in real deployment
# - **Continuous Monitoring**: Track performance drift over time

# In[ ]:


get_ipython().system('jupyter nbconvert --to python llm_train.ipynb --output llm_training_script')

