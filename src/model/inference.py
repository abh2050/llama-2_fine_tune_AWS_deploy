import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
import logging

logger = logging.getLogger(__name__)

class ModelInference:
    def __init__(self, model_path):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.model_info = {}
        self._load_model()
    
    def _load_model(self):
        """Load the fine-tuned model and tokenizer"""
        logger.info(f"Loading model from {self.model_path}")
        logger.info(f"Using device: {self.device}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Check if this is a merged model or LoRA model
            adapter_config_path = os.path.join(self.model_path, "adapter_config.json")
            
            if os.path.exists(adapter_config_path):
                # This is a LoRA model - need to load base + adapter
                logger.info("Loading LoRA model...")
                
                # Try to determine base model from adapter config
                try:
                    import json
                    with open(adapter_config_path, 'r') as f:
                        adapter_config = json.load(f)
                    base_model_name = adapter_config.get("base_model_name_or_path", "meta-llama/Llama-2-7b-hf")
                except:
                    base_model_name = "meta-llama/Llama-2-7b-hf"
                
                logger.info(f"Loading base model: {base_model_name}")
                
                # Load base model with appropriate settings
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                    trust_remote_code=True
                )
                
                # Load LoRA adapter
                self.model = PeftModel.from_pretrained(base_model, self.model_path)
                
                # Get model info
                self.model_info = {
                    "type": "LoRA",
                    "base_model": base_model_name,
                    "adapter_path": self.model_path
                }
                
            else:
                # This is a merged model
                logger.info("Loading merged model...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                    trust_remote_code=True
                )
                
                self.model_info = {
                    "type": "Merged",
                    "model_path": self.model_path
                }
            
            # Move to device if not using device_map
            if not torch.cuda.is_available():
                self.model = self.model.to(self.device)
            
            # Set to evaluation mode
            self.model.eval()
            
            # Update model info
            self.model_info.update({
                "device": str(self.device),
                "vocab_size": len(self.tokenizer),
                "model_loaded": True
            })
            
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model_info = {"error": str(e), "model_loaded": False}
            raise
    
    def generate_response(self, prompt, max_length=256, temperature=0.7, top_p=0.9, do_sample=True):
        """Generate response using the fine-tuned model"""
        
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded properly")
        
        # Format the prompt if it's not already formatted
        if not prompt.startswith("### Human:"):
            formatted_prompt = f"### Human: {prompt}\n\n### Assistant:"
        else:
            formatted_prompt = prompt
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=inputs["input_ids"].shape[1] + max_length,
                temperature=temperature,
                do_sample=do_sample,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3
            )
        
        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        if "### Assistant:" in response:
            response = response.split("### Assistant:")[-1].strip()
        
        return response
    
    def get_model_info(self):
        """Get information about the loaded model"""
        return self.model_info
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
