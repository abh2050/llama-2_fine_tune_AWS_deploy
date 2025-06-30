from flask import Flask, request, jsonify
import torch
import os
import logging
from model.inference import ModelInference
from utils.aws_helpers import download_model_from_s3

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize model (will be loaded on first request)
model_inference = None

def get_model():
    """Get or initialize the model inference object"""
    global model_inference
    if model_inference is None:
        try:
            # Model configuration
            model_path = "./models/llama2-claude-lora"
            s3_bucket = os.getenv('MODEL_BUCKET', 'llama2-poc-llama2-models-604770467350')
            s3_prefix = os.getenv('MODEL_S3_PREFIX', 'models/llama2-claude-lora/')
            
            logger.info("Starting model initialization...")
            
            # Download model from S3 if not present locally
            if not os.path.exists(model_path):
                logger.info(f"Downloading model from S3: {s3_bucket}/{s3_prefix}")
                download_model_from_s3(s3_bucket, s3_prefix, model_path)
            
            # Initialize model inference
            logger.info("Initializing model inference...")
            model_inference = ModelInference(model_path)
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    return model_inference

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        return jsonify({
            "status": "healthy",
            "model_loaded": model_inference is not None,
            "cuda_available": torch.cuda.is_available()
        }), 200
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Generate text response from prompt"""
    try:
        data = request.get_json()
        
        if not data or 'prompt' not in data:
            return jsonify({"error": "Missing 'prompt' in request body"}), 400
        
        prompt = data['prompt']
        max_length = data.get('max_length', 256)
        temperature = data.get('temperature', 0.7)
        top_p = data.get('top_p', 0.9)
        
        logger.info(f"Generating response for prompt: {prompt[:50]}...")
        
        # Get model and generate response
        model = get_model()
        response = model.generate_response(
            prompt=prompt,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p
        )
        
        return jsonify({
            "response": response,
            "prompt": prompt,
            "parameters": {
                "max_length": max_length,
                "temperature": temperature,
                "top_p": top_p
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error in predict endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/rag', methods=['POST'])
def rag_predict():
    """RAG (Retrieval-Augmented Generation) endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({"error": "Missing 'query' in request body"}), 400
        
        query = data['query']
        context = data.get('context', [])
        max_length = data.get('max_length', 512)
        
        logger.info(f"Processing RAG query: {query[:50]}...")
        
        # Format RAG prompt
        if context:
            context_text = "\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(context)])
            rag_prompt = f"### Human: {query}\n\n### Context:\n{context_text}\n\n### Assistant:"
        else:
            rag_prompt = f"### Human: {query}\n\n### Assistant:"
        
        # Get model and generate response
        model = get_model()
        response = model.generate_response(rag_prompt, max_length=max_length)
        
        return jsonify({
            "response": response,
            "query": query,
            "context_documents": len(context),
            "parameters": {
                "max_length": max_length
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error in RAG endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    try:
        if model_inference is None:
            return jsonify({"error": "Model not loaded"}), 400
        
        info = model_inference.get_model_info()
        return jsonify(info), 200
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # For development only
    app.run(host='0.0.0.0', port=5000, debug=False)
