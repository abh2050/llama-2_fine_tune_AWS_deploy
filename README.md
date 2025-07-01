![](https://media.licdn.com/dms/image/v2/D4D12AQF7eKpx4HnIMQ/article-cover_image-shrink_720_1280/article-cover_image-shrink_720_1280/0/1736873437744?e=2147483647&v=beta&t=XVOO2gNANH331BpmQfW6NX3f-PRIjWGYOquPBdjoqQ4)
# LLama2 Inference Service

A complete end-to-end solution for training, fine-tuning, and deploying LLaMA2 models with production-ready inference capabilities.

## 🚀 Complete Pipeline: Training → Deployment → Inference

This repository represents the final step in a complete LLM pipeline:

1. **Model Training** - Fine-tuned LLaMA2 with LoRA on custom datasets
2. **Model Optimization** - Optimized for inference performance 
3. **Containerization** - Docker-ready inference service
4. **Cloud Deployment** - AWS EC2/ECS deployment with S3 model storage
5. **Production API** - RESTful endpoints for text generation and RAG

## Features

- **Complete LLM Pipeline** from training to production deployment
- **Fine-tuned LLaMA2** with LoRA (Low-Rank Adaptation) support
- **RESTful API** with multiple endpoints for text generation
- **RAG Support** (Retrieval-Augmented Generation)  
- **Docker containerized** for easy deployment
- **AWS S3 integration** for model storage and distribution
- **Production-ready** with comprehensive testing and monitoring
- **Scalable deployment** on AWS EC2, ECS, or Fargate

## 📋 End-to-End Process Overview

### Phase 1: Model Training & Fine-tuning
The foundation of this service was built through:

1. **Data Preparation**:
   - Custom dataset creation with Claude-style conversational format
   - Data cleaning and formatting for LLaMA2 training
   - Quality assurance and validation

2. **Model Fine-tuning**:
   - Base model: LLaMA2-7B from Meta
   - Fine-tuning method: LoRA (Low-Rank Adaptation)
   - Training framework: Transformers + PEFT
   - Custom training pipeline with monitoring

3. **Model Optimization**:
   - Post-training quantization for inference speed
   - Model validation and performance testing
   - Checkpoint management and versioning

### Phase 2: Inference Service Development
4. **API Development**:
   - Flask-based RESTful API
   - Multiple inference endpoints (predict, RAG, health)
   - Error handling and logging
   - Parameter customization support

5. **Containerization**:
   - Docker image with optimized dependencies
   - Multi-stage builds for size optimization
   - Security hardening (non-root user)
   - Health checks and monitoring

### Phase 3: Cloud Deployment & Production
6. **AWS Infrastructure**:
   - S3 bucket for model storage and distribution
   - EC2 instances for initial deployment and testing
   - ECS/Fargate for scalable container orchestration
   - CloudWatch for logging and monitoring

7. **Production Deployment**:
   - Automated model downloading from S3
   - Environment-based configuration
   - Load balancing and auto-scaling capabilities
   - Comprehensive testing and validation

## API Endpoints

- `GET /health` - Health check
- `POST /predict` - Text generation
- `POST /rag` - RAG-based generation
- `GET /model/info` - Model information

## 🛠 Prerequisites & Setup

### Model Training Prerequisites
Before using this inference service, ensure you have:

1. **Trained Model**: A fine-tuned LLaMA2 model (LoRA or merged)
2. **Model Storage**: Models uploaded to AWS S3 or available locally
3. **AWS Access**: Configured AWS credentials for S3 access

### Training Pipeline Reference
This inference service was built to work with models trained using:

```python
# Training configuration used:
- Base Model: meta-llama/Llama-2-7b-hf
- Fine-tuning: LoRA with rank=64, alpha=16
- Dataset: Custom conversational data in Claude style
- Training Framework: transformers + peft + accelerate
- Optimization: 4-bit quantization with bitsandbytes
```

## Quick Start

### Local Development

1. **Clone and setup**:
   ```bash
   git clone <repository-url>
   cd llama2-inference-service
   pip install -r requirements.txt
   ```

2. **Prepare your model**:
   - Place your fine-tuned LLaMA2 model in `./models/` directory
   - Or configure S3 bucket for automatic model download

3. **Run the service**:
   ```bash
   cd src
   python app.py
   ```

### Docker Deployment

1. **Build the image**:
   ```bash
   docker build -t llama2-inference .
   ```

2. **Run the container**:
   ```bash
   docker run -p 8000:8000 \
     -e MODEL_BUCKET=your-s3-bucket \
     -e MODEL_S3_PREFIX=models/your-model/ \
     llama2-inference
   ```

### AWS ECS/Fargate Production Deployment

For production-scale deployment with auto-scaling:

1. **Create ECR Repository**:
   ```bash
   aws ecr create-repository --repository-name llama2-inference --region us-east-1
   ```

2. **Build and Push to ECR**:
   ```bash
   # Get login token
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
   
   # Build and tag
   docker build -t llama2-inference .
   docker tag llama2-inference:latest <account>.dkr.ecr.us-east-1.amazonaws.com/llama2-inference:latest
   
   # Push to ECR
   docker push <account>.dkr.ecr.us-east-1.amazonaws.com/llama2-inference:latest
   ```

3. **Deploy to ECS**:
   ```bash
   # Update task-definition.json with your account details
   # Register the task definition
   aws ecs register-task-definition --cli-input-json file://task-definition.json
   
   # Create or update ECS service
   aws ecs create-service --cluster your-cluster --service-name llama2-inference --task-definition llama2-inference-task
   ```

4. **Verify Deployment**:
   ```bash
   # Get service status
   aws ecs describe-services --cluster your-cluster --services llama2-inference
   
   # Test the deployed service
   curl http://your-load-balancer-url/health
   ```

### Complete AWS EC2 Deployment Guide

For step-by-step EC2 deployment:

1. **Launch EC2 Instance**:
   - Instance type: t3.medium or larger (for 7B models)
   - AMI: Amazon Linux 2 or Ubuntu 20.04+
   - Security group: Allow HTTP (80), HTTPS (443), SSH (22)

2. **Setup Environment**:
   ```bash
   # SSH into instance
   ssh -i your-key.pem ec2-user@your-instance-ip
   
   # Install Docker
   sudo yum update -y
   sudo yum install docker -y
   sudo systemctl start docker
   sudo usermod -a -G docker ec2-user
   
   # Install AWS CLI
   curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
   unzip awscliv2.zip
   sudo ./aws/install
   ```

3. **Deploy Service**:
   ```bash
   # Clone repository
   git clone <your-repo-url>
   cd llama2-inference-service
   
   # Build and run
   docker build -t llama2-inference .
   docker run -d -p 8000:8000 \
     -e MODEL_BUCKET=your-s3-bucket \
     -e MODEL_S3_PREFIX=models/your-model/ \
     -e AWS_DEFAULT_REGION=us-east-1 \
     --name llama2-service \
     llama2-inference
   ```

## API Usage Examples

### Text Generation

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain machine learning in simple terms",
    "max_length": 256,
    "temperature": 0.7
  }'
```

### RAG Generation

```bash
curl -X POST http://localhost:8000/rag \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the capital of France?",
    "context": ["France is a country in Europe.", "Paris is a major city in France."],
    "max_length": 256
  }'
```

## 🔧 Configuration

### Environment Variables

- `MODEL_BUCKET` - S3 bucket containing the trained model
- `MODEL_S3_PREFIX` - S3 prefix/path to model files (e.g., `models/llama2-claude-lora/`)
- `AWS_DEFAULT_REGION` - AWS region for S3 access (default: us-east-1)

### Model Requirements

The service supports models from our training pipeline:

- **LoRA Models** - Base LLaMA2 + LoRA adapter files
  ```
  models/
  ├── adapter_config.json
  ├── adapter_model.safetensors
  ├── tokenizer.json
  ├── tokenizer_config.json
  └── ...
  ```

- **Merged Models** - Fully merged fine-tuned models
- **Local Models** - Models stored locally in `./models/` directory

### Training Output Integration

The service automatically handles model artifacts from the training pipeline:
- Supports checkpoint formats from Transformers + PEFT
- Handles both 4-bit and 16-bit quantized models  
- Automatically detects LoRA vs merged model configurations
- Compatible with models trained using our custom training scripts

## Project Structure

```
llama2-inference-service/
├── src/
│   ├── app.py                 # Main Flask application
│   ├── model/
│   │   ├── __init__.py
│   │   └── inference.py       # Model inference logic
│   └── utils/
│       ├── __init__.py
│       └── aws_helpers.py     # AWS S3 utilities
├── tests/
│   ├── test_api.py            # Simple API tests
│   ├── test_llm_service.py    # Comprehensive LLM tests
│   └── test_load.py           # Load/performance tests
├── Dockerfile                 # Container definition
├── requirements.txt           # Python dependencies
├── task-definition.json       # ECS task definition (optional)
├── run_tests.py               # Test runner script
├── .gitignore                 # Git ignore file
└── README.md                  # This file
```

## Testing

The service includes comprehensive testing to verify LLM functionality and performance.

### Quick Test

Test the basic API functionality:
```bash
python run_tests.py --quick
```

### Comprehensive LLM Testing

Test all endpoints with various scenarios:
```bash
python run_tests.py --test comprehensive
```

### Load Testing

Test performance under concurrent requests:
```bash
python run_tests.py --test load
```

### Run All Tests

```bash
python run_tests.py --all
```

### Individual Test Scripts

1. **Simple API Test** (`tests/test_api.py`):
   ```bash
   cd tests && python test_api.py
   ```

2. **Comprehensive LLM Test** (`tests/test_llm_service.py`):
   ```bash
   cd tests && python test_llm_service.py
   ```
   
   This tests:
   - Health check endpoint
   - Text generation with various parameters
   - RAG (Retrieval-Augmented Generation) 
   - Model information
   - Error handling
   - Conversation flow

3. **Load Test** (`tests/test_load.py`):
   ```bash
   cd tests && python test_load.py --requests 10 --threads 3
   ```
   
   This tests:
   - Concurrent request handling
   - Response time statistics
   - Success rate under load
   - Performance metrics

### Test Examples

**Testing Text Generation:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain machine learning in simple terms",
    "max_length": 200,
    "temperature": 0.7
  }'
```

**Testing RAG:**
```bash
curl -X POST http://localhost:8000/rag \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is neural networks?",
    "context": ["Neural networks are computational models inspired by biological neural networks."],
    "max_length": 150
  }'
```

## 🔄 Complete Training-to-Production Pipeline

### Pipeline Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Prep     │    │  Model Training │    │  Model Export   │
│                 │    │                 │    │                 │
│ • Custom dataset│ => │ • LLaMA2 + LoRA │ => │ • Save adapters │
│ • Conversation  │    │ • Fine-tuning   │    │ • Upload to S3  │
│   formatting    │    │ • Validation    │    │ • Version mgmt  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Monitoring    │    │  Inference API  │    │ Model Loading   │
│                 │    │                 │    │                 │
│ • Health checks │ <= │ • REST endpoints│ <= │ • S3 download   │
│ • Performance   │    │ • Text gen      │    │ • LoRA loading  │
│ • Scaling       │    │ • RAG support   │    │ • GPU/CPU setup │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Training Integration

This inference service was designed to work seamlessly with models trained using:

1. **Training Script** (`llm_training_script.py`):
   ```python
   # Key training parameters used:
   model_name = "meta-llama/Llama-2-7b-hf"
   dataset = "custom_claude_style_data.json"
   
   # LoRA configuration
   lora_config = LoraConfig(
       r=64, 
       lora_alpha=16,
       target_modules=["q_proj", "v_proj"],
       lora_dropout=0.1
   )
   
   # Training arguments
   training_args = TrainingArguments(
       per_device_train_batch_size=4,
       gradient_accumulation_steps=4,
       num_train_epochs=3,
       learning_rate=2e-4
   )
   ```

2. **Training Notebook** (`llm_train.ipynb`):
   - Interactive model training and validation
   - Hyperparameter experimentation  
   - Training progress monitoring
   - Model evaluation and testing

3. **Model Artifacts**:
   ```
   dialogpt-claude-lora-cpu/
   ├── adapter_config.json      # LoRA configuration
   ├── adapter_model.safetensors # LoRA weights
   ├── tokenizer files          # Tokenization setup
   └── training_args.bin        # Training metadata
   ```

### Deployment Workflow

1. **Model Training** → Fine-tune LLaMA2 with custom data
2. **Model Upload** → Upload trained model to S3
3. **Container Build** → Create Docker image with inference code
4. **AWS Deployment** → Deploy to EC2/ECS with model auto-download
5. **Testing & Validation** → Run comprehensive test suite
6. **Production Ready** → Service available for inference requests

## ⚡ Performance Considerations

### Hardware Requirements

**Minimum Requirements:**
- **CPU**: 4+ cores, 2.5GHz+
- **RAM**: 8GB (16GB recommended for 7B models)
- **Storage**: 20GB+ SSD for model storage
- **Network**: Stable internet for S3 model download

**Recommended for Production:**
- **CPU**: 8+ cores, 3.0GHz+ (Intel Xeon, AMD EPYC)
- **RAM**: 32GB+ for optimal performance
- **GPU**: NVIDIA T4, V100, or A100 (optional but significantly faster)
- **Storage**: NVMe SSD for fast model loading
- **Network**: High bandwidth (1Gbps+) for large model downloads

### Performance Optimization

**Model Loading:**
- Cold start: 30-60 seconds (model download + loading)
- Warm start: 2-5 seconds (model already loaded)
- S3 download time: 2-10 minutes (depends on model size and network)

**Inference Performance:**
- **CPU Inference**: 2-5 seconds per request (256 tokens)
- **GPU Inference**: 0.5-1 second per request (256 tokens)  
- **Concurrent Requests**: 1-3 simultaneous (CPU), 5-10 (GPU)
- **Memory Scaling**: Linear with model size and batch size

### Scaling Strategies

1. **Horizontal Scaling** (Multiple instances):
   ```bash
   # Multiple containers with load balancer
   docker run -d -p 8001:8000 llama2-inference
   docker run -d -p 8002:8000 llama2-inference
   docker run -d -p 8003:8000 llama2-inference
   ```

2. **ECS Auto Scaling**:
   - CPU-based scaling triggers
   - Request-based scaling metrics
   - Predictive scaling for known traffic patterns

3. **Model Optimization**:
   - Use quantized models (4-bit, 8-bit) for faster inference
   - Model pruning for reduced memory footprint
   - Batch inference for higher throughput

## 🔒 Security

- Container runs as non-root user
- Environment variables for sensitive configuration
- AWS IAM roles for S3 access (no hardcoded credentials)

## 🔒 Security

### Container Security
- **Non-root user**: Container runs as unprivileged user
- **Minimal base image**: Python slim image with only necessary packages
- **No secrets in image**: All sensitive data via environment variables
- **Resource limits**: CPU and memory limits to prevent resource exhaustion

### AWS Security
- **IAM roles**: Use IAM roles instead of hardcoded credentials
- **S3 bucket policies**: Restrict access to model artifacts
- **VPC security groups**: Limit network access to required ports only
- **Encryption**: Models stored with S3 server-side encryption

### API Security
- **Input validation**: Strict validation of all API inputs
- **Rate limiting**: Prevent API abuse (implement via API Gateway)
- **Request logging**: Comprehensive logging for audit trails
- **Error handling**: Secure error messages (no sensitive info leakage)

### Recommended Security Setup
```bash
# Create IAM role for ECS tasks with minimal S3 permissions
aws iam create-role --role-name LLaMA2InferenceRole --assume-role-policy-document file://trust-policy.json

# Attach policy for S3 model access only
aws iam attach-role-policy --role-name LLaMA2InferenceRole --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess

# Use VPC security groups to restrict access
aws ec2 create-security-group --group-name llama2-inference-sg --description "LLaMA2 Inference Service"
aws ec2 authorize-security-group-ingress --group-name llama2-inference-sg --protocol tcp --port 8000 --cidr 10.0.0.0/8
```

## 📚 Additional Resources

### Training Materials
This inference service was built using these training resources:

- **`llm_train.ipynb`** - Interactive training notebook with step-by-step process
- **`llm_training_script.py`** - Production training script with full pipeline
- **`claude4_style_data.json`** - Custom conversational dataset for fine-tuning
- **`dialogpt-claude-lora-cpu/`** - Trained model artifacts and checkpoints

### Reference Implementation
The complete training-to-deployment pipeline includes:

1. **Data Preparation**: Custom conversational data formatting
2. **Model Training**: LLaMA2 + LoRA fine-tuning with PEFT
3. **Model Validation**: Comprehensive testing and evaluation
4. **Inference Optimization**: Performance tuning for production
5. **Cloud Deployment**: AWS-native deployment with auto-scaling
6. **Monitoring & Testing**: Health checks and performance monitoring

### Useful Commands

**Model Management:**
```bash
# Upload trained model to S3
aws s3 sync ./dialogpt-claude-lora-cpu/ s3://your-bucket/models/llama2-claude-lora/

# Download model for local testing  
aws s3 sync s3://your-bucket/models/llama2-claude-lora/ ./models/

# Check model size and structure
du -sh ./models/
find ./models/ -name "*.json" -exec jq . {} \;
```

**Service Management:**
```bash
# View service logs
docker logs -f llama2-service

# Monitor resource usage
docker stats llama2-service

# Update service with new model
docker pull your-registry/llama2-inference:latest
docker stop llama2-service && docker rm llama2-service
docker run -d --name llama2-service -p 8000:8000 your-registry/llama2-inference:latest
```

## Support

For issues and questions, please open a GitHub issue or contact the maintainers.
