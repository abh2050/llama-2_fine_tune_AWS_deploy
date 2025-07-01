# DialoGPT-small Model: Machine Requirements & Deployment Guide

## Model Specifications

**Base Model**: microsoft/DialoGPT-small (~340MB)  
**Fine-tuning**: LoRA adapters (~10-50MB)  
**Total Model Size**: ~400MB  

## Machine Requirements

### Minimum Requirements (Development/Testing)
- **Instance Type**: AWS t3.medium
- **RAM**: 4 GB
- **Disk**: 16 GB
- **CPU**: 2 vCPUs
- **Python**: 3.8+
- **Cost**: ~$30/month

### Recommended (Production)
- **Instance Type**: AWS t3.large ⭐ **RECOMMENDED**
- **RAM**: 8 GB
- **Disk**: 32 GB (gp3 SSD)
- **CPU**: 2 vCPUs
- **Python**: 3.9+
- **Cost**: ~$60/month

### High Performance (Heavy Load)
- **Instance Type**: AWS c5.xlarge
- **RAM**: 8 GB
- **Disk**: 32 GB
- **CPU**: 4 vCPUs
- **Cost**: ~$120/month

## Why These Specifications?

### Memory Breakdown
```
Base Model Loading:          ~800MB
PyTorch Framework:          ~1.5GB
Transformers Library:       ~500MB
LoRA Adapters:              ~50MB
Operating System:           ~1GB
Buffer for Processing:      ~2GB
--------------------------------
Total RAM Required:         ~5.8GB
Recommended with Buffer:    8GB
```

### Disk Space Breakdown
```
Operating System:           ~8GB
Python + Dependencies:     ~3GB
Model Files:               ~400MB
Docker (if used):          ~2GB
Logs & Temporary Files:    ~1GB
Development Tools:         ~1GB
--------------------------------
Total Disk Required:       ~15.4GB
Recommended with Buffer:   32GB
```

### CPU Requirements
- **DialoGPT-small**: Light CPU requirements
- **2 vCPUs**: Sufficient for single-user inference
- **4 vCPUs**: Better for concurrent requests (5-10 users)
- **CPU Architecture**: x86_64 (Intel/AMD)

## Deployment Options

### Option 1: Optimized Deployment (Recommended)
```bash
# Use the provided optimized deployment script
cd llama2-inference-service
./deploy_optimized_instance.sh
```

This script automatically:
- Creates a t3.large instance (8GB RAM, 32GB disk)
- Installs Python 3.9 and dependencies
- Deploys your DialoGPT-small model
- Configures security groups
- Tests the deployment

### Option 2: Manual Deployment
1. Create EC2 instance with recommended specs
2. Install dependencies manually
3. Copy model files
4. Start the service

### Option 3: Docker Deployment
```bash
# Build optimized Docker image
docker build -t dialogpt-inference .
docker run -p 8000:8000 dialogpt-inference
```

## Performance Expectations

### Inference Speed (CPU)
- **Small prompts (10-20 tokens)**: 1-3 seconds
- **Medium prompts (50-100 tokens)**: 3-8 seconds
- **Large prompts (200+ tokens)**: 8-15 seconds

### Concurrent Users
- **t3.medium**: 1-2 concurrent users
- **t3.large**: 3-5 concurrent users
- **c5.xlarge**: 5-10 concurrent users

### Memory Usage During Inference
- **Idle**: ~2-3GB RAM
- **Single Inference**: ~3-4GB RAM
- **Peak Load**: ~5-6GB RAM

## Cost Analysis

### Monthly AWS Costs

| Instance Type | vCPUs | RAM | Storage | Instance Cost | Storage Cost | Total/Month |
|---------------|-------|-----|---------|---------------|--------------|-------------|
| t3.medium     | 2     | 4GB | 16GB    | $30.37        | $1.60        | **$31.97**  |
| t3.large      | 2     | 8GB | 32GB    | $60.74        | $3.20        | **$63.94**  |
| c5.xlarge     | 4     | 8GB | 32GB    | $122.40       | $3.20        | **$125.60** |

*Prices for us-east-1 region, 24/7 operation*

### Cost Optimization Tips
1. **Stop instance when not in use**: Save ~70% on compute costs
2. **Use Spot Instances**: Save up to 90% (for development)
3. **Reserved Instances**: Save up to 75% (for production, 1-3 year terms)

## Monitoring Resources

Use the provided monitoring script:
```bash
# Check current resource usage
./monitor_resources.sh info

# Continuous monitoring
./monitor_resources.sh monitor

# Test model load impact
./monitor_resources.sh test
```

## Troubleshooting Common Issues

### Out of Memory (OOM)
- **Symptom**: Process killed, "Killed" message
- **Solution**: Upgrade to instance with more RAM
- **Quick Fix**: Reduce max_length parameter in API calls

### Slow Inference
- **Symptom**: API timeouts, long response times
- **Solution**: Upgrade CPU or optimize model loading
- **Check**: Monitor CPU usage during inference

### Disk Space Issues
- **Symptom**: "No space left on device"
- **Solution**: Increase EBS volume size
- **Quick Fix**: Clean up logs and temporary files

### Dependency Conflicts
- **Symptom**: Import errors, version conflicts
- **Solution**: Use Python 3.9+ and updated requirements.txt
- **Alternative**: Use Docker for isolated environment

## Scaling Considerations

### Horizontal Scaling (Multiple Instances)
- Use Application Load Balancer
- Auto Scaling Groups
- Shared model storage (EFS)

### Vertical Scaling (Bigger Instance)
- Easy upgrade path: t3.medium → t3.large → c5.xlarge
- Stop instance, change instance type, restart

### GPU Acceleration (Future)
- **For larger models**: Consider g4dn.xlarge with GPU
- **Current model**: CPU is sufficient and more cost-effective

## Security Considerations

### Network Security
- Restrict API access to specific IPs
- Use HTTPS in production (add SSL certificate)
- VPC configuration for internal access only

### Data Security
- Encrypt EBS volumes
- Use IAM roles instead of access keys
- Regular security updates

## Production Checklist

- [ ] Use t3.large or larger instance
- [ ] Enable CloudWatch monitoring
- [ ] Set up automated backups
- [ ] Configure SSL/HTTPS
- [ ] Implement request rate limiting
- [ ] Set up log rotation
- [ ] Configure auto-restart on failure
- [ ] Test disaster recovery procedures

## Summary

For your **DialoGPT-small with LoRA** model:

✅ **Recommended**: AWS t3.large (8GB RAM, 32GB disk) - $64/month  
⚡ **Performance**: 3-8 second inference, 3-5 concurrent users  
🔧 **Easy Setup**: Use `./deploy_optimized_instance.sh`  
📊 **Monitoring**: Use `./monitor_resources.sh`  

This setup provides the optimal balance of performance, cost, and reliability for your model.
