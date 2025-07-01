#!/bin/bash

# AWS CLI Deployment Script for Real LLaMA2 Model
# This script deploys the actual fine-tuned model to your existing EC2 instance

set -e

echo "🚀 AWS CLI Deployment: Real LLaMA2 Model"
echo "========================================"

# Configuration
INSTANCE_ID="i-0170e332cd00a1105"
KEY_FILE="/Users/abhishekshah/Desktop/train_llm/llama2-ec2-key.pem"
PUBLIC_IP="52.205.207.112"
REGION="us-east-1"

echo "📋 Configuration:"
echo "  Instance ID: $INSTANCE_ID"
echo "  Public IP: $PUBLIC_IP"
echo "  Region: $REGION"
echo ""

# Function to check if instance is running
check_instance_status() {
    echo "🔍 Checking instance status..."
    STATUS=$(aws ec2 describe-instances \
        --instance-ids $INSTANCE_ID \
        --query 'Reservations[0].Instances[0].State.Name' \
        --output text \
        --region $REGION)
    
    echo "  Instance status: $STATUS"
    
    if [ "$STATUS" != "running" ]; then
        echo "❌ Instance is not running. Starting instance..."
        aws ec2 start-instances --instance-ids $INSTANCE_ID --region $REGION
        
        echo "⏳ Waiting for instance to be running..."
        aws ec2 wait instance-running --instance-ids $INSTANCE_ID --region $REGION
        echo "✅ Instance is now running"
    else
        echo "✅ Instance is already running"
    fi
}

# Function to get current public IP
get_public_ip() {
    echo "🔍 Getting current public IP..."
    CURRENT_IP=$(aws ec2 describe-instances \
        --instance-ids $INSTANCE_ID \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text \
        --region $REGION)
    
    echo "  Current IP: $CURRENT_IP"
    
    if [ "$CURRENT_IP" != "$PUBLIC_IP" ]; then
        echo "⚠️  IP has changed from $PUBLIC_IP to $CURRENT_IP"
        echo "  You may need to update your DNS or firewall rules"
    fi
}

# Function to deploy via SSH
deploy_real_model() {
    echo "🚀 Deploying real model via SSH..."
    
    # Check if key file exists
    if [ ! -f "$KEY_FILE" ]; then
        echo "❌ Key file not found: $KEY_FILE"
        echo "   Please ensure the key file is in the correct location"
        exit 1
    fi
    
    # Set correct permissions on key file
    chmod 400 "$KEY_FILE"
    
    echo "📡 Connecting to EC2 instance..."
    
    # Deploy commands
    ssh -i "$KEY_FILE" -o StrictHostKeyChecking=no ec2-user@$CURRENT_IP << 'EOF'
        echo "🔄 Starting deployment on EC2 instance..."
        
        # Check if git is installed
        if ! command -v git &> /dev/null; then
            echo "📦 Installing git..."
            sudo yum update -y
            sudo yum install -y git
        fi
        
        # Check if repository exists, if not clone it
        if [ ! -d "llama-2_fine_tune_AWS_deploy" ]; then
            echo "📥 Cloning repository..."
            git clone https://github.com/abh2050/llama-2_fine_tune_AWS_deploy.git
        fi
        
        # Navigate to project directory
        cd llama-2_fine_tune_AWS_deploy || {
            echo "❌ Project directory not found after cloning"
            exit 1
        }
        
        echo "📥 Pulling latest code with real model..."
        git pull origin main
        
        echo "🛑 Stopping existing demo service..."
        sudo docker stop llama2-service 2>/dev/null || true
        sudo docker rm llama2-service 2>/dev/null || true
        
        echo "🧹 Cleaning up old images..."
        sudo docker image prune -f
        
        echo "🔨 Building new container with real model..."
        sudo docker build -t llama2-inference .
        
        echo "🚀 Starting service with real model..."
        sudo docker run -d \
            --name llama2-service \
            -p 8000:8000 \
            --restart unless-stopped \
            llama2-inference
        
        echo "⏳ Waiting for service to start..."
        sleep 10
        
        echo "🏥 Checking service health..."
        curl -f http://localhost:8000/health || {
            echo "❌ Health check failed"
            sudo docker logs llama2-service
            exit 1
        }
        
        echo "✅ Real model deployment completed successfully!"
        echo "🎯 Service is running at: http://$(curl -s ifconfig.me):8000"
EOF
    
    if [ $? -eq 0 ]; then
        echo "✅ Deployment completed successfully!"
    else
        echo "❌ Deployment failed!"
        exit 1
    fi
}

# Function to test the deployment
test_deployment() {
    echo "🧪 Testing real model deployment..."
    
    # Wait a moment for service to fully start
    sleep 5
    
    echo "  Testing health endpoint..."
    if curl -s -f http://$CURRENT_IP:8000/health > /dev/null; then
        echo "✅ Health check passed"
    else
        echo "❌ Health check failed"
        return 1
    fi
    
    echo "  Testing model inference..."
    RESPONSE=$(curl -s -X POST http://$CURRENT_IP:8000/generate \
        -H "Content-Type: application/json" \
        -d '{"prompt": "Hello", "max_length": 50}' | jq -r '.generated_text')
    
    echo "  Response: $RESPONSE"
    
    # Check if it's still a demo response
    if echo "$RESPONSE" | grep -i "simulated\|demo\|based on your prompt" > /dev/null; then
        echo "⚠️  Still getting demo responses - deployment may need troubleshooting"
        return 1
    else
        echo "🎉 Getting real model responses!"
        return 0
    fi
}

# Function to check logs if deployment fails
check_logs() {
    echo "📋 Checking deployment logs..."
    ssh -i "$KEY_FILE" -o StrictHostKeyChecking=no ec2-user@$CURRENT_IP << 'EOF'
        echo "=== Docker Container Logs ==="
        sudo docker logs --tail 50 llama2-service
        
        echo "=== Container Status ==="
        sudo docker ps -a | grep llama2
        
        echo "=== Disk Usage ==="
        df -h
        
        echo "=== Memory Usage ==="
        free -h
EOF
}

# Main deployment flow
main() {
    echo "Starting AWS CLI deployment process..."
    
    # Step 1: Check instance status
    check_instance_status
    
    # Step 2: Get current IP
    get_public_ip
    
    # Step 3: Deploy real model
    deploy_real_model
    
    # Step 4: Test deployment
    if test_deployment; then
        echo ""
        echo "🎉 SUCCESS: Real model is now deployed!"
        echo "🔗 API URL: http://$CURRENT_IP:8000"
        echo "📚 API Docs: http://$CURRENT_IP:8000/docs"
        echo ""
        echo "Test your model:"
        echo "curl -X POST http://$CURRENT_IP:8000/generate \\"
        echo "  -H 'Content-Type: application/json' \\"
        echo "  -d '{\"prompt\": \"What is AI?\", \"max_length\": 100}'"
    else
        echo ""
        echo "⚠️  Deployment completed but tests show issues"
        echo "🔍 Checking logs for troubleshooting..."
        check_logs
        echo ""
        echo "💡 Next steps:"
        echo "  1. Check the logs above for errors"
        echo "  2. Verify model files are included in the container"
        echo "  3. Check EC2 instance resources (CPU/Memory)"
    fi
}

# Run main function
main "$@"
