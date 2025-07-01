#!/bin/bash

# =============================================================================
# Optimized AWS Deployment Script for DialoGPT-small with LoRA
# Creates a properly sized instance for the model requirements
# =============================================================================

set -e  # Exit on any error

# Configuration
INSTANCE_TYPE="t3.large"  # 2 vCPUs, 8GB RAM - optimal for DialoGPT-small
VOLUME_SIZE=32           # 32GB storage for dependencies and model
KEY_NAME="llama2-ec2-key"
SECURITY_GROUP="llama2-sg"
REGION="us-east-1"
AMI_ID="ami-0c02fb55956c7d316"  # Amazon Linux 2023
SERVICE_PORT=8000

echo "🚀 Starting optimized deployment for DialoGPT-small model..."
echo "📊 Instance specs: ${INSTANCE_TYPE} (8GB RAM, 32GB storage)"

# Function to check if AWS CLI is configured
check_aws_cli() {
    if ! command -v aws &> /dev/null; then
        echo "❌ AWS CLI not found. Please install it first."
        exit 1
    fi
    
    if ! aws sts get-caller-identity &> /dev/null; then
        echo "❌ AWS CLI not configured. Please run 'aws configure' first."
        exit 1
    fi
    
    echo "✅ AWS CLI configured and working"
}

# Function to create security group if it doesn't exist
create_security_group() {
    if ! aws ec2 describe-security-groups --group-names "$SECURITY_GROUP" --region "$REGION" &> /dev/null; then
        echo "📡 Creating security group: $SECURITY_GROUP"
        
        SECURITY_GROUP_ID=$(aws ec2 create-security-group \
            --group-name "$SECURITY_GROUP" \
            --description "Security group for LLaMA2 inference service" \
            --region "$REGION" \
            --query 'GroupId' \
            --output text)
        
        # Add rules for SSH and API access
        aws ec2 authorize-security-group-ingress \
            --group-id "$SECURITY_GROUP_ID" \
            --protocol tcp \
            --port 22 \
            --cidr 0.0.0.0/0 \
            --region "$REGION"
        
        aws ec2 authorize-security-group-ingress \
            --group-id "$SECURITY_GROUP_ID" \
            --protocol tcp \
            --port "$SERVICE_PORT" \
            --cidr 0.0.0.0/0 \
            --region "$REGION"
        
        echo "✅ Security group created: $SECURITY_GROUP_ID"
    else
        echo "✅ Security group already exists: $SECURITY_GROUP"
    fi
}

# Function to launch optimized EC2 instance
launch_instance() {
    echo "🖥️  Launching optimized EC2 instance..."
    
    # Create user data script for instance initialization
    USER_DATA=$(cat << 'EOF'
#!/bin/bash
yum update -y
yum install -y git docker htop

# Install Python 3.9 (required for modern ML packages)
yum install -y python3.9 python3.9-pip python3.9-devel

# Set Python 3.9 as default
alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1
alternatives --install /usr/bin/pip3 pip3 /usr/bin/pip3.9 1

# Start and enable Docker
systemctl start docker
systemctl enable docker
usermod -a -G docker ec2-user

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Create application directory
mkdir -p /home/ec2-user/llama2-service
chown ec2-user:ec2-user /home/ec2-user/llama2-service

# Log instance readiness
echo "$(date): Instance initialization complete" >> /var/log/user-data.log
EOF
    )
    
    # Launch instance with optimized specifications
    INSTANCE_ID=$(aws ec2 run-instances \
        --image-id "$AMI_ID" \
        --count 1 \
        --instance-type "$INSTANCE_TYPE" \
        --key-name "$KEY_NAME" \
        --security-groups "$SECURITY_GROUP" \
        --user-data "$USER_DATA" \
        --block-device-mappings "[{\"DeviceName\":\"/dev/xvda\",\"Ebs\":{\"VolumeSize\":$VOLUME_SIZE,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=llama2-inference-optimized},{Key=Purpose,Value=DialoGPT-small-inference}]" \
        --region "$REGION" \
        --query 'Instances[0].InstanceId' \
        --output text)
    
    echo "🎯 Instance launched: $INSTANCE_ID"
    echo "⏳ Waiting for instance to be running..."
    
    aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"
    
    # Get public IP
    PUBLIC_IP=$(aws ec2 describe-instances \
        --instance-ids "$INSTANCE_ID" \
        --region "$REGION" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text)
    
    echo "✅ Instance ready! Public IP: $PUBLIC_IP"
    echo "$INSTANCE_ID" > .instance_id
    echo "$PUBLIC_IP" > .instance_ip
}

# Function to deploy the application
deploy_application() {
    PUBLIC_IP=$(cat .instance_ip)
    
    echo "📦 Deploying application to $PUBLIC_IP..."
    echo "⏳ Waiting for SSH to be available..."
    
    # Wait for SSH to be ready
    for i in {1..30}; do
        if ssh -i "$KEY_NAME.pem" -o StrictHostKeyChecking=no ec2-user@"$PUBLIC_IP" "echo 'SSH Ready'" 2>/dev/null; then
            break
        fi
        echo "Attempt $i/30: SSH not ready, waiting 10 seconds..."
        sleep 10
    done
    
    echo "🔄 Cloning repository..."
    ssh -i "$KEY_NAME.pem" -o StrictHostKeyChecking=no ec2-user@"$PUBLIC_IP" << 'EOSSH'
        cd /home/ec2-user/llama2-service
        if [ -d "train_llm" ]; then
            rm -rf train_llm
        fi
        git clone https://github.com/abhishek-shah-7/train_llm.git
        cd train_llm/llama2-inference-service
EOSSH
    
    echo "📁 Copying model files..."
    # Copy the trained model
    scp -i "$KEY_NAME.pem" -o StrictHostKeyChecking=no -r \
        ./models/dialogpt-claude-lora-cpu \
        ec2-user@"$PUBLIC_IP":/home/ec2-user/llama2-service/train_llm/llama2-inference-service/models/
    
    echo "📦 Installing dependencies and starting service..."
    ssh -i "$KEY_NAME.pem" -o StrictHostKeyChecking=no ec2-user@"$PUBLIC_IP" << EOSSH
        cd /home/ec2-user/llama2-service/train_llm/llama2-inference-service
        
        # Install Python dependencies
        python3 -m pip install --user --upgrade pip
        python3 -m pip install --user -r requirements.txt
        
        # Kill any existing service
        pkill -f "python.*app.py" || true
        
        # Start the service
        nohup python3 src/app.py > service.log 2>&1 &
        
        echo "Service started. PID: \$!"
        sleep 5
        
        # Check if service is running
        if pgrep -f "python.*app.py" > /dev/null; then
            echo "✅ Service is running"
        else
            echo "❌ Service failed to start. Check logs:"
            tail -20 service.log
        fi
EOSSH
}

# Function to test the deployment
test_deployment() {
    PUBLIC_IP=$(cat .instance_ip)
    
    echo "🧪 Testing deployment..."
    
    # Wait a bit for service to fully start
    sleep 10
    
    # Test health endpoint
    if curl -s "http://$PUBLIC_IP:$SERVICE_PORT/health" | grep -q "healthy"; then
        echo "✅ Health check passed"
    else
        echo "❌ Health check failed"
        return 1
    fi
    
    # Test inference endpoint
    echo "🤖 Testing inference..."
    RESPONSE=$(curl -s -X POST "http://$PUBLIC_IP:$SERVICE_PORT/generate" \
        -H "Content-Type: application/json" \
        -d '{"prompt": "Hello, how are you?", "max_length": 50}')
    
    if echo "$RESPONSE" | grep -q "generated_text"; then
        echo "✅ Inference test passed"
        echo "📄 Sample response: $RESPONSE"
    else
        echo "❌ Inference test failed"
        echo "📄 Response: $RESPONSE"
        return 1
    fi
}

# Function to display deployment info
show_deployment_info() {
    PUBLIC_IP=$(cat .instance_ip)
    INSTANCE_ID=$(cat .instance_id)
    
    echo ""
    echo "🎉 Deployment completed successfully!"
    echo "================================================"
    echo "📊 Instance Details:"
    echo "   • Instance ID: $INSTANCE_ID"
    echo "   • Instance Type: $INSTANCE_TYPE (8GB RAM, 32GB storage)"
    echo "   • Public IP: $PUBLIC_IP"
    echo "   • Region: $REGION"
    echo ""
    echo "🔗 API Endpoints:"
    echo "   • Health: http://$PUBLIC_IP:$SERVICE_PORT/health"
    echo "   • Generate: http://$PUBLIC_IP:$SERVICE_PORT/generate"
    echo ""
    echo "💰 Estimated Monthly Cost:"
    echo "   • Instance: ~\$60/month (t3.large)"
    echo "   • Storage: ~\$3.20/month (32GB gp3)"
    echo "   • Total: ~\$63.20/month"
    echo ""
    echo "🔧 Management Commands:"
    echo "   • SSH: ssh -i $KEY_NAME.pem ec2-user@$PUBLIC_IP"
    echo "   • Stop: aws ec2 stop-instances --instance-ids $INSTANCE_ID --region $REGION"
    echo "   • Terminate: aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION"
    echo "================================================"
}

# Main deployment process
main() {
    echo "🔍 Checking prerequisites..."
    check_aws_cli
    
    if [ ! -f "$KEY_NAME.pem" ]; then
        echo "❌ Key file $KEY_NAME.pem not found in current directory"
        exit 1
    fi
    
    chmod 400 "$KEY_NAME.pem"
    
    echo "🔧 Setting up AWS resources..."
    create_security_group
    
    echo "🚀 Launching optimized instance..."
    launch_instance
    
    echo "📦 Deploying application..."
    deploy_application
    
    echo "🧪 Testing deployment..."
    if test_deployment; then
        show_deployment_info
    else
        echo "❌ Deployment test failed. Check the service manually."
        PUBLIC_IP=$(cat .instance_ip)
        echo "SSH command: ssh -i $KEY_NAME.pem ec2-user@$PUBLIC_IP"
    fi
}

# Run main function
main "$@"
