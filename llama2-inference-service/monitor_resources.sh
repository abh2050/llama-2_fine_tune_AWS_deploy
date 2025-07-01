#!/bin/bash

# =============================================================================
# Resource Monitor for DialoGPT-small Model
# Monitors CPU, memory, and disk usage during model operation
# =============================================================================

echo "🔍 DialoGPT-small Model Resource Monitor"
echo "=========================================="

# Function to get memory info
get_memory_info() {
    if command -v free &> /dev/null; then
        echo "💾 Memory Usage:"
        free -h | grep -E "(Mem|Swap)" | while read line; do
            echo "   $line"
        done
        
        # Calculate memory usage percentage
        MEM_USED=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
        echo "   Memory Usage: ${MEM_USED}%"
    else
        echo "💾 Memory info not available (free command not found)"
    fi
}

# Function to get CPU info
get_cpu_info() {
    echo "🖥️  CPU Usage:"
    if command -v top &> /dev/null; then
        CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//')
        echo "   CPU Usage: ${CPU_USAGE}%"
    fi
    
    echo "   CPU Cores: $(nproc)"
    
    if [ -f /proc/cpuinfo ]; then
        CPU_MODEL=$(grep "model name" /proc/cpuinfo | head -n1 | cut -d':' -f2 | xargs)
        echo "   CPU Model: $CPU_MODEL"
    fi
}

# Function to get disk info
get_disk_info() {
    echo "💽 Disk Usage:"
    df -h / | tail -n1 | while read line; do
        echo "   Root: $line"
    done
    
    # Check model directory size if it exists
    if [ -d "./models" ]; then
        MODEL_SIZE=$(du -sh ./models 2>/dev/null | cut -f1)
        echo "   Model Size: $MODEL_SIZE"
    fi
}

# Function to monitor Python processes
get_python_processes() {
    echo "🐍 Python Processes:"
    if pgrep -f python > /dev/null; then
        ps aux | grep python | grep -v grep | while read line; do
            PID=$(echo $line | awk '{print $2}')
            CPU=$(echo $line | awk '{print $3}')
            MEM=$(echo $line | awk '{print $4}')
            CMD=$(echo $line | awk '{for(i=11;i<=NF;i++) printf "%s ", $i; print ""}')
            echo "   PID: $PID | CPU: ${CPU}% | MEM: ${MEM}% | CMD: $CMD"
        done
    else
        echo "   No Python processes running"
    fi
}

# Function to get network connections
get_network_info() {
    echo "🌐 Network Connections:"
    if command -v netstat &> /dev/null; then
        netstat -tlnp 2>/dev/null | grep :8000 | while read line; do
            echo "   $line"
        done
    elif command -v ss &> /dev/null; then
        ss -tlnp | grep :8000 | while read line; do
            echo "   $line"
        done
    else
        echo "   Network info not available"
    fi
}

# Function to estimate model requirements
show_model_requirements() {
    echo ""
    echo "📊 DialoGPT-small Model Requirements"
    echo "===================================="
    echo "Base Model: microsoft/DialoGPT-small (~340MB)"
    echo "LoRA Adapters: ~10-50MB"
    echo "PyTorch + Dependencies: ~2-3GB"
    echo ""
    echo "Recommended Specs:"
    echo "• RAM: 4-8GB (6GB recommended)"
    echo "• Disk: 16-32GB (32GB recommended)"
    echo "• CPU: 2-4 cores"
    echo "• Python: 3.8+"
    echo ""
    echo "AWS Instance Recommendations:"
    echo "• Development: t3.medium (4GB RAM, 2 vCPU) - ~$30/month"
    echo "• Production: t3.large (8GB RAM, 2 vCPU) - ~$60/month"
    echo "• High Performance: c5.xlarge (8GB RAM, 4 vCPU) - ~$120/month"
}

# Function to run continuous monitoring
monitor_continuous() {
    echo "🔄 Starting continuous monitoring (Ctrl+C to stop)..."
    echo "Updating every 10 seconds..."
    echo ""
    
    while true; do
        clear
        echo "🕐 $(date)"
        echo "=========================================="
        get_memory_info
        echo ""
        get_cpu_info
        echo ""
        get_python_processes
        echo ""
        echo "Press Ctrl+C to stop monitoring"
        sleep 10
    done
}

# Function to test model load
test_model_load() {
    echo "🧪 Testing Model Load Impact..."
    
    # Get baseline metrics
    echo "📊 Baseline metrics:"
    BASELINE_MEM=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
    echo "   Memory usage: ${BASELINE_MEM}%"
    
    # Test API call if service is running
    if curl -s http://localhost:8000/health &> /dev/null; then
        echo "🔥 Service is running. Testing inference..."
        
        RESPONSE=$(curl -s -X POST "http://localhost:8000/generate" \
            -H "Content-Type: application/json" \
            -d '{"prompt": "Hello, how are you?", "max_length": 50}' \
            --max-time 30)
        
        if echo "$RESPONSE" | grep -q "generated_text"; then
            echo "✅ Inference successful"
            
            # Check memory after inference
            AFTER_MEM=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
            echo "   Memory after inference: ${AFTER_MEM}%"
            
            PROCESSING_TIME=$(echo "$RESPONSE" | grep -o '"processing_time":[0-9.]*' | cut -d':' -f2)
            if [ ! -z "$PROCESSING_TIME" ]; then
                echo "   Processing time: ${PROCESSING_TIME}s"
            fi
        else
            echo "❌ Inference failed"
            echo "Response: $RESPONSE"
        fi
    else
        echo "❌ Service not running on localhost:8000"
    fi
}

# Main function
main() {
    case "${1:-info}" in
        "info")
            get_memory_info
            echo ""
            get_cpu_info
            echo ""
            get_disk_info
            echo ""
            get_python_processes
            echo ""
            get_network_info
            show_model_requirements
            ;;
        "monitor")
            monitor_continuous
            ;;
        "test")
            test_model_load
            ;;
        *)
            echo "Usage: $0 [info|monitor|test]"
            echo "  info    - Show current resource usage (default)"
            echo "  monitor - Continuous monitoring"
            echo "  test    - Test model load and measure impact"
            ;;
    esac
}

main "$@"
