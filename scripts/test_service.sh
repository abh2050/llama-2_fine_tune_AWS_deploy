#!/bin/bash

# LLaMA2 Inference Service Test Script
# Tests all API endpoints to verify service functionality

set -e

# Configuration
SERVICE_URL="${SERVICE_URL:-http://localhost:8000}"

echo "🧪 Testing LLaMA2 Inference Service"
echo "=================================="
echo "Service URL: ${SERVICE_URL}"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test function
test_endpoint() {
    local endpoint=$1
    local method=$2
    local data=$3
    local expected_status=$4
    
    echo -n "Testing ${method} ${endpoint}... "
    
    if [ "$method" = "GET" ]; then
        response=$(curl -s -w "%{http_code}" "${SERVICE_URL}${endpoint}")
    else
        response=$(curl -s -w "%{http_code}" -X ${method} \
            -H "Content-Type: application/json" \
            -d "${data}" \
            "${SERVICE_URL}${endpoint}")
    fi
    
    status_code="${response: -3}"
    body="${response%???}"
    
    if [ "$status_code" -eq "$expected_status" ]; then
        echo -e "${GREEN}✅ PASS${NC} (${status_code})"
        return 0
    else
        echo -e "${RED}❌ FAIL${NC} (Expected: ${expected_status}, Got: ${status_code})"
        echo "Response: $body"
        return 1
    fi
}

# Wait for service to be ready
echo "⏳ Waiting for service to be ready..."
for i in {1..30}; do
    if curl -s "${SERVICE_URL}/health" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Service is ready!${NC}"
        break
    fi
    echo "   Attempt $i/30: Waiting..."
    sleep 2
done

echo ""

# Test health endpoint
test_endpoint "/health" "GET" "" 200

# Test predict endpoint
echo ""
echo "🤖 Testing text generation..."
predict_data='{
    "prompt": "What is machine learning?",
    "max_length": 100,
    "temperature": 0.7
}'
test_endpoint "/predict" "POST" "$predict_data" 200

# Test RAG endpoint
echo ""
echo "📚 Testing RAG generation..."
rag_data='{
    "query": "What is the capital of France?",
    "context": ["France is a country in Europe.", "Paris is a major city in France."],
    "max_length": 150
}'
test_endpoint "/rag" "POST" "$rag_data" 200

# Test model info endpoint
echo ""
echo "ℹ️  Testing model info..."
test_endpoint "/model/info" "GET" "" 200

# Test error cases
echo ""
echo "🚫 Testing error cases..."

# Missing prompt
invalid_data='{"max_length": 100}'
test_endpoint "/predict" "POST" "$invalid_data" 400

# Missing query for RAG
invalid_rag_data='{"context": ["test"]}'
test_endpoint "/rag" "POST" "$invalid_rag_data" 400

# Non-existent endpoint
test_endpoint "/nonexistent" "GET" "" 404

echo ""
echo "📊 Full health check response:"
curl -s "${SERVICE_URL}/health" | jq '.' 2>/dev/null || curl -s "${SERVICE_URL}/health"

echo ""
echo -e "${GREEN}🎉 All tests completed!${NC}"
echo ""
echo "🔗 Useful endpoints:"
echo "   Health: ${SERVICE_URL}/health"
echo "   Predict: ${SERVICE_URL}/predict"
echo "   RAG: ${SERVICE_URL}/rag"
echo "   Model Info: ${SERVICE_URL}/model/info"
echo ""
echo "📖 Example usage:"
echo 'curl -X POST '"${SERVICE_URL}"'/predict \'
echo '  -H "Content-Type: application/json" \'
echo '  -d '"'"'{"prompt": "Hello, how are you?", "max_length": 100}'"'"
