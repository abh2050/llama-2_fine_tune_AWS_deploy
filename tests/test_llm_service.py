#!/usr/bin/env python3
"""
Comprehensive test suite for the LLaMA2 Inference Service
Tests all endpoints with various scenarios including edge cases
"""

import requests
import json
import sys
import time
from typing import Dict, Any

class LLaMAInferenceServiceTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = []
    
    def wait_for_service(self, max_retries: int = 30, delay: int = 2) -> bool:
        """Wait for the service to be ready"""
        print(f"🔄 Waiting for service at {self.base_url} to be ready...")
        
        for attempt in range(max_retries):
            try:
                response = self.session.get(f"{self.base_url}/health", timeout=5)
                if response.status_code == 200:
                    print(f"✅ Service is ready after {attempt + 1} attempts!")
                    return True
            except requests.exceptions.RequestException:
                pass
            
            if attempt < max_retries - 1:
                print(f"   Attempt {attempt + 1}/{max_retries}: Service not ready, waiting {delay}s...")
                time.sleep(delay)
        
        print(f"❌ Service failed to start after {max_retries} attempts")
        return False
    
    def run_test(self, test_name: str, test_func) -> bool:
        """Run a single test and record results"""
        print(f"\n🧪 Testing: {test_name}")
        print("-" * 50)
        
        try:
            result = test_func()
            if result:
                print(f"✅ {test_name}: PASSED")
                self.test_results.append((test_name, True, None))
                return True
            else:
                print(f"❌ {test_name}: FAILED")
                self.test_results.append((test_name, False, "Test returned False"))
                return False
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {str(e)}")
            self.test_results.append((test_name, False, str(e)))
            return False
    
    def test_health_endpoint(self) -> bool:
        """Test the health check endpoint"""
        response = self.session.get(f"{self.base_url}/health")
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code != 200:
            print(f"Expected 200, got {response.status_code}")
            return False
        
        data = response.json()
        print(f"Response: {json.dumps(data, indent=2)}")
        
        # Check required fields
        required_fields = ["status", "model_loaded", "cuda_available"]
        for field in required_fields:
            if field not in data:
                print(f"Missing required field: {field}")
                return False
        
        if data["status"] != "healthy":
            print(f"Service status is not healthy: {data['status']}")
            return False
        
        return True
    
    def test_predict_basic(self) -> bool:
        """Test basic text prediction"""
        payload = {
            "prompt": "What is machine learning?",
            "max_length": 100,
            "temperature": 0.7
        }
        
        response = self.session.post(f"{self.base_url}/predict", json=payload)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code != 200:
            print(f"Expected 200, got {response.status_code}")
            print(f"Response: {response.text}")
            return False
        
        data = response.json()
        
        # Check required fields
        required_fields = ["response", "prompt", "parameters"]
        for field in required_fields:
            if field not in data:
                print(f"Missing required field: {field}")
                return False
        
        print(f"Generated response: {data['response'][:200]}...")
        print(f"Parameters used: {data['parameters']}")
        
        # Verify response is not empty
        if not data["response"].strip():
            print("Generated response is empty")
            return False
        
        return True
    
    def test_predict_custom_parameters(self) -> bool:
        """Test prediction with custom parameters"""
        payload = {
            "prompt": "Explain artificial intelligence in simple terms.",
            "max_length": 150,
            "temperature": 0.5,
            "top_p": 0.8
        }
        
        response = self.session.post(f"{self.base_url}/predict", json=payload)
        
        if response.status_code != 200:
            print(f"Expected 200, got {response.status_code}")
            return False
        
        data = response.json()
        
        # Verify custom parameters were used
        params = data.get("parameters", {})
        if params.get("temperature") != 0.5:
            print(f"Temperature not set correctly: expected 0.5, got {params.get('temperature')}")
            return False
        
        if params.get("top_p") != 0.8:
            print(f"Top_p not set correctly: expected 0.8, got {params.get('top_p')}")
            return False
        
        print(f"Custom parameters applied correctly: {params}")
        print(f"Generated response: {data['response'][:150]}...")
        
        return True
    
    def test_rag_endpoint(self) -> bool:
        """Test RAG (Retrieval-Augmented Generation) endpoint"""
        payload = {
            "query": "What is the capital of France?",
            "context": [
                "France is a country located in Western Europe.",
                "Paris is the largest city in France and serves as its capital.",
                "The Seine River flows through Paris."
            ],
            "max_length": 200
        }
        
        response = self.session.post(f"{self.base_url}/rag", json=payload)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code != 200:
            print(f"Expected 200, got {response.status_code}")
            print(f"Response: {response.text}")
            return False
        
        data = response.json()
        
        # Check required fields
        required_fields = ["response", "query", "context_documents", "parameters"]
        for field in required_fields:
            if field not in data:
                print(f"Missing required field: {field}")
                return False
        
        print(f"Query: {data['query']}")
        print(f"Context documents: {data['context_documents']}")
        print(f"Generated response: {data['response'][:200]}...")
        
        # Verify context was used
        if data["context_documents"] != 3:
            print(f"Expected 3 context documents, got {data['context_documents']}")
            return False
        
        return True
    
    def test_model_info(self) -> bool:
        """Test model information endpoint"""
        response = self.session.get(f"{self.base_url}/model/info")
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code != 200:
            print(f"Expected 200, got {response.status_code}")
            print(f"Response: {response.text}")
            return False
        
        data = response.json()
        print(f"Model info: {json.dumps(data, indent=2)}")
        
        # Check if model is loaded
        if not data.get("model_loaded", False):
            print("Model is not loaded according to info endpoint")
            return False
        
        return True
    
    def test_error_cases(self) -> bool:
        """Test various error scenarios"""
        # Test missing prompt
        response = self.session.post(f"{self.base_url}/predict", json={})
        if response.status_code != 400:
            print(f"Expected 400 for missing prompt, got {response.status_code}")
            return False
        print("✓ Missing prompt correctly returns 400")
        
        # Test missing query for RAG
        response = self.session.post(f"{self.base_url}/rag", json={"context": ["test"]})
        if response.status_code != 400:
            print(f"Expected 400 for missing query, got {response.status_code}")
            return False
        print("✓ Missing RAG query correctly returns 400")
        
        # Test non-existent endpoint
        response = self.session.get(f"{self.base_url}/nonexistent")
        if response.status_code != 404:
            print(f"Expected 404 for non-existent endpoint, got {response.status_code}")
            return False
        print("✓ Non-existent endpoint correctly returns 404")
        
        return True
    
    def test_conversation_flow(self) -> bool:
        """Test a conversation-like interaction"""
        messages = [
            "Hello, can you help me understand neural networks?",
            "What are the main components of a neural network?",
            "How do neural networks learn from data?"
        ]
        
        for i, message in enumerate(messages):
            print(f"Message {i+1}: {message}")
            
            payload = {
                "prompt": message,
                "max_length": 120,
                "temperature": 0.7
            }
            
            response = self.session.post(f"{self.base_url}/predict", json=payload)
            
            if response.status_code != 200:
                print(f"Failed at message {i+1}: {response.status_code}")
                return False
            
            data = response.json()
            print(f"Response {i+1}: {data['response'][:100]}...")
            print("")
        
        return True
    
    def run_all_tests(self) -> bool:
        """Run the complete test suite"""
        print("🚀 LLaMA2 Inference Service Test Suite")
        print("=" * 60)
        
        # Wait for service to be ready
        if not self.wait_for_service():
            return False
        
        # Define test cases
        test_cases = [
            ("Health Check", self.test_health_endpoint),
            ("Basic Text Generation", self.test_predict_basic),
            ("Custom Parameters", self.test_predict_custom_parameters),
            ("RAG Generation", self.test_rag_endpoint),
            ("Model Information", self.test_model_info),
            ("Error Handling", self.test_error_cases),
            ("Conversation Flow", self.test_conversation_flow),
        ]
        
        # Run all tests
        passed = 0
        total = len(test_cases)
        
        for test_name, test_func in test_cases:
            if self.run_test(test_name, test_func):
                passed += 1
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        for test_name, success, error in self.test_results:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{status:<8} {test_name}")
            if error:
                print(f"         Error: {error}")
        
        print(f"\nResults: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All tests passed! Service is working correctly.")
            return True
        else:
            print(f"❌ {total - passed} test(s) failed. Please check the service.")
            return False

def main():
    """Main function to run tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test LLaMA2 Inference Service")
    parser.add_argument("--url", default="http://localhost:8000", 
                        help="Service URL (default: http://localhost:8000)")
    parser.add_argument("--wait-time", type=int, default=30,
                        help="Max time to wait for service (default: 30s)")
    
    args = parser.parse_args()
    
    tester = LLaMAInferenceServiceTester(args.url)
    success = tester.run_all_tests()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
