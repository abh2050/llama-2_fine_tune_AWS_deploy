#!/usr/bin/env python3
"""
Simple test script for the LLaMA2 Inference Service API
Run this after starting the service to verify it's working
"""

import requests
import json
import sys

# Configuration
SERVICE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("Testing /health endpoint...")
    try:
        response = requests.get(f"{SERVICE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_predict():
    """Test prediction endpoint"""
    print("\nTesting /predict endpoint...")
    try:
        data = {
            "prompt": "What is machine learning?",
            "max_length": 100,
            "temperature": 0.7
        }
        response = requests.post(f"{SERVICE_URL}/predict", json=data)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Generated text: {result['response'][:200]}...")
        else:
            print(f"Error: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_model_info():
    """Test model info endpoint"""
    print("\nTesting /model/info endpoint...")
    try:
        response = requests.get(f"{SERVICE_URL}/model/info")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print(f"Model info: {response.json()}")
        else:
            print(f"Error: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 LLaMA2 Inference Service API Test")
    print("=" * 40)
    
    # Test all endpoints
    tests = [
        ("Health Check", test_health),
        ("Text Generation", test_predict),
        ("Model Info", test_model_info)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}")
        if test_func():
            print("✅ PASSED")
            passed += 1
        else:
            print("❌ FAILED")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed")
        sys.exit(1)
