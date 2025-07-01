#!/usr/bin/env python3
"""
Test script for the real LLaMA2 model deployment
This verifies that the deployed API is using the actual fine-tuned model
"""

import requests
import json
import time

def test_real_model_deployment(api_url="http://52.205.207.112:8000"):
    """Test if the deployed API is using the real model"""
    
    print("🧪 Testing Real Model Deployment")
    print(f"API URL: {api_url}")
    print("=" * 50)
    
    # Test health endpoint
    try:
        health_response = requests.get(f"{api_url}/health", timeout=5)
        if health_response.status_code == 200:
            print("✅ Health check: PASSED")
        else:
            print("❌ Health check: FAILED")
            return
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return
    
    # Test prompts to distinguish real model from demo
    test_cases = [
        {
            "prompt": "Hello",
            "expected_demo_pattern": "Based on your prompt",
            "description": "Simple greeting"
        },
        {
            "prompt": "What is AI?",
            "expected_demo_pattern": "simulated response",
            "description": "AI question"
        },
        {
            "prompt": "Write Python code",
            "expected_demo_pattern": "This is a demo response",
            "description": "Code request"
        }
    ]
    
    real_model_detected = 0
    total_tests = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔍 Test {i}/{total_tests}: {test_case['description']}")
        print(f"Prompt: '{test_case['prompt']}'")
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{api_url}/generate",
                headers={"Content-Type": "application/json"},
                json={
                    "prompt": test_case["prompt"],
                    "max_length": 100,
                    "temperature": 0.7
                },
                timeout=30
            )
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                generated_text = data.get("generated_text", "")
                processing_time = data.get("processing_time", end_time - start_time)
                
                print(f"✅ Response: '{generated_text[:100]}...'")
                print(f"⏱️  Processing time: {processing_time:.2f}s")
                
                # Check if this looks like demo response
                is_demo = any(pattern in generated_text.lower() for pattern in [
                    "based on your prompt",
                    "simulated response",
                    "demo response",
                    "here's a simulated"
                ])
                
                if is_demo:
                    print("⚠️  This looks like a DEMO response")
                else:
                    print("🎯 This looks like a REAL model response")
                    real_model_detected += 1
                    
            else:
                print(f"❌ HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"❌ Request failed: {e}")
    
    print("\n" + "=" * 50)
    print("📊 DEPLOYMENT TEST RESULTS")
    print(f"Real model responses: {real_model_detected}/{total_tests}")
    
    if real_model_detected == total_tests:
        print("🎉 SUCCESS: Real model is deployed!")
    elif real_model_detected > 0:
        print("⚠️  MIXED: Some real, some demo responses")
    else:
        print("❌ FAILED: Still using demo model")
        print("\n💡 Next steps:")
        print("1. Check deployment logs")
        print("2. Verify model files are included")
        print("3. Restart the service")
    
    return real_model_detected == total_tests

if __name__ == "__main__":
    import sys
    
    # Allow custom API URL
    api_url = sys.argv[1] if len(sys.argv) > 1 else "http://52.205.207.112:8000"
    
    success = test_real_model_deployment(api_url)
    sys.exit(0 if success else 1)
