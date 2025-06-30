#!/usr/bin/env python3
"""
Load test for the LLaMA2 Inference Service
Tests performance under concurrent requests
"""

import requests
import time
import threading
import json
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

class LoadTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = []
        self.errors = []
    
    def single_request(self, prompt: str, request_id: int) -> dict:
        """Perform a single request and measure response time"""
        start_time = time.time()
        
        try:
            payload = {
                "prompt": f"{prompt} (Request #{request_id})",
                "max_length": 100,
                "temperature": 0.7
            }
            
            response = requests.post(f"{self.base_url}/predict", json=payload, timeout=60)
            end_time = time.time()
            
            return {
                "request_id": request_id,
                "status_code": response.status_code,
                "response_time": end_time - start_time,
                "success": response.status_code == 200,
                "response_length": len(response.text) if response.status_code == 200 else 0
            }
            
        except Exception as e:
            end_time = time.time()
            return {
                "request_id": request_id,
                "status_code": 0,
                "response_time": end_time - start_time,
                "success": False,
                "error": str(e),
                "response_length": 0
            }
    
    def run_load_test(self, num_requests: int = 10, num_threads: int = 3, prompt: str = "Explain machine learning"):
        """Run load test with specified parameters"""
        print(f"🚀 Starting load test:")
        print(f"   Requests: {num_requests}")
        print(f"   Threads: {num_threads}")
        print(f"   Prompt: {prompt[:50]}...")
        print("-" * 50)
        
        start_test_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit all requests
            futures = []
            for i in range(num_requests):
                future = executor.submit(self.single_request, prompt, i + 1)
                futures.append(future)
            
            # Collect results
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                self.results.append(result)
                completed += 1
                
                if result["success"]:
                    print(f"✅ Request {result['request_id']:2d}: {result['response_time']:.2f}s")
                else:
                    error_msg = result.get('error', f"HTTP {result['status_code']}")
                    print(f"❌ Request {result['request_id']:2d}: {error_msg}")
                    self.errors.append(result)
        
        total_test_time = time.time() - start_test_time
        
        # Calculate statistics
        successful_requests = [r for r in self.results if r["success"]]
        response_times = [r["response_time"] for r in successful_requests]
        
        print("\n" + "=" * 50)
        print("📊 LOAD TEST RESULTS")
        print("=" * 50)
        
        print(f"Total test time: {total_test_time:.2f}s")
        print(f"Total requests: {num_requests}")
        print(f"Successful requests: {len(successful_requests)}")
        print(f"Failed requests: {len(self.errors)}")
        print(f"Success rate: {len(successful_requests)/num_requests*100:.1f}%")
        
        if response_times:
            print(f"\nResponse Time Statistics:")
            print(f"  Average: {statistics.mean(response_times):.2f}s")
            print(f"  Median: {statistics.median(response_times):.2f}s")
            print(f"  Min: {min(response_times):.2f}s")
            print(f"  Max: {max(response_times):.2f}s")
            print(f"  Std Dev: {statistics.stdev(response_times) if len(response_times) > 1 else 0:.2f}s")
            
            # Requests per second
            rps = len(successful_requests) / total_test_time
            print(f"  Requests/second: {rps:.2f}")
        
        if self.errors:
            print(f"\n❌ Errors encountered:")
            for error in self.errors[:5]:  # Show first 5 errors
                error_msg = error.get('error', f"HTTP {error['status_code']}")
                print(f"  Request {error['request_id']}: {error_msg}")
            if len(self.errors) > 5:
                print(f"  ... and {len(self.errors) - 5} more errors")
        
        return len(self.errors) == 0

def main():
    """Main function for load testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Load test LLaMA2 Inference Service")
    parser.add_argument("--url", default="http://localhost:8000", 
                        help="Service URL (default: http://localhost:8000)")
    parser.add_argument("--requests", type=int, default=10,
                        help="Number of requests (default: 10)")
    parser.add_argument("--threads", type=int, default=3,
                        help="Number of concurrent threads (default: 3)")
    parser.add_argument("--prompt", default="Explain machine learning in simple terms",
                        help="Prompt to test with")
    
    args = parser.parse_args()
    
    # Check if service is available
    try:
        response = requests.get(f"{args.url}/health", timeout=5)
        if response.status_code != 200:
            print(f"❌ Service not available at {args.url}")
            return 1
    except:
        print(f"❌ Cannot connect to service at {args.url}")
        return 1
    
    print("✅ Service is available, starting load test...")
    
    tester = LoadTester(args.url)
    success = tester.run_load_test(args.requests, args.threads, args.prompt)
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
