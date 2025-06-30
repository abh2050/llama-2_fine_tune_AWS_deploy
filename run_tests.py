#!/usr/bin/env python3
"""
Test runner for LLaMA2 Inference Service
Runs all available tests with options for different test types
"""

import os
import sys
import subprocess
import argparse

def run_command(command, description):
    """Run a command and return success status"""
    print(f"\n🚀 {description}")
    print("-" * 60)
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=False, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run LLaMA2 Inference Service tests")
    parser.add_argument("--url", default="http://localhost:8000",
                        help="Service URL (default: http://localhost:8000)")
    parser.add_argument("--quick", action="store_true",
                        help="Run quick tests only")
    parser.add_argument("--load", action="store_true",
                        help="Run load tests")
    parser.add_argument("--all", action="store_true",
                        help="Run all tests")
    parser.add_argument("--test", choices=["simple", "comprehensive", "load"],
                        help="Run specific test type")
    
    args = parser.parse_args()
    
    # Change to tests directory
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(tests_dir)
    
    print("🧪 LLaMA2 Inference Service Test Runner")
    print("=" * 60)
    print(f"Service URL: {args.url}")
    print(f"Tests directory: {tests_dir}")
    
    success_count = 0
    total_count = 0
    
    # Determine which tests to run
    run_simple = args.quick or args.all or args.test == "simple" or not any([args.quick, args.load, args.all, args.test])
    run_comprehensive = args.all or args.test == "comprehensive"
    run_load = args.load or args.all or args.test == "load"
    
    # Run simple tests
    if run_simple:
        total_count += 1
        if run_command(f"python test_api.py", "Simple API Tests"):
            success_count += 1
    
    # Run comprehensive tests
    if run_comprehensive:
        total_count += 1
        if run_command(f"python test_llm_service.py --url {args.url}", "Comprehensive LLM Tests"):
            success_count += 1
    
    # Run load tests
    if run_load:
        total_count += 1
        if run_command(f"python test_load.py --url {args.url} --requests 5 --threads 2", "Load Tests"):
            success_count += 1
    
    # Print final summary
    print("\n" + "=" * 60)
    print("📊 FINAL TEST SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {success_count}/{total_count}")
    
    if success_count == total_count:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
