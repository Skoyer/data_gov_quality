#!/usr/bin/env python3
"""
Data Governance Framework - Test Runner
Simple script to run tests and basic functionality checks
"""

import sys
import subprocess
import os
from pathlib import Path

def print_banner():
    """Print a nice banner"""
    print("=" * 70)
    print("[ROCKET] DATA GOVERNANCE FRAMEWORK - TEST RUNNER")
    print("=" * 70)

def check_dependencies():
    """Check if required dependencies are available"""
    print("[SEARCH] Checking dependencies...")

    required_packages = ['pandas', 'numpy']
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"   [CHECK] {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"   [ERROR] {package} - MISSING")

    if missing_packages:
        print(f"\n[ERROR] Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False

    print("   [CHECK] All dependencies available")
    return True

def run_basic_test():
    """Run the basic functionality test"""
    print("\n[TEST] Running basic functionality test...")

    try:
        # Run the main script with --test flag
        result = subprocess.run([
            sys.executable, 'main.py', '--test'
        ], capture_output=True, text=True, timeout=60)

        if result.returncode == 0:
            print("[CHECK] Basic functionality test PASSED")
            print("\nTest output:")
            print("-" * 40)
            print(result.stdout)
            return True
        else:
            print("[ERROR] Basic functionality test FAILED")
            print("\nError output:")
            print("-" * 40)
            print(result.stderr)
            return False

    except subprocess.TimeoutExpired:
        print("[ERROR] Basic functionality test TIMED OUT (>60 seconds)")
        return False
    except Exception as e:
        print(f"[ERROR] Error running basic test: {e}")
        return False

def run_comprehensive_tests():
    """Run the comprehensive test suite"""
    print("\n[TEST] Running comprehensive test suite...")

    try:
        # Run the test suite
        result = subprocess.run([
            sys.executable, 'test_framework.py'
        ], capture_output=True, text=True, timeout=120)

        if result.returncode == 0:
            print("[CHECK] Comprehensive tests PASSED")
            print("\nTest output:")
            print("-" * 40)
            print(result.stdout)
            return True
        else:
            print("[ERROR] Comprehensive tests FAILED")
            print("\nError output:")
            print("-" * 40)
            print(result.stderr)
            if result.stdout:
                print("\nStandard output:")
                print("-" * 40)
                print(result.stdout)
            return False

    except subprocess.TimeoutExpired:
        print("[ERROR] Comprehensive tests TIMED OUT (>120 seconds)")
        return False
    except Exception as e:
        print(f"[ERROR] Error running comprehensive tests: {e}")
        return False

def check_files():
    """Check if required files exist"""
    print("\n[FOLDER] Checking required files...")

    required_files = ['main.py', 'test_framework.py']
    missing_files = []

    for file in required_files:
        if Path(file).exists():
            print(f"   [CHECK] {file}")
        else:
            missing_files.append(file)
            print(f"   [ERROR] {file} - MISSING")

    if missing_files:
        print(f"\n[ERROR] Missing required files: {', '.join(missing_files)}")
        return False

    return True

def create_directories():
    """Create required directories"""
    print("\n[FOLDER] Creating required directories...")

    directories = ['data', 'processed', 'archive', 'test_data']

    for dir_name in directories:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            dir_path.mkdir(exist_ok=True)
            print(f"   [CHECK] Created {dir_name}/")
        else:
            print(f"   [CHECK] {dir_name}/ (already exists)")

def show_usage():
    """Show usage instructions"""
    print("\n" + "=" * 70)
    print("[BOOK] USAGE INSTRUCTIONS")
    print("=" * 70)
    print("\n1. Run basic functionality test:")
    print("   python run_tests.py --basic")
    print("\n2. Run comprehensive test suite:")
    print("   python run_tests.py --comprehensive")
    print("\n3. Run all tests:")
    print("   python run_tests.py --all")
    print("\n4. Process a specific file:")
    print("   python main.py --file your_data.csv")
    print("\n5. Process all files in data/ directory:")
    print("   python main.py")

def main():
    """Main function"""
    print_banner()

    # Parse command line arguments
    if len(sys.argv) < 2:
        print("[ERROR] No arguments provided")
        show_usage()
        return 1

    command = sys.argv[1].lower()

    # Check files first
    if not check_files():
        return 1

    # Check dependencies
    if not check_dependencies():
        return 1

    # Create directories
    create_directories()

    success = True

    if command == '--basic':
        success = run_basic_test()

    elif command == '--comprehensive':
        success = run_comprehensive_tests()

    elif command == '--all':
        print("\n[ROCKET] Running ALL tests...")
        basic_success = run_basic_test()
        comprehensive_success = run_comprehensive_tests()
        success = basic_success and comprehensive_success

        print("\n" + "=" * 70)
        print("[TARGET] FINAL RESULTS")
        print("=" * 70)
        print(f"Basic Test: {'[CHECK] PASSED' if basic_success else '[ERROR] FAILED'}")
        print(f"Comprehensive Tests: {'[CHECK] PASSED' if comprehensive_success else '[ERROR] FAILED'}")
        print(f"Overall: {'[CHECK] ALL TESTS PASSED' if success else '[ERROR] SOME TESTS FAILED'}")

    else:
        print(f"[ERROR] Unknown command: {command}")
        show_usage()
        return 1

    if success:
        print("\n[PARTY] SUCCESS! Framework is working correctly.")
        return 0
    else:
        print("\n[ERROR] FAILURE! Please check the error messages above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
