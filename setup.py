#!/usr/bin/env python3
"""
Data Governance Framework - Setup Script
Prepares the environment and validates the installation
"""

import sys
import subprocess
import os
from pathlib import Path

def print_banner():
    """Print setup banner"""
    print("=" * 70)
    print("[TOOLS] DATA GOVERNANCE FRAMEWORK - SETUP")
    print("=" * 70)

def check_python_version():
    """Check if Python version is compatible"""
    print("[SNAKE] Checking Python version...")

    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print(f"   [ERROR] Python {version.major}.{version.minor} detected")
        print("   [ERROR] Python 3.7+ required")
        return False

    print(f"   [CHECK] Python {version.major}.{version.minor}.{version.micro}")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\n[PACKAGE] Installing dependencies...")

    dependencies = ['pandas>=1.3.0', 'numpy>=1.20.0']

    for dep in dependencies:
        print(f"   Installing {dep}...")
        try:
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', dep
            ], capture_output=True, text=True, check=True)
            print(f"   [CHECK] {dep} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"   [ERROR] Failed to install {dep}")
            print(f"   Error: {e.stderr}")
            return False

    return True

def create_directory_structure():
    """Create required directory structure"""
    print("\n[FOLDER] Creating directory structure...")

    directories = [
        'data',
        'processed', 
        'archive',
        'test_data',
        'logs'
    ]

    for dir_name in directories:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"   [CHECK] Created {dir_name}/")
        else:
            print(f"   [CHECK] {dir_name}/ (already exists)")

    # Create .gitkeep files to ensure directories are tracked
    for dir_name in directories:
        gitkeep_path = Path(dir_name) / '.gitkeep'
        if not gitkeep_path.exists():
            gitkeep_path.touch()
            print(f"   [CHECK] Created {dir_name}/.gitkeep")

def validate_installation():
    """Validate that everything is working"""
    print("\n[SEARCH] Validating installation...")

    # Check if main files exist
    required_files = ['main.py', 'test_framework.py', 'run_tests.py']
    for file in required_files:
        if Path(file).exists():
            print(f"   [CHECK] {file}")
        else:
            print(f"   [ERROR] {file} - MISSING")
            return False

    # Try importing required modules
    try:
        import pandas as pd
        import numpy as np
        print("   [CHECK] pandas and numpy import successfully")
    except ImportError as e:
        print(f"   [ERROR] Import error: {e}")
        return False

    return True

def run_quick_test():
    """Run a quick test to ensure everything works"""
    print("\n[TEST] Running quick validation test...")

    try:
        # Import and test basic functionality
        sys.path.insert(0, '.')
        from main import create_test_data, DataGovernanceFramework

        # Create small test dataset
        df = create_test_data()
        print(f"   [CHECK] Test data created: {df.shape}")

        # Initialize framework
        framework = DataGovernanceFramework()
        print("   [CHECK] Framework initialized")

        # Run quick assessment (just field analysis)
        field_profiles = framework.identifier.analyze_field_characteristics(df.head(100))
        print(f"   [CHECK] Field analysis completed: {len(field_profiles)} fields analyzed")

        return True

    except Exception as e:
        print(f"   [ERROR] Quick test failed: {e}")
        return False

def show_next_steps():
    """Show what to do next"""
    print("\n" + "=" * 70)
    print("[TARGET] NEXT STEPS")
    print("=" * 70)
    print("\n1. Run basic functionality test:")
    print("   python run_tests.py --basic")
    print("\n2. Run comprehensive tests:")
    print("   python run_tests.py --comprehensive")
    print("\n3. Test with your own data:")
    print("   - Place CSV/Excel files in the data/ directory")
    print("   - Run: python main.py")
    print("\n4. Test with a specific file:")
    print("   python main.py --file path/to/your/file.csv")
    print("\n5. View generated reports in the processed/ directory")

def main():
    """Main setup function"""
    print_banner()

    # Check Python version
    if not check_python_version():
        return 1

    # Install dependencies
    if not install_dependencies():
        print("\n[ERROR] Failed to install dependencies")
        return 1

    # Create directory structure
    create_directory_structure()

    # Validate installation
    if not validate_installation():
        print("\n[ERROR] Installation validation failed")
        return 1

    # Run quick test
    if not run_quick_test():
        print("\n[ERROR] Quick test failed")
        return 1

    print("\n" + "=" * 70)
    print("[PARTY] SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 70)

    show_next_steps()

    return 0

if __name__ == "__main__":
    sys.exit(main())
