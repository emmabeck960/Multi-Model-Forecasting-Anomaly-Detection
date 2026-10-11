#!/usr/bin/env python3
"""
Quick test script to verify the time series analysis project works.

This script tests the basic functionality without requiring all dependencies.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_basic_imports():
    """Test basic imports."""
    print("Testing basic imports...")
    
    try:
        import pandas as pd
        import numpy as np
        print("✓ pandas and numpy imported successfully")
    except ImportError as e:
        print(f"✗ Error importing basic libraries: {e}")
        return False
    
    return True

def test_data_generation():
    """Test data generation."""
    print("\nTesting data generation...")
    
    try:
        from data_utils import DataGenerator
        
        generator = DataGenerator(random_state=42)
        data = generator.generate_trend_data(
            start_date="2020-01-01",
            end_date="2020-01-10"
        )
        
        print(f"✓ Generated {len(data)} data points")
        print(f"✓ Data columns: {list(data.columns)}")
        print(f"✓ Date range: {data['ds'].min()} to {data['ds'].max()}")
        
        return True
    except Exception as e:
        print(f"✗ Error in data generation: {e}")
        return False

def test_config_loading():
    """Test configuration loading."""
    print("\nTesting configuration loading...")
    
    try:
        import yaml
        
        config_path = Path("config/config.yaml")
        if config_path.exists():
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            print("✓ Configuration loaded successfully")
            print(f"✓ Data source: {config['data']['source']}")
            print(f"✓ Forecast periods: {config['data']['forecast_periods']}")
            
            return True
        else:
            print("✗ Configuration file not found")
            return False
    except Exception as e:
        print(f"✗ Error loading configuration: {e}")
        return False

def test_project_structure():
    """Test project structure."""
    print("\nTesting project structure...")
    
    required_dirs = ["src", "config", "tests", "notebooks", "data", "models", "results", "logs"]
    required_files = [
        "src/__init__.py",
        "src/data_utils.py", 
        "src/models.py",
        "src/visualization.py",
        "src/app.py",
        "config/config.yaml",
        "requirements.txt",
        "README.md",
        ".gitignore"
    ]
    
    all_good = True
    
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"✓ Directory {dir_name} exists")
        else:
            print(f"✗ Directory {dir_name} missing")
            all_good = False
    
    for file_name in required_files:
        if Path(file_name).exists():
            print(f"✓ File {file_name} exists")
        else:
            print(f"✗ File {file_name} missing")
            all_good = False
    
    return all_good

def main():
    """Run all tests."""
    print("=" * 60)
    print("TIME SERIES ANALYSIS PROJECT - QUICK TEST")
    print("=" * 60)
    
    tests = [
        test_project_structure,
        test_config_loading,
        test_basic_imports,
        test_data_generation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Project is ready to use.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run CLI: python cli.py")
        print("3. Launch web interface: streamlit run streamlit_app.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    print("=" * 60)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
