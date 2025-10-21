#!/usr/bin/env python3
"""
Master Pipeline Execution Script
Runs the complete REE discovery pipeline in sequence.
"""

import subprocess
import sys
import os
import time
from datetime import datetime

def run_script(script_path, description):
    """Run a Python script and handle errors."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, check=True)
        print("✓ SUCCESS")
        if result.stdout:
            print("Output:", result.stdout[-500:])  # Last 500 chars
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ ERROR: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout[-500:])
        if e.stderr:
            print("STDERR:", e.stderr[-500:])
        return False
    except Exception as e:
        print(f"✗ UNEXPECTED ERROR: {e}")
        return False

def check_requirements():
    """Check if required files and dependencies are available."""
    print("Checking requirements...")
    
    # Check if GEE is authenticated
    try:
        import ee
        ee.Initialize(project='robotic-rampart-474204-c0')
        print("✓ Google Earth Engine authenticated")
    except Exception as e:
        print(f"✗ Google Earth Engine not authenticated: {e}")
        print("Please run: earthengine authenticate")
        return False
    
    # Check Python packages
    required_packages = [
        'pandas', 'numpy', 'geopandas', 'sklearn', 
        'folium', 'streamlit', 'plotly', 'requests'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package}")
    
    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        print("Please install: pip install -r requirements.txt")
        return False
    
    print("✓ All requirements satisfied")
    return True

def main():
    """Run the complete pipeline."""
    print("="*80)
    print("RARE EARTH ELEMENT DISCOVERY PIPELINE")
    print("="*80)
    print(f"Started at: {datetime.now()}")
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Requirements check failed. Please fix issues and try again.")
        return False
    
    # Pipeline steps (skip prediction to avoid rate limiting)
    steps = [
        ("src/gee_feature_extractor.py", "Feature Extraction from Google Earth Engine"),
        ("src/model_training.py", "Machine Learning Model Training"),
        # Skip prediction step - use existing test data
        # ("src/predict_new_areas.py", "REE Potential Prediction"),
        # ("src/validation.py", "Model Validation"),
    ]
    
    # Run each step
    success_count = 0
    for script_path, description in steps:
        if os.path.exists(script_path):
            if run_script(script_path, description):
                success_count += 1
            else:
                print(f"\n❌ Pipeline failed at: {description}")
                print("Please check the error messages above and fix issues.")
                return False
        else:
            print(f"✗ Script not found: {script_path}")
            return False
        
        # Small delay between steps
        time.sleep(2)
    
    # Summary
    print(f"\n{'='*80}")
    print("PIPELINE EXECUTION SUMMARY")
    print(f"{'='*80}")
    print(f"Completed steps: {success_count}/{len(steps)}")
    print(f"Finished at: {datetime.now()}")
    
    if success_count == len(steps):
        print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("\nNext steps:")
        print("1. Launch the visualization app:")
        print("   streamlit run src/visualization_app.py")
        print("\n2. Check the results in:")
        print("   - data/gee_features_california.csv")
        print("   - data/ree_predictions_california.geojson")
        print("   - models/ (model files and plots)")
        return True
    else:
        print("\n❌ PIPELINE FAILED")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
