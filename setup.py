#!/usr/bin/env python3
"""
Setup Script for REE Discovery System
Installs dependencies and sets up the environment.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False

def main():
    """Setup the REE discovery system."""
    print("="*60)
    print("REE DISCOVERY SYSTEM SETUP")
    print("="*60)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    
    print(f"✓ Python version: {sys.version}")
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing Python packages"):
        print("❌ Failed to install requirements")
        return False
    
    # Check Google Earth Engine
    print("\nChecking Google Earth Engine...")
    try:
        import ee
        print("✓ Earth Engine API available")
    except ImportError:
        print("❌ Earth Engine API not found. Installing...")
        if not run_command("pip install earthengine-api", "Installing Earth Engine API"):
            return False
    
    # Create directories
    directories = ['data', 'models', 'src']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    # Check authentication
    print("\nChecking Google Earth Engine authentication...")
    try:
        import ee
        ee.Initialize()
        print("✓ Google Earth Engine authenticated")
    except Exception as e:
        print("⚠ Google Earth Engine not authenticated")
        print("Please run: earthengine authenticate")
        print("Then follow the authentication instructions")
    
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Authenticate with Google Earth Engine:")
    print("   earthengine authenticate")
    print("\n2. Run the complete pipeline:")
    print("   python run_pipeline.py")
    print("\n3. Launch the visualization app:")
    print("   streamlit run src/visualization_app.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
