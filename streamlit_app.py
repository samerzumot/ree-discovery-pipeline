#!/usr/bin/env python3
"""
REE Discovery Pipeline - Streamlit Cloud Deployment
Main entry point for Streamlit Cloud deployment.
"""

import sys
import os

# Add src directory to path
sys.path.append('src')

# Import and run the visualization app
from visualization_app import REEVisualizationApp

if __name__ == "__main__":
    app = REEVisualizationApp()
    app.run_app()
