#!/usr/bin/env python3
"""
TrackForm AI Backend Server Launcher
"""

import sys
import os
import subprocess

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import flask
        import mediapipe
        import cv2
        import numpy
        print("✓ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def main():
    print("TrackForm AI Backend Server")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not os.path.exists('app.py'):
        print("Error: app.py not found. Please run this script from the backend directory.")
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    print("\nStarting Flask server...")
    print("Server will be available at: http://localhost:5000")
    print("Press Ctrl+C to stop the server\n")
    
    try:
        # Import and run the Flask app
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n\nServer stopped by user")
    except Exception as e:
        print(f"\nError starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
