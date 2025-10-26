#!/usr/bin/env python3
"""
Test script for TrackForm AI Backend
"""

import requests
import json
import time

def test_backend():
    base_url = "http://localhost:5001"
    
    print("Testing TrackForm AI Backend")
    print("=" * 40)
    
    # Test health endpoint
    print("1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✓ Health check passed")
            print(f"  Response: {response.json()}")
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"✗ Cannot connect to backend: {e}")
        print("  Make sure the backend server is running on http://localhost:5000")
        return False
    
    print("\n2. Backend is running successfully!")
    print("   You can now:")
    print("   - Start the Flutter app")
    print("   - Record or upload sprint videos")
    print("   - Get AI-powered form analysis")
    
    return True

if __name__ == "__main__":
    test_backend()
