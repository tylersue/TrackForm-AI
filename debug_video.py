#!/usr/bin/env python3
"""
Debug script to test video processing
"""

import requests
import os

def test_video_upload():
    # Test with a small video file or create a dummy file
    print("Testing video upload to backend...")
    
    # Create a dummy video file for testing
    dummy_video_path = "test_video.mp4"
    
    # Check if we have any video files in the current directory
    video_files = [f for f in os.listdir('.') if f.endswith(('.mp4', '.mov', '.avi'))]
    
    if video_files:
        test_file = video_files[0]
        print(f"Using existing video file: {test_file}")
    else:
        print("No video files found. Creating dummy file for testing...")
        # Create a minimal dummy file
        with open(dummy_video_path, 'wb') as f:
            f.write(b'dummy video content')
        test_file = dummy_video_path
    
    try:
        with open(test_file, 'rb') as f:
            files = {'video': (test_file, f, 'video/mp4')}
            
            print("Uploading to backend...")
            response = requests.post('http://localhost:5001/analyze', files=files, timeout=30)
            
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.text}")
            
            if response.status_code == 200:
                print("✓ Video upload successful!")
                result = response.json()
                print(f"Overall Score: {result.get('overall_score', 'N/A')}")
            else:
                print("✗ Video upload failed")
                
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up dummy file
        if test_file == dummy_video_path and os.path.exists(dummy_video_path):
            os.remove(dummy_video_path)

if __name__ == "__main__":
    test_video_upload()
