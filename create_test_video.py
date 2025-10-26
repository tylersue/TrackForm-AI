#!/usr/bin/env python3
"""
Create a simple test video for testing the TrackForm AI system
"""

import cv2
import numpy as np
import os

def create_test_video():
    # Create a simple test video with a moving rectangle (simulating a person)
    output_path = "test_sprint_video.mp4"
    
    # Video properties
    width, height = 640, 480
    fps = 30
    duration = 3  # 3 seconds
    total_frames = fps * duration
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Creating test video: {output_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps}, Duration: {duration}s")
    
    for frame_num in range(total_frames):
        # Create a black frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add some background elements
        cv2.rectangle(frame, (50, 400), (width-50, 450), (100, 100, 100), -1)  # Ground
        
        # Simulate a simple "person" with moving rectangles
        t = frame_num / total_frames
        
        # Body (moving left to right)
        body_x = int(100 + t * 400)
        body_y = 300
        cv2.rectangle(frame, (body_x-20, body_y), (body_x+20, body_y+100), (0, 255, 0), -1)
        
        # Head
        cv2.circle(frame, (body_x, body_y-30), 25, (0, 255, 0), -1)
        
        # Arms (moving up and down)
        arm_swing = np.sin(t * 4 * np.pi) * 30
        cv2.rectangle(frame, (body_x-40, body_y+20), (body_x-20, body_y+60), (0, 255, 0), -1)
        cv2.rectangle(frame, (body_x+20, body_y+20), (body_x+40, body_y+60), (0, 255, 0), -1)
        
        # Legs (alternating)
        leg_offset = int(np.sin(t * 6 * np.pi) * 20)
        cv2.rectangle(frame, (body_x-15, body_y+100), (body_x-5, body_y+150), (0, 255, 0), -1)
        cv2.rectangle(frame, (body_x+5, body_y+100), (body_x+15, body_y+150), (0, 255, 0), -1)
        
        # Add some text
        cv2.putText(frame, f"Test Sprint Video - Frame {frame_num+1}/{total_frames}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Time: {t:.2f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        out.write(frame)
    
    out.release()
    print(f"✓ Test video created: {output_path}")
    print(f"File size: {os.path.getsize(output_path)} bytes")
    
    return output_path

if __name__ == "__main__":
    create_test_video()
