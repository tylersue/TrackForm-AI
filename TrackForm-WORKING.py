# Existing vid --> new vid:
# python3 TrackForm-WORKING.py --input path/to/input_video.mp4 --output path/to/output_video.mp4

# Webcam --> output vid:
# python3 TrackForm-WORKING.py --output path/to/output_video.mp4

import cv2
import mediapipe as mp
import numpy as np
import argparse
import math
import time

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--input', help='Path to input video file (optional)')
parser.add_argument('--output', help='Path to output video file')
args = parser.parse_args()

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize video capture
if args.input:
    cap = cv2.VideoCapture(args.input)
else:
    cap = cv2.VideoCapture(0)  # Use webcam if no input file is specified

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Initialize video writer if output is specified
if args.output:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

# Initialize variables for sprint detection and timing
prev_landmarks = None
sprint_started = False
sprint_start_frame = None
current_frame = 0

# Variables for timing
frame_time = 1 / fps
last_time = time.time()

# Updated IDEAL_ANGLES dictionary with separate ranges for 'sprint' and 'start'
IDEAL_ANGLES = {
    'sprint': {
        'hip': (85, 175),
        'knee': (90, 170),
        'ankle': (119, 143),
        'armpit': (20, 60),  # Adjusted for the corrected calculation
        'elbow': (35, 90),
        'shin': (35, 55),
    },
    'start': {
        'hip': (85, 165),
        'knee': (80, 110),  # Near 90 degrees for start position
        'ankle': (119, 143),
        'armpit': (80, 110),
        'elbow': (160, 180),  # Nearly straight for start position
        'shin': (35, 55),  # Adjust if necessary
    }
}

def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# New function to draw angle
def draw_angle(image, point, angle, is_correct):
    # Convert the point from ratio to pixel coordinates
    h, w, c = image.shape
    cx, cy = int(point[0] * w), int(point[1] * h)

    color = (0, 255, 0) if is_correct else (0, 0, 255)  # Green if correct, red if not
    cv2.putText(image, f"{angle:.1f}", 
                (cx, cy), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)  # Font size

def is_angle_correct(angle, joint, position):
    if position in IDEAL_ANGLES and joint in IDEAL_ANGLES[position]:
        lower, upper = IDEAL_ANGLES[position][joint]
        return lower <= angle <= upper
    return True  # For joints we're not checking, assume correct

# Updated provide_feedback function
def provide_feedback(joint, angle, is_correct):
    global feedback_messages
    if not is_correct:
        joint_parts = joint.split()
        side = joint_parts[0]  # 'left' or 'right'
        joint_type = joint_parts[1]  # e.g., 'elbow', 'knee', etc.
        ideal_range = IDEAL_ANGLES.get(current_position, {}).get(joint_type, (0, 180))  # Get ideal range

        if angle < ideal_range[0]:
            action = "Raise"
            difference = ideal_range[0] - angle
        elif angle > ideal_range[1]:
            action = "Lower"
            difference = angle - ideal_range[1]
        else:
            return  # No feedback needed if within range

        feedback = f"{action} {side} {joint_type} by {difference:.1f}°"
        feedback_messages.append(feedback)

# Function to display feedback messages
def display_feedback(frame):
    y_offset = 110  # Increased initial y_offset to start below sprint status message
    for i, message in enumerate(feedback_messages):
        cv2.putText(frame, message, (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_offset += 30  # Spacing between lines

def is_in_block_start(landmarks):
    # Check if hips are above shoulders
    left_hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
    right_hip_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
    left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
    right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
    
    avg_hip_y = (left_hip_y + right_hip_y) / 2
    avg_shoulder_y = (left_shoulder_y + right_shoulder_y) / 2
    
    hips_above_shoulders = avg_hip_y < avg_shoulder_y  # Remember, y-axis is inverted in image coordinates

    # Check if hands are at a similar y-level as the feet
    left_wrist_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y
    right_wrist_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
    left_ankle_y = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
    right_ankle_y = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
    
    avg_wrist_y = (left_wrist_y + right_wrist_y) / 2
    avg_ankle_y = (left_ankle_y + right_ankle_y) / 2
    
    hands_near_feet = abs(avg_wrist_y - avg_ankle_y) < 0.2  # Adjust threshold as needed

    return hips_above_shoulders and hands_near_feet

def calculate_shin_angle(knee, ankle):
    # Calculate the angle between the shin and the vertical
    dx = knee[0] - ankle[0]
    dy = knee[1] - ankle[1]
    shin_angle = math.degrees(math.atan2(dx, -dy))  # Note the negative dy to account for y-axis direction
    
    # Ensure the angle is always positive and less than 180
    shin_angle = (shin_angle + 360) % 180
    
    return shin_angle

def is_sprint_started(landmarks, prev_landmarks):
    if not prev_landmarks:
        return False

    # Calculate the change in position for key points
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
    left_ankle_prev = prev_landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    right_ankle_prev = prev_landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    left_wrist_prev = prev_landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_wrist_prev = prev_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

    # Calculate displacement
    ankle_displacement = max(
        math.sqrt((left_ankle.x - left_ankle_prev.x)**2 + (left_ankle.y - left_ankle_prev.y)**2),
        math.sqrt((right_ankle.x - right_ankle_prev.x)**2 + (right_ankle.y - right_ankle_prev.y)**2)
    )

    wrist_displacement = max(
        math.sqrt((left_wrist.x - left_wrist_prev.x)**2 + (left_wrist.y - left_wrist_prev.y)**2),
        math.sqrt((right_wrist.x - right_wrist_prev.x)**2 + (right_wrist.y - right_wrist_prev.y)**2)
    )

    # Thresholds for sprint detection (adjust these values based on testing)
    ANKLE_THRESHOLD = 0.03  # Significant ankle movement
    WRIST_THRESHOLD = 0.04  # Significant arm movement

    return ankle_displacement > ANKLE_THRESHOLD or wrist_displacement > WRIST_THRESHOLD

# Add these variables before the main loop
feedback_messages = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_frame += 1  # Increment frame counter
    process_start_time = time.time()

    # Clear feedback messages at the start of each frame
    feedback_messages = []

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect poses
    results = pose.process(rgb_frame)

    # Draw pose landmarks on the frame
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )

        # Get landmarks
        landmarks = results.pose_landmarks.landmark

        # Check if in block start position
        in_block_start = is_in_block_start(landmarks)

        # Check if sprint has started
        if not sprint_started:
            sprint_started = is_sprint_started(landmarks, prev_landmarks)
            if sprint_started:
                sprint_start_frame = current_frame

        # Update previous landmarks
        prev_landmarks = landmarks

        # Determine current position for angle checking
        current_position = 'start' if in_block_start else 'sprint'

        # Display sprint status
        if sprint_started:
            elapsed_frames = current_frame - sprint_start_frame
            elapsed_time = elapsed_frames / fps
            cv2.putText(frame, f"Sprinting: {elapsed_time:.2f}s", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Not Sprinting", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display block start position status
        if in_block_start:
            cv2.putText(frame, "In Block Start Position", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Not in Block Start Position", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Calculate and check angles for each joint

        # -------------------
        # Left Elbow
        # -------------------
        left_shoulder_coords = [
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
        ]
        left_elbow_coords = [
            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y
        ]
        left_wrist_coords = [
            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y
        ]
        left_elbow_angle = calculate_angle(left_shoulder_coords, left_elbow_coords, left_wrist_coords)
        is_left_elbow_correct = is_angle_correct(left_elbow_angle, 'elbow', current_position)
        draw_angle(frame, left_elbow_coords, left_elbow_angle, is_left_elbow_correct)
        provide_feedback('left elbow', left_elbow_angle, is_left_elbow_correct)

        # -------------------
        # Right Elbow
        # -------------------
        right_shoulder_coords = [
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
        ]
        right_elbow_coords = [
            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y
        ]
        right_wrist_coords = [
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
        ]
        right_elbow_angle = calculate_angle(right_shoulder_coords, right_elbow_coords, right_wrist_coords)
        is_right_elbow_correct = is_angle_correct(right_elbow_angle, 'elbow', current_position)
        draw_angle(frame, right_elbow_coords, right_elbow_angle, is_right_elbow_correct)
        provide_feedback('right elbow', right_elbow_angle, is_right_elbow_correct)

        # -------------------
        # Left Knee
        # -------------------
        left_hip_coords = [
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
        ]
        left_knee_coords = [
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
        ]
        left_ankle_coords = [
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
        ]
        left_knee_angle = calculate_angle(left_hip_coords, left_knee_coords, left_ankle_coords)
        is_left_knee_correct = is_angle_correct(left_knee_angle, 'knee', current_position)
        draw_angle(frame, left_knee_coords, left_knee_angle, is_left_knee_correct)
        provide_feedback('left knee', left_knee_angle, is_left_knee_correct)

        # -------------------
        # Right Knee
        # -------------------
        right_hip_coords = [
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
        ]
        right_knee_coords = [
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y
        ]
        right_ankle_coords = [
            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
        ]
        right_knee_angle = calculate_angle(right_hip_coords, right_knee_coords, right_ankle_coords)
        is_right_knee_correct = is_angle_correct(right_knee_angle, 'knee', current_position)
        draw_angle(frame, right_knee_coords, right_knee_angle, is_right_knee_correct)
        provide_feedback('right knee', right_knee_angle, is_right_knee_correct)

        # -------------------
        # Left Shin
        # -------------------
        left_knee_coords = [
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
        ]
        left_ankle_coords = [
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
        ]
        left_shin_angle = calculate_shin_angle(left_knee_coords, left_ankle_coords)
        is_left_shin_correct = is_angle_correct(left_shin_angle, 'shin', current_position)
        draw_angle(frame, left_knee_coords, left_shin_angle, is_left_shin_correct)
        provide_feedback('left shin', left_shin_angle, is_left_shin_correct)

        # -------------------
        # Right Shin
        # -------------------
        right_knee_coords = [
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y
        ]
        right_ankle_coords = [
            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
        ]
        right_shin_angle = calculate_shin_angle(right_knee_coords, right_ankle_coords)
        is_right_shin_correct = is_angle_correct(right_shin_angle, 'shin', current_position)
        draw_angle(frame, right_knee_coords, right_shin_angle, is_right_shin_correct)
        provide_feedback('right shin', right_shin_angle, is_right_shin_correct)

        # -------------------
        # Left Armpit
        # -------------------
        left_shoulder = [
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
        ]
        left_elbow = [
            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y
        ]
        left_hip = [
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
        ]
        # Calculate angle at the armpit using shoulder as the vertex
        left_armpit_angle = calculate_angle(left_elbow, left_shoulder, left_hip)
        is_left_armpit_correct = is_angle_correct(left_armpit_angle, 'armpit', current_position)
        draw_angle(frame, left_shoulder, left_armpit_angle, is_left_armpit_correct)
        provide_feedback('left armpit', left_armpit_angle, is_left_armpit_correct)

        # -------------------
        # Right Armpit
        # -------------------
        right_shoulder = [
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
        ]
        right_elbow = [
            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y
        ]
        right_hip = [
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
        ]
        # Calculate angle at the armpit using shoulder as the vertex
        right_armpit_angle = calculate_angle(right_elbow, right_shoulder, right_hip)
        is_right_armpit_correct = is_angle_correct(right_armpit_angle, 'armpit', current_position)
        draw_angle(frame, right_shoulder, right_armpit_angle, is_right_armpit_correct)
        provide_feedback('right armpit', right_armpit_angle, is_right_armpit_correct)

    # Display all feedback after processing all joints
    display_feedback(frame)

    # Display the frame
    cv2.imshow('Sprinting Form Analysis', frame)

    # Write the frame to the output video if specified
    if args.output:
        out.write(frame)

    # Calculate processing time and delay if necessary
    process_time = time.time() - process_start_time
    if process_time < frame_time:
        time.sleep(frame_time - process_time)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture, writer, and close windows
cap.release()
if args.output:
    out.release()
cv2.destroyAllWindows()
