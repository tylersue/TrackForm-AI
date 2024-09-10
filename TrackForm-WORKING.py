# Existing vid --> new vid:
# python3 TrackForm-WORKING.py --input path/to/input_video.mp4 --output path/to/output_video.mp4

# Webcam --> output vid:
# python3 TrackForm-WORKING.py --output path/to/output_video.mp4

import cv2
import mediapipe as mp
import numpy as np
import argparse

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

# Get video properties for output
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Initialize video writer if output is specified
if args.output:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

# Define ideal angle ranges for sprinting
IDEAL_ANGLES = {
    'hip': (85, 175),  # Updated hip flexion angle
    'knee': (90, 170),  # Knee flexion during recovery phase
    'ankle': (115, 140),  # Ankle dorsiflexion (shin to top of foot, that's why angle is so large)
    'armpit': (40, 60),  # Arm swing angle
    'elbow': (35, 90),  # Elbow flexion
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
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2, cv2.LINE_AA)  # Font size increased to 1.5

def is_angle_correct(angle, joint):
    if joint in IDEAL_ANGLES:
        return IDEAL_ANGLES[joint][0] <= angle <= IDEAL_ANGLES[joint][1]
    return True  # For joints we're not checking, assume correct

# Global variable to keep track of feedback messages
feedback_messages = []

def provide_feedback(joint, angle, is_correct):
    global feedback_messages
    if not is_correct:
        joint_type = joint.split()[-1]  # Extract joint type (e.g., "armpit" from "left armpit")
        ideal_range = IDEAL_ANGLES.get(joint_type, (0, 180))  # Get ideal range, default to (0, 180) if not found
        
        if angle < ideal_range[0]:
            action = "Raise"
            difference = ideal_range[0] - angle
        elif angle > ideal_range[1]:
            action = "Lower"
            difference = angle - ideal_range[1]
        else:
            action = "Adjust"  # Fallback, should rarely occur
            difference = 0
        
        feedback = f"{action} {joint} by {difference:.1f}°"
        feedback_messages.append(feedback)

def display_feedback(frame):
    y_offset = 50  # Increased initial y_offset
    for i, message in enumerate(feedback_messages):
        cv2.putText(frame, message, (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        y_offset += 40  # Increased spacing between lines

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

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

        # Calculate angles
        # Left armpit
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        left_armpit_angle = calculate_angle(left_elbow, left_shoulder, left_hip)
        is_left_armpit_correct = is_angle_correct(left_armpit_angle, 'armpit')
        draw_angle(frame, left_shoulder, left_armpit_angle, is_left_armpit_correct)
        provide_feedback('left armpit', left_armpit_angle, is_left_armpit_correct)

        # Right armpit
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        right_armpit_angle = calculate_angle(right_elbow, right_shoulder, right_hip)
        is_right_armpit_correct = is_angle_correct(right_armpit_angle, 'armpit')
        draw_angle(frame, right_shoulder, right_armpit_angle, is_right_armpit_correct)
        provide_feedback('right armpit', right_armpit_angle, is_right_armpit_correct)

        # Left elbow
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        is_left_elbow_correct = is_angle_correct(left_elbow_angle, 'elbow')
        draw_angle(frame, left_elbow, left_elbow_angle, is_left_elbow_correct)
        provide_feedback('left elbow', left_elbow_angle, is_left_elbow_correct)

        # Right elbow
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        is_right_elbow_correct = is_angle_correct(right_elbow_angle, 'elbow')
        draw_angle(frame, right_elbow, right_elbow_angle, is_right_elbow_correct)
        provide_feedback('right elbow', right_elbow_angle, is_right_elbow_correct)

        # Left knee
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        is_left_knee_correct = is_angle_correct(left_knee_angle, 'knee')
        draw_angle(frame, left_knee, left_knee_angle, is_left_knee_correct)
        provide_feedback('left knee', left_knee_angle, is_left_knee_correct)

        # Right knee
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        is_right_knee_correct = is_angle_correct(right_knee_angle, 'knee')
        draw_angle(frame, right_knee, right_knee_angle, is_right_knee_correct)
        provide_feedback('right knee', right_knee_angle, is_right_knee_correct)

        # Left hip
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
        is_left_hip_correct = is_angle_correct(left_hip_angle, 'hip')
        draw_angle(frame, left_hip, left_hip_angle, is_left_hip_correct)
        provide_feedback('left hip', left_hip_angle, is_left_hip_correct)

        # Right hip
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
        is_right_hip_correct = is_angle_correct(right_hip_angle, 'hip')
        draw_angle(frame, right_hip, right_hip_angle, is_right_hip_correct)
        provide_feedback('right hip', right_hip_angle, is_right_hip_correct)

        # Left ankle
        left_toe = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
        left_ankle_angle = calculate_angle(left_knee, left_ankle, left_toe)
        is_left_ankle_correct = is_angle_correct(left_ankle_angle, 'ankle')
        draw_angle(frame, left_ankle, left_ankle_angle, is_left_ankle_correct)
        provide_feedback('left ankle', left_ankle_angle, is_left_ankle_correct)

        # Right ankle
        right_toe = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]
        right_ankle_angle = calculate_angle(right_knee, right_ankle, right_toe)
        is_right_ankle_correct = is_angle_correct(right_ankle_angle, 'ankle')
        draw_angle(frame, right_ankle, right_ankle_angle, is_right_ankle_correct)
        provide_feedback('right ankle', right_ankle_angle, is_right_ankle_correct)

    # Display all feedback after processing all joints
    display_feedback(frame)

    # Display the frame
    cv2.imshow('Sprinting Form Analysis', frame)

    # Write the frame to the output video if specified
    if args.output:
        out.write(frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture, writer, and close windows
cap.release()
if args.output:
    out.release()
cv2.destroyAllWindows()
