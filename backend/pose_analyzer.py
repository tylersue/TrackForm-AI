import cv2
import mediapipe as mp
import numpy as np
import math
from typing import List, Dict, Tuple, Optional

class PoseAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False, 
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )
        
        # Ideal angles for sprint analysis
        self.IDEAL_ANGLES = {
            'sprint': {
                'hip': (85, 175),
                'knee': (90, 170),
                'ankle': (119, 143),
                'armpit': (20, 60),
                'elbow': (35, 90),
                'shin': (35, 55),
            },
            'start': {
                'hip': (85, 165),
                'knee': (80, 110),
                'ankle': (119, 143),
                'armpit': (80, 110),
                'elbow': (160, 180),
                'shin': (35, 55),
            }
        }

    def calculate_angle(self, a: List[float], b: List[float], c: List[float]) -> float:
        """Calculate angle between three points"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle

    def calculate_shin_angle(self, knee: List[float], ankle: List[float]) -> float:
        """Calculate the angle between the shin and the vertical"""
        dx = knee[0] - ankle[0]
        dy = knee[1] - ankle[1]
        shin_angle = math.degrees(math.atan2(dx, -dy))
        
        # Ensure the angle is always positive and less than 180
        shin_angle = (shin_angle + 360) % 180
        
        return shin_angle

    def is_angle_correct(self, angle: float, joint: str, position: str) -> bool:
        """Check if angle is within ideal range"""
        if position in self.IDEAL_ANGLES and joint in self.IDEAL_ANGLES[position]:
            lower, upper = self.IDEAL_ANGLES[position][joint]
            return lower <= angle <= upper
        return True

    def is_in_block_start(self, landmarks) -> bool:
        """Check if athlete is in block start position"""
        left_hip_y = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y
        right_hip_y = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y
        left_shoulder_y = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
        right_shoulder_y = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
        
        avg_hip_y = (left_hip_y + right_hip_y) / 2
        avg_shoulder_y = (left_shoulder_y + right_shoulder_y) / 2
        
        hips_above_shoulders = avg_hip_y < avg_shoulder_y

        left_wrist_y = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y
        right_wrist_y = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y
        left_ankle_y = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y
        right_ankle_y = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
        
        avg_wrist_y = (left_wrist_y + right_wrist_y) / 2
        avg_ankle_y = (left_ankle_y + right_ankle_y) / 2
        
        hands_near_feet = abs(avg_wrist_y - avg_ankle_y) < 0.2

        return hips_above_shoulders and hands_near_feet

    def is_sprint_started(self, landmarks, prev_landmarks) -> bool:
        """Detect if sprint has started based on movement"""
        if not prev_landmarks:
            return False

        left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        left_ankle_prev = prev_landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle_prev = prev_landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        left_wrist_prev = prev_landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist_prev = prev_landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]

        ankle_displacement = max(
            math.sqrt((left_ankle.x - left_ankle_prev.x)**2 + (left_ankle.y - left_ankle_prev.y)**2),
            math.sqrt((right_ankle.x - right_ankle_prev.x)**2 + (right_ankle.y - right_ankle_prev.y)**2)
        )

        wrist_displacement = max(
            math.sqrt((left_wrist.x - left_wrist_prev.x)**2 + (left_wrist.y - left_wrist_prev.y)**2),
            math.sqrt((right_wrist.x - right_wrist_prev.x)**2 + (right_wrist.y - right_wrist_prev.y)**2)
        )

        ANKLE_THRESHOLD = 0.03
        WRIST_THRESHOLD = 0.04

        return ankle_displacement > ANKLE_THRESHOLD or wrist_displacement > WRIST_THRESHOLD

    def analyze_frame(self, landmarks, prev_landmarks=None) -> Dict:
        """Analyze a single frame and return metrics"""
        if not landmarks:
            return None

        # Determine position type
        in_block_start = self.is_in_block_start(landmarks)
        current_position = 'start' if in_block_start else 'sprint'
        
        # Check if sprint started
        sprint_started = self.is_sprint_started(landmarks, prev_landmarks) if prev_landmarks else False

        # Calculate all angles
        angles = {}
        feedback_messages = []

        # Left Elbow
        left_shoulder_coords = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                               landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow_coords = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                            landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_wrist_coords = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                            landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        left_elbow_angle = self.calculate_angle(left_shoulder_coords, left_elbow_coords, left_wrist_coords)
        is_left_elbow_correct = self.is_angle_correct(left_elbow_angle, 'elbow', current_position)
        angles['left_elbow'] = {'angle': left_elbow_angle, 'correct': is_left_elbow_correct}

        # Right Elbow
        right_shoulder_coords = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_elbow_coords = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                             landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        right_wrist_coords = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                             landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        right_elbow_angle = self.calculate_angle(right_shoulder_coords, right_elbow_coords, right_wrist_coords)
        is_right_elbow_correct = self.is_angle_correct(right_elbow_angle, 'elbow', current_position)
        angles['right_elbow'] = {'angle': right_elbow_angle, 'correct': is_right_elbow_correct}

        # Left Knee
        left_hip_coords = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                          landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
        left_knee_coords = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                           landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        left_ankle_coords = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                            landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        left_knee_angle = self.calculate_angle(left_hip_coords, left_knee_coords, left_ankle_coords)
        is_left_knee_correct = self.is_angle_correct(left_knee_angle, 'knee', current_position)
        angles['left_knee'] = {'angle': left_knee_angle, 'correct': is_left_knee_correct}

        # Right Knee
        right_hip_coords = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                           landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        right_knee_coords = [landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                            landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        right_ankle_coords = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                             landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        right_knee_angle = self.calculate_angle(right_hip_coords, right_knee_coords, right_ankle_coords)
        is_right_knee_correct = self.is_angle_correct(right_knee_angle, 'knee', current_position)
        angles['right_knee'] = {'angle': right_knee_angle, 'correct': is_right_knee_correct}

        # Left Shin
        left_shin_angle = self.calculate_shin_angle(left_knee_coords, left_ankle_coords)
        is_left_shin_correct = self.is_angle_correct(left_shin_angle, 'shin', current_position)
        angles['left_shin'] = {'angle': left_shin_angle, 'correct': is_left_shin_correct}

        # Right Shin
        right_shin_angle = self.calculate_shin_angle(right_knee_coords, right_ankle_coords)
        is_right_shin_correct = self.is_angle_correct(right_shin_angle, 'shin', current_position)
        angles['right_shin'] = {'angle': right_shin_angle, 'correct': is_right_shin_correct}

        # Left Armpit
        left_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
        left_armpit_angle = self.calculate_angle(left_elbow, left_shoulder, left_hip)
        is_left_armpit_correct = self.is_angle_correct(left_armpit_angle, 'armpit', current_position)
        angles['left_armpit'] = {'angle': left_armpit_angle, 'correct': is_left_armpit_correct}

        # Right Armpit
        right_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                         landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                      landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        right_hip = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        right_armpit_angle = self.calculate_angle(right_elbow, right_shoulder, right_hip)
        is_right_armpit_correct = self.is_angle_correct(right_armpit_angle, 'armpit', current_position)
        angles['right_armpit'] = {'angle': right_armpit_angle, 'correct': is_right_armpit_correct}

        # Generate feedback messages
        for joint_name, data in angles.items():
            if not data['correct']:
                side = joint_name.split('_')[0]
                joint_type = joint_name.split('_')[1]
                ideal_range = self.IDEAL_ANGLES.get(current_position, {}).get(joint_type, (0, 180))
                
                if data['angle'] < ideal_range[0]:
                    action = "Raise"
                    difference = ideal_range[0] - data['angle']
                elif data['angle'] > ideal_range[1]:
                    action = "Lower"
                    difference = data['angle'] - ideal_range[1]
                else:
                    continue
                
                feedback = f"{action} {side} {joint_type} by {difference:.1f}°"
                feedback_messages.append(feedback)

        return {
            'angles': angles,
            'position': current_position,
            'sprint_started': sprint_started,
            'in_block_start': in_block_start,
            'feedback_messages': feedback_messages
        }

    def analyze_video(self, video_path: str) -> Dict:
        """Analyze entire video and return comprehensive results"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_results = []
        prev_landmarks = None
        sprint_start_frame = None
        sprint_frames = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                frame_analysis = self.analyze_frame(landmarks, prev_landmarks)
                
                if frame_analysis:
                    frame_analysis['frame_number'] = len(frame_results)
                    frame_analysis['timestamp'] = len(frame_results) / fps
                    frame_results.append(frame_analysis)
                    
                    # Track sprint timing
                    if frame_analysis['sprint_started'] and sprint_start_frame is None:
                        sprint_start_frame = len(frame_results) - 1
                    
                    if frame_analysis['sprint_started']:
                        sprint_frames += 1
                
                prev_landmarks = landmarks
        
        cap.release()
        
        return {
            'total_frames': total_frames,
            'fps': fps,
            'duration': total_frames / fps if fps > 0 else 0,
            'sprint_start_frame': sprint_start_frame,
            'sprint_duration': sprint_frames / fps if fps > 0 else 0,
            'frame_results': frame_results
        }
