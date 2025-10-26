from typing import Dict, List
import numpy as np

class SprintScorer:
    def __init__(self):
        # Weighting for different aspects of sprint form
        self.weights = {
            'knee_angle': 0.25,      # 25% - Knee drive is crucial
            'elbow_angle': 0.20,     # 20% - Arm swing efficiency
            'shin_angle': 0.20,      # 20% - Ground contact angle
            'armpit_angle': 0.15,    # 15% - Trunk position
            'consistency': 0.20      # 20% - Consistency across frames
        }

    def calculate_joint_score(self, joint_angles: List[Dict], joint_name: str) -> float:
        """Calculate score for a specific joint based on angle correctness"""
        if not joint_angles:
            return 0.0
        
        correct_count = 0
        total_angles = []
        
        for frame_data in joint_angles:
            if joint_name in frame_data['angles']:
                angle_data = frame_data['angles'][joint_name]
                total_angles.append(angle_data['angle'])
                if angle_data['correct']:
                    correct_count += 1
        
        if not total_angles:
            return 0.0
        
        # Base score from correctness percentage
        correctness_score = (correct_count / len(total_angles)) * 100
        
        # Bonus for consistency (lower standard deviation = more consistent)
        if len(total_angles) > 1:
            std_dev = np.std(total_angles)
            # Normalize std_dev (assuming angles range 0-180, max std_dev would be ~90)
            consistency_bonus = max(0, 10 - (std_dev / 9))  # Up to 10 bonus points
        else:
            consistency_bonus = 0
        
        return min(100, correctness_score + consistency_bonus)

    def calculate_overall_score(self, video_analysis: Dict) -> Dict:
        """Calculate comprehensive score for the sprint video"""
        frame_results = video_analysis.get('frame_results', [])
        
        if not frame_results:
            return {
                'overall_score': 0,
                'breakdown': {},
                'feedback': ['No pose data detected in video'],
                'sprint_stats': {
                    'sprint_duration': 0,
                    'total_frames_analyzed': 0,
                    'sprint_start_time': 0
                }
            }
        
        # Filter to only sprint frames for more accurate scoring
        sprint_frames = [frame for frame in frame_results if frame.get('sprint_started', False)]
        
        if not sprint_frames:
            return {
                'overall_score': 0,
                'breakdown': {},
                'feedback': ['No sprint movement detected in video'],
                'sprint_stats': {
                    'sprint_duration': 0,
                    'total_frames_analyzed': 0,
                    'sprint_start_time': 0
                }
            }
        
        # Calculate individual joint scores
        knee_scores = []
        elbow_scores = []
        shin_scores = []
        armpit_scores = []
        
        for frame in sprint_frames:
            angles = frame['angles']
            
            # Average left and right scores for each joint type
            knee_avg = (angles.get('left_knee', {}).get('correct', False) + 
                       angles.get('right_knee', {}).get('correct', False)) / 2
            elbow_avg = (angles.get('left_elbow', {}).get('correct', False) + 
                        angles.get('right_elbow', {}).get('correct', False)) / 2
            shin_avg = (angles.get('left_shin', {}).get('correct', False) + 
                       angles.get('right_shin', {}).get('correct', False)) / 2
            armpit_avg = (angles.get('left_armpit', {}).get('correct', False) + 
                         angles.get('right_armpit', {}).get('correct', False)) / 2
            
            knee_scores.append(knee_avg)
            elbow_scores.append(elbow_avg)
            shin_scores.append(shin_avg)
            armpit_scores.append(armpit_avg)
        
        # Calculate weighted scores
        knee_score = np.mean(knee_scores) * 100 if knee_scores else 0
        elbow_score = np.mean(elbow_scores) * 100 if elbow_scores else 0
        shin_score = np.mean(shin_scores) * 100 if shin_scores else 0
        armpit_score = np.mean(armpit_scores) * 100 if armpit_scores else 0
        
        # Calculate consistency score
        all_scores = knee_scores + elbow_scores + shin_scores + armpit_scores
        consistency_score = (1 - np.std(all_scores)) * 100 if len(all_scores) > 1 else 100
        
        # Calculate weighted overall score
        overall_score = (
            knee_score * self.weights['knee_angle'] +
            elbow_score * self.weights['elbow_angle'] +
            shin_score * self.weights['shin_angle'] +
            armpit_score * self.weights['armpit_angle'] +
            consistency_score * self.weights['consistency']
        )
        
        # Generate feedback
        feedback = self.generate_feedback(knee_score, elbow_score, shin_score, armpit_score, consistency_score)
        
        return {
            'overall_score': round(overall_score, 1),
            'breakdown': {
                'knee_angle': round(knee_score, 1),
                'elbow_angle': round(elbow_score, 1),
                'shin_angle': round(shin_score, 1),
                'armpit_angle': round(armpit_score, 1),
                'consistency': round(consistency_score, 1)
            },
            'feedback': feedback,
            'sprint_stats': {
                'sprint_duration': video_analysis.get('sprint_duration', 0),
                'total_frames_analyzed': len(sprint_frames),
                'sprint_start_time': sprint_frames[0]['timestamp'] if sprint_frames else 0
            }
        }
    
    def generate_feedback(self, knee_score: float, elbow_score: float, shin_score: float, 
                         armpit_score: float, consistency_score: float) -> List[str]:
        """Generate human-readable feedback based on scores"""
        feedback = []
        
        # Overall assessment
        total_score = (knee_score + elbow_score + shin_score + armpit_score + consistency_score) / 5
        if total_score >= 80:
            feedback.append("Excellent sprint form! Keep up the great work.")
        elif total_score >= 60:
            feedback.append("Good sprint form with room for improvement.")
        elif total_score >= 40:
            feedback.append("Fair sprint form. Focus on the key areas below.")
        else:
            feedback.append("Needs significant improvement. Practice the fundamentals.")
        
        # Specific feedback
        if knee_score < 60:
            feedback.append("Focus on driving your knees higher during the sprint.")
        
        if elbow_score < 60:
            feedback.append("Work on maintaining a 90-degree elbow angle for better arm drive.")
        
        if shin_score < 60:
            feedback.append("Improve your ground contact by keeping your shin more vertical.")
        
        if armpit_score < 60:
            feedback.append("Maintain better trunk position and forward lean.")
        
        if consistency_score < 60:
            feedback.append("Work on maintaining consistent form throughout your sprint.")
        
        return feedback
