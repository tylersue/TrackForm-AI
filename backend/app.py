from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import tempfile
import uuid
from werkzeug.utils import secure_filename
from pose_analyzer import PoseAnalyzer
from scoring import SprintScorer

app = Flask(__name__)
CORS(app)  # Enable CORS for Flutter app

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize analyzers
pose_analyzer = PoseAnalyzer()
scorer = SprintScorer()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Serve the web interface"""
    return send_from_directory('../web_test', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files from web_test directory"""
    return send_from_directory('../web_test', filename)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'TrackForm AI Backend is running'})

@app.route('/analyze', methods=['POST'])
def analyze_video():
    """Main endpoint to analyze sprint video"""
    try:
        # Check if file was uploaded
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload MP4, AVI, MOV, MKV, or WebM'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(file_path)
        
        try:
            # Validate video file
            import cv2
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                raise ValueError(f"Invalid video file: {file_path}")
            cap.release()
            
            # Analyze the video
            print(f"Analyzing video: {file_path}")
            video_analysis = pose_analyzer.analyze_video(file_path)
            
            # Calculate score
            print("Calculating score...")
            score_results = scorer.calculate_overall_score(video_analysis)
            
            # Combine results
            results = {
                'success': True,
                'overall_score': score_results['overall_score'],
                'breakdown': score_results['breakdown'],
                'feedback': score_results['feedback'],
                'sprint_stats': score_results['sprint_stats'],
                'video_info': {
                    'duration': video_analysis['duration'],
                    'total_frames': video_analysis['total_frames'],
                    'fps': video_analysis['fps']
                },
                'frame_analysis': video_analysis['frame_results'][:10]  # First 10 frames for preview
            }
            
            print(f"Analysis complete. Score: {score_results['overall_score']}")
            return jsonify(results)
            
        except Exception as e:
            print(f"Error analyzing video: {str(e)}")
            return jsonify({'error': f'Failed to analyze video: {str(e)}'}), 500
        
        finally:
            # Clean up uploaded file
            try:
                os.remove(file_path)
            except:
                pass
    
    except Exception as e:
        print(f"Error in analyze_video: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/analyze/detailed', methods=['POST'])
def analyze_video_detailed():
    """Detailed analysis endpoint with all frame data"""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(file_path)
        
        try:
            # Analyze the video
            video_analysis = pose_analyzer.analyze_video(file_path)
            
            # Calculate score
            score_results = scorer.calculate_overall_score(video_analysis)
            
            # Return detailed results with all frames
            results = {
                'success': True,
                'overall_score': score_results['overall_score'],
                'breakdown': score_results['breakdown'],
                'feedback': score_results['feedback'],
                'sprint_stats': score_results['sprint_stats'],
                'video_info': {
                    'duration': video_analysis['duration'],
                    'total_frames': video_analysis['total_frames'],
                    'fps': video_analysis['fps']
                },
                'frame_analysis': video_analysis['frame_results']  # All frames
            }
            
            return jsonify(results)
            
        except Exception as e:
            return jsonify({'error': f'Failed to analyze video: {str(e)}'}), 500
        
        finally:
            # Clean up uploaded file
            try:
                os.remove(file_path)
            except:
                pass
    
    except Exception as e:
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("Starting TrackForm AI Backend...")
    print("Available endpoints:")
    print("  GET  /health - Health check")
    print("  POST /analyze - Analyze video (basic results)")
    print("  POST /analyze/detailed - Analyze video (detailed results)")
    print("\nServer starting on http://localhost:5001")
    app.run(debug=True, host='0.0.0.0', port=5001)
