# TrackForm AI MVP

A mobile app that analyzes sprint form using AI-powered pose estimation, providing real-time feedback on body angles and performance scoring.

## Architecture

- **Frontend**: Flutter mobile app (iOS/Android)
- **Backend**: Python Flask server with MediaPipe pose estimation
- **Analysis**: Real-time biomechanics analysis with scoring

## Features

- Record or upload sprint videos
- AI-powered pose estimation using MediaPipe
- Real-time angle analysis (knee, elbow, shin, armpit)
- Sprint detection and timing
- Comprehensive scoring system (0-100)
- Detailed feedback and improvement tips
- Beautiful, modern UI

## Setup Instructions

### Backend Setup

1. **Install Python dependencies**:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. **Start the Flask server**:
   ```bash
   python app.py
   ```
   The server will run on `http://localhost:5000`

### Mobile App Setup

1. **Install Flutter** (if not already installed):
   - Follow the official Flutter installation guide for your platform
   - Ensure Flutter is in your PATH

2. **Navigate to mobile directory**:
   ```bash
   cd mobile
   ```

3. **Install dependencies**:
   ```bash
   flutter pub get
   ```

4. **Run the app**:
   ```bash
   flutter run
   ```

## API Endpoints

- `GET /health` - Health check
- `POST /analyze` - Analyze video (basic results)
- `POST /analyze/detailed` - Analyze video (detailed results)

## Scoring System

The app evaluates sprint form based on:

- **Knee Angle** (25%): Optimal drive phase angle 90-170°
- **Elbow Angle** (20%): 90° elbow angle for efficient arm swing
- **Shin Angle** (20%): Ground contact angle 35-55°
- **Armpit Angle** (15%): Trunk position and forward lean
- **Consistency** (20%): Form consistency across frames

## Usage

1. **Start the backend server** (Python Flask)
2. **Launch the Flutter app**
3. **Record or upload a sprint video**
4. **View detailed analysis results** with scores and feedback

## Technical Details

### Backend
- Flask web framework
- MediaPipe for pose estimation
- OpenCV for video processing
- NumPy for calculations

### Frontend
- Flutter framework
- Camera integration
- File picker for video uploads
- HTTP client for API communication
- Material Design 3 UI

## File Structure

```
TrackForm AI (app)/
├── backend/
│   ├── app.py              # Flask server
│   ├── pose_analyzer.py    # MediaPipe pose analysis
│   ├── scoring.py          # Scoring algorithm
│   └── requirements.txt    # Python dependencies
├── mobile/
│   ├── lib/
│   │   ├── main.dart       # App entry point
│   │   ├── models/         # Data models
│   │   ├── screens/        # UI screens
│   │   └── services/       # API service
│   └── pubspec.yaml        # Flutter dependencies
└── README.md
```

## Development Notes

- Backend runs on localhost:5000 for MVP
- Videos processed synchronously
- MediaPipe provides fast, accurate pose detection
- Flutter camera plugin enables cross-platform recording

## Next Steps

- Add async video processing queue
- Implement user accounts and history
- Add more detailed biomechanics analysis
- Deploy to cloud infrastructure
- Add real-time analysis during recording
