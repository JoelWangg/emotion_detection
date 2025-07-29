# DeepFace Video Emotion Detection API

This Flask API accepts a video, analyzes each frame using DeepFace for emotion detection, and returns a processed video with bounding boxes and emotion labels.

## 🛠 Environment Setup

```bash
# 1. Create and activate Python virtual environment
python3.10 -m venv deepface_env
source deepface_env/bin/activate

# 2. Install required dependencies
pip install -r requirements.txt

# 3. Run the Flask server (assuming your script is test.py)
python test.py     
``` 

### API Usage
API will be available at: http://localhost:5000/process-video



