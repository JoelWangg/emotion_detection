from flask import Flask, request, send_file, jsonify
import cv2
import os
import uuid
from deepface import DeepFace
import boto3
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Ensure folders exist
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

EMOTION_COLORS = {
    'happy': (0, 255, 0),        # green
    'sad': (255, 0, 0),          # blue
    'angry': (0, 0, 255),        # red
    'surprise': (255, 255, 0),   # cyan
    'fear': (128, 0, 128),       # purple
    'disgust': (0, 255, 255),    # yellow
    'neutral': (200, 200, 200)   # grey
}

# Get AWS credentials from environment variables
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION')
bucket_name = os.getenv('BUCKET_NAME')



# Ensure AWS credentials are available
if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, bucket_name]):
    raise ValueError("AWS credentials or bucket name are not set in environment variables.")

# Configure S3 connection
s3 = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

@app.route('/process-video', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return {'error': 'No video file provided'}, 400

    video_file = request.files['video']
    unique_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_FOLDER, f'temp_{unique_id}.mp4')
    output_path = os.path.join(OUTPUT_FOLDER, f'processed_{unique_id}.mp4')

    video_file.save(input_path)
    print(f"[INFO] Saved input video to: {input_path}")

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 25

    ret, frame = cap.read()
    if not ret or frame is None:
        return {'error': 'Failed to read video frames.'}, 500

    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    print(f"[INFO] VideoWriter initialized with width={width}, height={height}, fps={fps}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_count = 0
    emotions_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_count += 1


        if frame.shape[1] != width or frame.shape[0] != height:
            print(f"[WARN] Frame {frame_count} size mismatch")
            frame = cv2.resize(frame, (width, height))

        
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            for face in result:
                x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']

                        # ✅ Skip if region is empty or covers nearly the whole frame (false detection)
                frame_height, frame_width = frame.shape[:2]
                if w == 0 or h == 0:
                    print(f"[FRAME {frame_count}] Empty face region — skipping.")
                    emotions_data.append({
                    "frame": frame_count,
                    "face_detected": False,
                    "emotion": None,
                    "confidence": None,
                    "x": None,
                    "y": None,
                    "width": None,
                    "height": None
                    })
                    continue
                if w >= frame_width * 0.95 and h >= frame_height * 0.95:
                    print(f"[FRAME {frame_count}] Full-frame box — likely false positive. Skipping.")
                    emotions_data.append({
                    "frame": frame_count,
                    "face_detected": False,
                    "emotion": None,
                    "confidence": None,
                    "x": None,
                    "y": None,
                    "width": None,
                    "height": None
                    })
                    continue

                emotion = face['dominant_emotion']
                score = face['emotion'][emotion]

                color = EMOTION_COLORS.get(emotion, (255, 255, 255))  # default: white
                # cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                # cv2.putText(frame, f"{emotion} {score:.2f}", (x, y - 10),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                # Draw black outline rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 4)  # black thick border
                # Draw colored inner rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                # Draw black text outline (thicker and under)
                cv2.putText(frame, f"{emotion} {score:.2f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
                # Draw colored text on top
                cv2.putText(frame, f"{emotion} {score:.2f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                emotions_data.append({
                    "frame": frame_count,
                    "face_detected": True,
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                    "emotion": emotion,
                    "confidence": float(score)
                })

                print(f"[FRAME {frame_count}] Emotion: {emotion}, Score: {score:.2f}")
        except Exception as e:
            print(f"[WARN] Frame {frame_count} skipped: {e}")

        out.write(frame)

    cap.release()
    out.release()
    print(f"[INFO] Processed video saved to: {output_path}")

    # Optional: clean up input file
    # os.remove(input_path)

    #  Upload to S3
    s3_key = f"videos/{os.path.basename(output_path)}"
    try:
        s3.upload_file(output_path, bucket_name, s3_key)
        s3_url = f"https://{bucket_name}.s3.amazonaws.com/{s3_key}"
        print(f"Uploaded {output_path} to S3 bucket: {s3_url}")
    except Exception as e:
        print(f"Error uploading to S3: {e}")
        return jsonify({'error': 'Failed to upload to S3'}), 500

    return jsonify({
        "message": "Video processed successfully.",
        "output_video_path": output_path,
        "s3_url": s3_url,
        "results": emotions_data,
    })

if __name__ == '__main__':
    app.run(debug=True)

