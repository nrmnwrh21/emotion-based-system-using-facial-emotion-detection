"""
Emotion Recognition Review System
- Facial emotion detection
- Rating system
- MongoDB GridFS image storage
- Image viewing endpoint
- Preprocessing proof image saving (for presentation)
"""

import cv2
import uuid
import numpy as np
from datetime import datetime

from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import tensorflow as tf

from pymongo import MongoClient
import gridfs
from bson.objectid import ObjectId

# ======================================================
# APP INIT
# ======================================================
app = Flask(__name__)
CORS(app)

# ======================================================
# MONGODB (ADMIN AUTH)
# ======================================================
client = MongoClient(
    "mongodb://admin:admin123@localhost:27017/?authSource=admin"
)

db = client["emotion_rating_db"]
collection = db["ratings"]
fs = gridfs.GridFS(db)

# ======================================================
# GLOBAL STATE
# ======================================================
detecting = False
latest_result = None
latest_face_frame = None
ratings = []

current_user = {
    "name": None,
    "email": None,
    "consent": None
}

# ======================================================
# EMOTION MODEL
# ======================================================
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

EMOTION_TO_STARS = {
    'Angry': 1,
    'Disgust': 1,
    'Fear': 2,
    'Sad': 2,
    'Neutral': 3,
    'Surprise': 4,
    'Happy': 5
}

emotion_model = tf.keras.models.load_model("emotion_model.h5")

# ======================================================
# MEDIAPIPE FACE DETECTOR
# ======================================================
base_options = python.BaseOptions(
    model_asset_path="blaze_face_short_range.tflite"
)

options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

# ======================================================
# CAMERA
# ======================================================
cap = cv2.VideoCapture(0)

# ======================================================
# HELPERS
# ======================================================
def blur_face(img):
    return cv2.GaussianBlur(img, (99, 99), 30)

# SAVE PREPROCESSING PROOF IMAGES (FOR SLIDE 8)
def save_proof_images(face):
    # Cropped face
    cv2.imwrite("proof_cropped.jpg", face)

    # Grayscale + resize
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    gray_resized = cv2.resize(gray, (48, 48))
    cv2.imwrite("proof_grayscale.jpg", gray_resized)

    # Blurred face (privacy)
    blurred = blur_face(face)
    cv2.imwrite("proof_blurred.jpg", blurred)

# ======================================================
# VIDEO STREAM
# ======================================================
def generate_frames():
    global detecting, latest_result, latest_face_frame, ratings

    while True:
        success, frame = cap.read()
        if not success:
            break

        if detecting:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            result = detector.detect(mp_image)

            if result.detections:
                bbox = result.detections[0].bounding_box
                x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height

                face = frame[y:y+h, x:x+w]
                if face.size > 0:
                    latest_face_frame = face.copy()

                    # ===== SAVE PREPROCESSING PROOF IMAGES (RUN ONCE) =====
                    save_proof_images(face)

                    # Preprocessing for CNN
                    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    gray = cv2.resize(gray, (48, 48))
                    gray = gray / 255.0
                    gray = np.reshape(gray, (1, 48, 48, 1))

                    preds = emotion_model.predict(gray, verbose=0)
                    emotion = EMOTIONS[np.argmax(preds)]
                    stars = EMOTION_TO_STARS[emotion]

                    ratings.append(stars)
                    avg = round(sum(ratings) / len(ratings), 2)

                    latest_result = {
                        "emotion": emotion,
                        "stars": stars,
                        "average": avg
                    }

                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(
                        frame, emotion, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0), 2
                    )

        _, buffer = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            buffer.tobytes() +
            b"\r\n"
        )

# ======================================================
# ROUTES
# ======================================================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video")
def video():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/register", methods=["POST"])
def register():
    data = request.json

    if not data.get("consent"):
        return jsonify({"error": "Consent required"}), 403

    current_user["name"] = data.get("name")
    current_user["email"] = data.get("email")
    current_user["consent"] = {
        "given": True,
        "timestamp": datetime.now(),
        "purpose": "Emotion recognition research"
    }

    return jsonify({"status": "registered"})

@app.route("/start", methods=["POST"])
def start_detection():
    global detecting, ratings
    ratings = []
    detecting = True
    return jsonify({"status": "started"})

@app.route("/stop", methods=["POST"])
def stop_detection():
    global detecting
    detecting = False
    return jsonify(latest_result)

@app.route("/rating")
def rating():
    if not latest_result:
        return jsonify({"emotion": "Neutral", "stars": "☆☆☆☆☆", "average": 0})

    stars = latest_result["stars"]
    return jsonify({
        "emotion": latest_result["emotion"],
        "stars": "★" * stars + "☆" * (5 - stars),
        "average": latest_result["average"]
    })

@app.route("/submit", methods=["POST"])
def submit():
    if latest_result is None or latest_face_frame is None:
        return jsonify({"error": "No data"}), 400

    blurred = blur_face(latest_face_frame)
    success, buffer = cv2.imencode(".jpg", blurred)

    if not success:
        return jsonify({"error": "Encoding failed"}), 500

    image_id = fs.put(
        buffer.tobytes(),
        filename=f"{uuid.uuid4()}.jpg",
        contentType="image/jpeg"
    )

    collection.insert_one({
        "name": current_user["name"],
        "email": current_user["email"],
        "emotion": latest_result["emotion"],
        "stars": latest_result["stars"],
        "average": latest_result["average"],
        "image_id": image_id,
        "consent": current_user["consent"],
        "timestamp": datetime.now()
    })

    return jsonify({
        "status": "submitted",
        "image_id": str(image_id)
    })

# ======================================================
# IMAGE VIEW
# ======================================================
@app.route("/image/<image_id>")
def view_image(image_id):
    try:
        file = fs.get(ObjectId(image_id))
        return Response(file.read(), mimetype=file.content_type)
    except Exception:
        return jsonify({"error": "Image not found"}), 404

# ======================================================
# RUN
# ======================================================
if __name__ == "__main__":
    app.run(debug=True, threaded=True)
