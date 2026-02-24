# 🎭 Emotion-Based System Using Facial Emotion Detection
<p align="center"> <b>AI-Powered Facial Emotion Recognition Web Application</b> <br><br> 🎓 Final Year Project — Bachelor of Computer Engineering (Computer Systems) <br> 👩‍💻 Developed by <b>Nur Munawwarah</b> </p>
<p align="center">










</p>

## 📌 Project Overview

This project is an AI-driven facial emotion recognition system that detects and classifies human emotions from facial expressions using deep learning techniques.

The system integrates:

🔍 BlazeFace for lightweight and fast face detection

🧠 Convolutional Neural Network (CNN) for emotion classification

🌐 Flask Web Framework for backend deployment

💻 Clean frontend interface using HTML, CSS, and JavaScript

The application demonstrates a complete end-to-end AI pipeline, from image preprocessing to real-time emotion prediction in a web-based environment.

## 🚀 Key Features

✅ Real-time facial emotion detection

✅ Lightweight face detection using TensorFlow Lite (BlazeFace)

✅ Deep learning-based emotion classification (.h5 model)

✅ Web-based deployment using Flask

✅ Modular and scalable backend architecture

✅ Clean preprocessing pipeline (crop → grayscale → resize → normalize)


## 🏗️ System Architecture

<img width="634" height="647" alt="Screenshot 2026-01-30 191047" src="https://github.com/user-attachments/assets/59c38e97-4946-4d5d-bb12-6d96cdf2b76f" />

## 🧠 AI Model Details
🔍 Face Detection

- Model: BlazeFace (Short Range)
- Format: .tflite
- Purpose: Fast and lightweight facial region detection

## 🎯 Emotion Classification

- Model Format: .h5
- Framework: TensorFlow / Keras
- Architecture: Convolutional Neural Network (CNN)
- Output: Probability distribution across emotion classes

## 😊 Supported Emotion Classes

- Happy
- Sad
- Angry
- Fear
- Surprise
- Disgust
- Neutral

 ## 🛠️ Tech Stack
| Layer | Technology |
|-------|------------|
| Programming | Python 3.13 |
| Backend | Flask |
| AI Framework | TensorFlow / Keras |
| Computer Vision | OpenCV |
| Frontend | HTML, CSS, JavaScript |
| Model Formats | .h5, .tflite |

## 📂 Project Structure

```bash
emotion-based-system-using-facial-emotion-detection/
│
├── backend/                     # Main application folder
│   ├── static/                  # CSS & JavaScript files
│   │   ├── script.js
│   │   └── style.css
│   │
│   ├── templates/               # HTML templates
│   │   └── index.html
│   │
│   ├── app.py                   # Flask backend application
│   ├── emotion_model.h5         # Trained CNN emotion model
│   ├── blaze_face_short_range.tflite  # Face detection model
│   └── requirements.txt         # Project dependencies
│
├── train_dataset/               # Training dataset
├── README.md                    # Project documentation
└── LICENSE
```
## ⚙️ Installation & Setup

1️⃣ Clone the Repository

git clone https://github.com/nrmnwrh21/emotion-based-system-using-facial-emotion-detection.git

cd emotion-based-system-using-facial-emotion-detection/backend

2️⃣ Create Virtual Environment

python -m venv venv
venv\Scripts\activate

3️⃣ Install Dependencies

pip install -r requirements.txt

4️⃣ Run the Application

python app.py

Access via browser:

http://127.0.0.1:5000/

## 📊 Model Pipeline

Image Preprocessing Steps:

<img width="828" height="476" alt="Screenshot 2026-01-25 032615" src="https://github.com/user-attachments/assets/cd0af56e-b725-4570-9649-162850a70f13" />

Sample preprocessing outputs are included in the repository:

- Cropped image
- Grayscale image
- Blurred/processed image

## 📈 Performance & Improvements

Current Strengths

- Lightweight face detection model
- Clean modular backend design
- Real-time web deployment
- Easy to scale and integrate

Future Enhancements

- Deploy to cloud (Render / AWS / Railway)
- Convert model fully to TensorFlow Lite
- Add emotion-based recommendation system
- Implement emotion history tracking
- Improve dataset diversity for higher accuracy

## 🎓 Academic & Engineering Contribution

This project demonstrates:

- End-to-end AI system development
- Deep learning model integration into web applications
- Computer vision implementation using OpenCV
- Backend-frontend system integration
- Real-world deployment architecture

It reflects practical skills in:

- AI model handling
- Model deployment
- Backend API integration
- Software system design

## System Interface
<img width="620" height="400" alt="Screenshot 2026-01-30 212021" src="https://github.com/user-attachments/assets/0d95661f-9101-4cdd-925a-6f17d16556ee" />
