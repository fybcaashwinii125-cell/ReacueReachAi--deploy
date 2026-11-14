# RescueReach AI 🚨

RescueReach AI is a project I'm building to explore how AI can help identify emergency situations through sound detection.

The idea is simple: recognize sounds that may indicate an emergency and generate an alert based on the model's prediction.

## About the Project

This project currently focuses on detecting emergency-related sounds using a machine learning model.

It takes an audio recording, processes it, and predicts whether the sound belongs to an emergency or normal category.

I'm working on making the system useful as an early alert tool, with the aim of helping people become aware of possible emergencies sooner.

## Features

- Emergency sound detection using machine learning
- Audio recording through the web interface
- Audio processing and feature extraction
- Emergency and normal sound classification
- Prediction confidence displayed by the system
- Flask-based web application

## Technologies Used

- Python
- Flask
- HTML
- CSS
- JavaScript
- Scikit-learn
- Librosa
- NumPy
- ESC-50 Dataset

## Project Structure

```text
RescueReachAI/
│
├── app.py
├── model.pkl
├── scaler.pkl
├── threshold.pkl
├── requirements.txt
│
├── templates/
│   └── index.html
│
├── static/
│   ├── style.css
│   └── script.js
│
└── README.md
```

## Dataset

The project uses the ESC-50 environmental sound dataset for training and testing the sound classification model.

The dataset is not included in this repository.

## Current Status

The sound detection and classification system has been developed, and I'm working on deployment and further improvements.

This is an ongoing project, and I plan to explore more ways to make emergency detection and alert systems useful in real-world situations.

## Note

RescueReach AI is currently a prototype. Its predictions may not always be accurate and should not be treated as a substitute for emergency services.

---

**Built as a learning project with the aim of using AI for something meaningful.**
