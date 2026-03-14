# 🦴 AASHVAS – AI Knee Disease Detection

![AI](https://img.shields.io/badge/AI-Machine%20Learning-blue)
![Python](https://img.shields.io/badge/Python-3.9-green)
![Healthcare](https://img.shields.io/badge/Domain-Healthcare-red)

AI-powered knee joint risk detection system using **acoustic signal analysis and machine learning**.

The system records knee joint sound, processes it using **Digital Signal Processing (DSP)** techniques, and predicts the risk of joint disorders such as **osteoarthritis**.

---

# 🚀 Project Overview

Joint disorders like **osteoarthritis** are often diagnosed only after significant damage has occurred. Traditional diagnosis relies on **X-rays or MRI scans**, which can be expensive and unsuitable for frequent monitoring.

**AASHVAS** introduces an **AI-based acoustic screening approach** that analyzes knee joint sounds and predicts risk levels using machine learning.

The system integrates:

- 🤖 AI (Machine Learning)
- 🎛 DSP (Digital Signal Processing)
- 📱 Mobile/Web Interface

---

# 🧠 Key Features

- Record or upload knee joint audio
- Noise reduction and preprocessing
- Log-Mel Spectrogram generation
- AI-based risk classification
- Risk level with confidence score
- Visual spectrogram analysis
- Lightweight and scalable system

---

# 🔬 How It Works

The system follows this pipeline:
Audio Input
↓
Noise Filtering
↓
FFT (Frequency Analysis)
↓
Log-Mel Spectrogram
↓
AI Model (Random Forest / CNN)
↓
Risk Prediction


---

# ⚙️ Technology Stack

## AI & Machine Learning
- Python
- Scikit-learn
- XGBoost
- TensorFlow / TensorFlow Lite

## Signal Processing
- Librosa
- FFT (Fast Fourier Transform)
- Log-Mel Spectrogram
- Bandpass Filtering

## Application Layer
- HTML
- CSS
- Android Studio (future integration)

---

