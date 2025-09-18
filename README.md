# Real-Time-Facial-Emotion-Detection-CNN-OpenCV-TensorFlow-
A deep learning-based real-time facial emotion detection system using CNN, TensorFlow/Keras, and OpenCV. The model classifies 7 emotions (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise) from live webcam input and displays results instantly. Applications include HCI, mental health, education, and security.

---

# 😊 Real-Time Facial Emotion Detection (CNN + OpenCV + TensorFlow)

## 📌 Overview

This project is a **real-time facial emotion detection system** built using **Convolutional Neural Networks (CNNs)** with **TensorFlow/Keras** and **OpenCV**.
The model recognizes **seven emotions** — *Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise* — from facial expressions captured via a webcam.

It demonstrates how deep learning can be applied in **computer vision** and **affective computing**, with potential applications in:

* Human-Computer Interaction (HCI)
* Mental health monitoring
* Online education
* Customer experience analytics
* Security and surveillance

---

## ✨ Key Features

* 🎥 **Real-Time Detection** – Uses webcam input for live face detection.
* 🧠 **Deep Learning Model** – CNN trained on FER-2013 dataset for robust emotion classification.
* ⚡ **Efficient Integration** – Combines TensorFlow/Keras with OpenCV for real-time performance.
* 🎭 **Multi-Emotion Classification** – Detects 7 emotion categories.
* 🖥️ **Interactive Output** – Displays predicted emotion label on video feed.

---

## 📂 Project Structure

```
Facial-Emotion-Detection/
├── facialemotionmodel.h5     # Pre-trained CNN model
├── requirements.txt          # Dependencies
├── dataset/                  # (Optional) Training dataset
├── src/                      
│   ├── train_model.py        # Script for training CNN
│   ├── realtime_detection.py # Real-time webcam-based detection
│   └── utils.py              # Helper functions
└── README.md
```

---

## ⚙️ Installation

### 🔧 Prerequisites

* Python **3.7+**
* Webcam-enabled device

### 📥 Setup Instructions

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/Facial-Emotion-Detection.git
   cd Facial-Emotion-Detection
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Running the Application

### ▶️ Real-Time Emotion Detection

```bash
python src/realtime_detection.py
```

This will start the webcam feed and display real-time emotion predictions.

### 🏋️‍♂️ Training the Model (Optional)

If you want to retrain the model:

```bash
python src/train_model.py
```

---

## 🛠️ Tech Stack

* **Programming Language:** Python
* **Deep Learning:** TensorFlow, Keras
* **Computer Vision:** OpenCV
* **Data Handling:** Pandas, NumPy
* **Utilities:** Scikit-learn, tqdm

---

## 📊 Dataset

The model is trained on the **FER-2013** dataset, which contains over **35,000 labeled grayscale facial images** across 7 emotion categories.
📌 Dataset link: [FER-2013 Kaggle Dataset](https://www.kaggle.com/datasets/msambare/fer2013)

---

## 📌 Applications

* Human-Computer Interaction (HCI)
* Mental health and therapy monitoring
* Smart surveillance and security
* Personalized e-learning systems
* Customer behavior and feedback analysis

---

## 📜 License

This project is licensed under the **MIT License**.

---
