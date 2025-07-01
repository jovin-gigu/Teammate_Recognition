# 🧠 Teammate Recognition System (Face + Voice)

A multimodal biometric recognition system that identifies team members using both **face** and **voice**. Built with Python using `face_recognition`, OpenCV, and voice embedding models like `resemblyzer`.

---

## 🚀 Features

- 👤 Face recognition from webcam input
- 🎙️ Voice recognition from recorded samples or microphone
- 📦 Stores face and voice features in `.pkl` files
- 🔐 Dual-modal verification for more secure recognition
- 🧩 Modular structure (train, recognize, fuse)

---

## 📁 Project Structure
```bash
TeammateRecognition/
├── face_recognition/
│   ├── known_faces/
│   ├── train_face.py
│   └── recognize_face.py
├── voice_recognition/
│   ├── voice_dataset/
│   ├── train_voice.py
│   └── recognize_voice.py
├── models/
│   ├── face_encodings.pkl
│   └── voice_embeddings.pkl
├── main.py
└── README.md
```


---

## 📦 Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

## Run the files
Train models:
```bash
python face_recognition/train_face.py
python voice_recognition/train_voice.py
```
Run recognizer:
```bash
python main.py
```

