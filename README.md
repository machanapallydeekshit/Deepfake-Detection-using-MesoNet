# 🧠 Deepfake Detection using CNN

A certified research project focused on detecting deepfakes using Convolutional Neural Networks (CNNs). With the rise of AI-generated media, it's increasingly important to develop systems that identify fake images or videos to prevent misinformation and misuse.

---

## 🎯 Problem Statement

Deepfakes are synthetic media where a person’s face or voice is replaced with someone else's using deep learning. They pose threats to digital trust and authenticity. This project aims to develop an AI model that detects such fake content with high accuracy.

---

## 🧰 Tech Stack

- **Language**: Python
- **GUI**: Streamlit
- **Libraries**: TensorFlow, Keras, NumPy, OpenCV, Matplotlib, Seaborn
- **Model**: Custom MesoNet Model
- **Platform**: kaggle notebook
- **Version Control**: Git & GitHub

---

## 📂 Dataset

- **Name**: [Celeb DF V2](https://www.kaggle.com/datasets/reubensuju/celeb-df-v2)
- **Samples**: ~5000 real and fake video/image samples
- **Preprocessing**: Frame extraction → Face detection (MTCNN) → Resizing (128x128)

---

# Project Output Screenshots

## Uploading Video From Local Files
<img width="1920" height="1080" alt="Screenshot (69)" src="https://github.com/user-attachments/assets/78749c82-be99-4d97-9ddc-671484cbe516" />

## Post Analysis Message (if No faces Detected in the Video)
<img width="1920" height="1080" alt="Screenshot (73)" src="https://github.com/user-attachments/assets/42c19962-fc50-44f6-ac7b-d7aadab26565" />

## Post Analysis Message (When faces are detected in the Video)
<img width="1920" height="1080" alt="Screenshot (74)" src="https://github.com/user-attachments/assets/cf76126a-f1d9-41bb-b12e-6e3fd2819afb" />

## Frame Analysis Tab
<img width="1920" height="1080" alt="Screenshot (75)" src="https://github.com/user-attachments/assets/01a389ed-1c63-45a8-832b-98ede42f3ad6" />

## Most Activated Region 
<img width="1920" height="1080" alt="Screenshot (76)" src="https://github.com/user-attachments/assets/ce817661-b75a-4c9c-8cc3-17783f6a4b8e" />

## Summary Tab Anomaly Detection
<img width="1920" height="1080" alt="Screenshot (77)" src="https://github.com/user-attachments/assets/f4882509-707b-45b7-bd1f-fabf651354e0" />

## Heat Maps With Toggle option
<img width="1920" height="1080" alt="Screenshot (78)" src="https://github.com/user-attachments/assets/11377a8b-529c-409d-b134-de3b2afb9fa2" />
<img width="1920" height="1080" alt="Screenshot (80)" src="https://github.com/user-attachments/assets/9d91672a-568b-4c2f-abcd-1ea5e299e867" />
