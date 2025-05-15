# 🎤 Emotion Recognition from Speech  
> 💼 Machine Learning Internship Project @ CodeAlpha  

In today’s fast-paced digital world, where most communication happens virtually, understanding human emotion from speech is a major challenge. Whether it’s online classrooms, customer service calls, or virtual therapy — machines often fail to interpret emotions behind voice, which can lead to miscommunication and emotional disconnect.

This project addresses that modern problem by developing an intelligent system that detects emotions like **Happy**, **Fearful**, and **Disgust** using Machine Learning and Speech Processing. By combining the power of **MFCC features**, **Random Forest Classifier**, and a user-friendly **Tkinter GUI**, this tool allows users to record their voice and instantly receive emotion predictions.

---

### 🚀 Features  
- 🎙️ Real-time speech input recording  
- 📊 Emotion classification using trained ML model  
- 🧠 Uses MFCC feature extraction and Random Forest algorithm  
- 🖥️ Clean and attractive GUI built with Tkinter  
- ⚡ Fast, lightweight, and easy to use  
- 📁 Organised folder structure for modularity  

---

### 🛠 Tech Stack  
- Python 3.10+  
- Scikit-learn  
- Librosa  
- NumPy  
- SoundDevice  
- Tkinter  
- Joblib  

---

### 📁 Project Structure  
Emotion-Recognition-From-Speech/
├── models/
│ ├── emotion_recognition_model.pkl
│ └── label_encoder.pkl
├── audio_samples/
│ └── recorded.wav
├── gui.py
├── README.md
└── requirements.txt

yaml
Copy
Edit

---

### ⚙️ How to Run Locally  

1. 📥 Clone the repository  

git clone https://github.com/your-username/Emotion-Recognition-From-Speech.git
cd Emotion-Recognition-From-Speech

🐍 Create and activate conda environment
conda create -n emotion_env python=3.10  
conda activate emotion_env  

📦 Install all dependencies
pip install -r requirements.txt

▶️ Run the GUI
python gui.py


Record your voice and let the model predict how you feel!


🎓 Internship Acknowledgement
This project was developed as part of my Machine Learning Internship at CodeAlpha, where I explored practical applications of ML models in speech emotion recognition using real-time user input.

🤝 Contributions
If you’d like to improve this project (e.g., add more emotion classes, improve accuracy, or support longer audio), feel free to fork, raise an issue, or submit a pull request!

🔗 Connect with Me
💼 LinkedIn: linkedin.com/in/your-profile

📬 Email: your-email@example.com

