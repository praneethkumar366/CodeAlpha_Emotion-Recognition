import tkinter as tk
from tkinter import messagebox
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import os
import joblib
import librosa
import time
from datetime import datetime
import csv
import warnings
# Constants
SAMPLE_RATE = 22050
DURATION = 3  # seconds
AUDIO_FILE = "audio_samples/recorded.wav"

# Load model and encoder
model = joblib.load("models/emotion_recognition_model.pkl")
le = joblib.load("models/label_encoder.pkl")

# Ensure audio_samples directory exists
os.makedirs("audio_samples", exist_ok=True)

def extract_features(file_path):
    X, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
    mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean.reshape(1, -1)

def save_to_history(emotion):
    with open("prediction_history.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), emotion])

def record_audio():
    try:
        status_label.config(text="Recording...", fg="blue")
        window.update()
        time.sleep(0.5)

        recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()
        write(AUDIO_FILE, SAMPLE_RATE, recording)
        
        status_label.config(text="Analyzing...", fg="orange")
        window.update()
        
        features = extract_features(AUDIO_FILE)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prediction = model.predict(features)
        
        predicted_label = le.inverse_transform(prediction)[0]

        result_label.config(text=f"🎤 Emotion: {predicted_label}", fg="#0F9D58")
        status_label.config(text="Prediction complete.", fg="black")
        save_to_history(predicted_label)

    except Exception as e:
        messagebox.showerror("Error", str(e))
        status_label.config(text="Error occurred", fg="red")

# --- Tkinter GUI Setup ---
window = tk.Tk()
window.title("🎙️ Emotion Recognition from Speech")
window.geometry("500x350")
window.resizable(False, False)
window.configure(bg="#E8F0FE")

title_label = tk.Label(window, text="Emotion Recognition", font=("Helvetica", 20, "bold"), bg="#E8F0FE", fg="#202124")
title_label.pack(pady=20)

record_button = tk.Button(window, text="🎙️ Record & Predict", font=("Helvetica", 14, "bold"), bg="#1A73E8", fg="white",
                        padx=20, pady=10, relief="flat", activebackground="#174EA6", command=record_audio)
record_button.pack(pady=20)

result_label = tk.Label(window, text="", font=("Helvetica", 16), bg="#E8F0FE")
result_label.pack(pady=10)

status_label = tk.Label(window, text="", font=("Helvetica", 10), bg="#E8F0FE")
status_label.pack(pady=5)

footer_label = tk.Label(window, text="Created by Praneet • 2025", font=("Helvetica", 9), bg="#E8F0FE", fg="#5F6368")
footer_label.pack(side="bottom", pady=10)

window.mainloop()
