import sounddevice as sd
from scipy.io.wavfile import write
import uuid
import joblib
import os
from features_extraction import extract_features

fs = 44100
seconds = 3
filename = f"audio_samples/{uuid.uuid4()}.wav"
os.makedirs("audio_samples", exist_ok=True)

print("🎙️ Recording...")
recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()
write(filename, fs, recording)

model = joblib.load("models/emotion_recognition_model.pkl")
le = joblib.load("models/label_encoder.pkl")
features = extract_features(filename).reshape(1, -1)

prediction = model.predict(features)
emotion = le.inverse_transform(prediction)[0]
print(f"🎤 Predicted Emotion: {emotion}")
