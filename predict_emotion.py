import librosa
import numpy as np
import joblib
import sys

# Load model and label encoder
model = joblib.load("emotion_recognition_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T, axis=0)
        return mfccs_processed
    except Exception as e:
        print("❌ Error extracting features:", e)
        return None

def predict_emotion(file_path):
    features = extract_features(file_path)
    if features is None:
        return "Could not extract features"

    features = features.reshape(1, -1)  # Reshape for model input
    prediction = model.predict(features)
    predicted_label = label_encoder.inverse_transform(prediction)
    return predicted_label[0]

# Example usage:
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❗ Usage: python predict_emotion.py path_to_audio.wav")
    else:
        file_path = sys.argv[1]
        emotion = predict_emotion(file_path)
        print(f"🎤 Predicted Emotion: {emotion}")
