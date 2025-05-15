import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load features from CSV
df = pd.read_csv('emotion_features.csv')

# Separate features and labels
X = df.drop('emotion', axis=1)
y = df['emotion']

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save model and label encoder
joblib.dump(model, 'emotion_recognition_model.pkl')
joblib.dump(le, 'label_encoder.pkl')
print("✅ Model and label encoder saved.")
