# train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

data = pd.read_csv('gesture_data.csv')
X = data.iloc[:, 1:]
y = data.iloc[:, 0]

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, 'gesture_model.pkl')
print("Model berhasil disimpan.")
