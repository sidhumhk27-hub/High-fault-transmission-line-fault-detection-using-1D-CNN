# ==========================================================
# ABB Power Systems - Fault Classification Using 1D CNN
# ==========================================================

# -----------------------------
# 0. Import Libraries
# -----------------------------
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Flatten

# -----------------------------
# 1. Upload Dataset
# -----------------------------
from google.colab import files
import io

print("📌 Upload your dataset CSV file")
uploaded = files.upload()

for filename in uploaded.keys():
    print(f"Uploaded file: {filename}")
    DATA_PATH = filename

df = pd.read_csv(io.BytesIO(uploaded[filename]), parse_dates=["timestamp"])

print("Data loaded successfully ✔")
print(df.head())
print("\nFault distribution:")
print(df["fault_type"].value_counts())

# -----------------------------
# 2. Preprocessing
# -----------------------------
df = df.sort_values(["line_id", "timestamp"]).reset_index(drop=True)

# Label Encoding
label_encoder = LabelEncoder()
df["fault_label"] = label_encoder.fit_transform(df["fault_type"])
num_classes = len(label_encoder.classes_)
print("\nDetected Classes:", label_encoder.classes_)

# Feature Columns
feature_cols = [
    "voltage_phase_a", "voltage_phase_b", "voltage_phase_c",
    "current_phase_a", "current_phase_b", "current_phase_c"
]

# -----------------------------
# 3. Create Time-Series Sequences
# -----------------------------
SEQ_LEN = 20  # Number of time steps per sample

def create_sequences(group):
    features = group[feature_cols].values
    labels = group["fault_label"].values
    X_seq = []
    y_seq = []
    for i in range(len(group) - SEQ_LEN + 1):
        X_seq.append(features[i:i+SEQ_LEN])
        y_seq.append(labels[i + SEQ_LEN - 1])
    return np.array(X_seq), np.array(y_seq)

X_list = []
y_list = []

for line_id, group in df.groupby("line_id"):
    X_seq, y_seq = create_sequences(group)
    X_list.append(X_seq)
    y_list.append(y_seq)

X = np.vstack(X_list)
y = np.hstack(y_list)
print("\nSequence data shape:", X.shape, y.shape)

# -----------------------------
# 4. Scaling + Train-Test Split
# -----------------------------
n_samples, seq_len, n_features = X.shape
scaler = StandardScaler()

X_flat = X.reshape(-1, n_features)
X_scaled_flat = scaler.fit_transform(X_flat)
X_scaled = X_scaled_flat.reshape(n_samples, seq_len, n_features)

y_cat = to_categorical(y, num_classes=num_classes)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_cat, test_size=0.2, stratify=y, random_state=42
)

print("Train:", X_train.shape)
print("Test:", X_test.shape)

# -----------------------------
# 5. Build 1D CNN Model
# -----------------------------
model = Sequential([
    Conv1D(64, 3, activation="relu", input_shape=(SEQ_LEN, n_features)),
    Conv1D(64, 3, activation="relu"),
    MaxPooling1D(2),

    Conv1D(128, 3, activation="relu"),
    MaxPooling1D(2),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(num_classes, activation="softmax")
])

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

# -----------------------------
# 6. Train Model
# -----------------------------
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)

# -----------------------------
# 7. Evaluation
# -----------------------------
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n🎯 Test Accuracy: {test_acc:.4f}")

y_test_label = np.argmax(y_test, axis=1)
y_pred = np.argmax(model.predict(X_test), axis=1)

print("\n🔍 Classification Report:")
print(classification_report(y_test_label, y_pred, target_names=label_encoder.classes_))

print("\n📌 Confusion Matrix:")
print(confusion_matrix(y_test_label, y_pred))