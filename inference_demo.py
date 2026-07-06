import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.ensemble import RandomForestClassifier

print("\n==============================")
print("WEEK 4: INFERENCE ENGINE DEMO")
print("==============================")

# -----------------------------
# Configuration
# -----------------------------
DATASET_PATH = "thermal_data"
IMG_SIZE = 224
BATCH_SIZE = 16

# -----------------------------
# Load Dataset
# -----------------------------
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    shuffle=False
)

# -----------------------------
# Load MobileNetV2
# -----------------------------
print("\nLoading MobileNetV2...")

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    pooling="avg"
)

# -----------------------------
# Extract Features
# -----------------------------
print("Extracting features...")

train_features = base_model.predict(train_gen, verbose=0)
train_labels = train_gen.classes

# -----------------------------
# Train Random Forest
# -----------------------------
print("Training Random Forest...")

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(train_features, train_labels)

print("\nModel ready for inference.")

# -----------------------------
# Inference Engine
# -----------------------------
print("\nStarting simulated live inspection...\n")

for root, dirs, files in os.walk(DATASET_PATH):

    for file in files:

        if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):

            img_path = os.path.join(root, file)

            frame = cv2.imread(img_path)

            if frame is None:
                continue

            start = time.time()

            # Preprocess image
            img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)

            # Feature extraction
            features = base_model.predict(img, verbose=0)

            # Prediction
            prediction = rf.predict(features)[0]

            label = "Faulty" if prediction == 1 else "Healthy"

            latency = time.time() - start

            print(f"{file} → {label} | Latency: {latency:.4f} sec")

            # -----------------------------
            # Display Result (Matplotlib)
            # -----------------------------
            display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            plt.figure(figsize=(4,4))
            plt.imshow(display)
            plt.title(f"{label}")
            plt.axis("off")
            plt.show(block=False)
            plt.pause(0.5)
            plt.close()

print("\nLive inspection completed.")
