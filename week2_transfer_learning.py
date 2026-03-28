import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, roc_curve, auc

import seaborn as sns
import pickle

# -----------------------------
# Dataset Path
# -----------------------------
DATASET_PATH = "thermal_data"
IMG_SIZE = 224
BATCH_SIZE = 16

# -----------------------------
# Data Generator
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

val_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

print("\nClass Mapping:", train_gen.class_indices)

# -----------------------------
# Load MobileNetV2
# -----------------------------
print("\nLoading MobileNetV2 Feature Extractor...")

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    pooling="avg"
)

base_model.trainable = False

# -----------------------------
# Feature Extraction
# -----------------------------
print("\nExtracting training features...")

train_features = base_model.predict(train_gen, verbose=1)
train_labels = train_gen.classes

print("\nExtracting validation features...")

val_features = base_model.predict(val_gen, verbose=1)
val_labels = val_gen.classes

print("\nFeature Shape:", train_features.shape)

# -----------------------------
# Random Forest Training
# -----------------------------
print("\nTraining Random Forest Classifier...")

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)

rf.fit(train_features, train_labels)

# -----------------------------
# Predictions
# -----------------------------
val_preds = rf.predict(val_features)
val_probs = rf.predict_proba(val_features)[:,1]

accuracy = accuracy_score(val_labels, val_preds)

print("\nHybrid Model Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(val_labels, val_preds))

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(val_labels, val_preds)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.show()

# -----------------------------
# ROC Curve
# -----------------------------
fpr, tpr, _ = roc_curve(val_labels, val_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))

plt.plot(fpr, tpr, label="AUC = {:.2f}".format(roc_auc))
plt.plot([0,1],[0,1],'--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")
plt.legend()

plt.show()

# -----------------------------
# Save Random Forest Model
# -----------------------------
print("\nSaving Random Forest model...")

with open("random_forest_model.pkl", "wb") as f:
    pickle.dump(rf, f)

print("\nModel saved successfully!")


import cv2

print("\nStarting inference demo...")

TEST_FOLDER = "thermal_data/Healthy"   # or Faulty for testing

for img_name in os.listdir(TEST_FOLDER):

    img_path = os.path.join(TEST_FOLDER, img_name)

    img = cv2.imread(img_path)
    img = cv2.resize(img, (224,224))
    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    features = base_model.predict(img)

    prediction = rf.predict(features)[0]

    label = "Faulty" if prediction == 1 else "Healthy"

    print(img_name, "->", label)





    import os
import cv2
import time 
import numpy as np

print("\n==============================")
print("WEEK 4: INFERENCE ENGINE DEMO")
print("==============================")

TEST_FOLDER = "thermal_data"
IMG_SIZE = 224

print("\nStarting simulated live inspection...\n")

# Walk through all folders and images
for root, dirs, files in os.walk(TEST_FOLDER):

    for file in files:

        if file.lower().endswith((".jpg",".jpeg",".png",".bmp")):

            img_path = os.path.join(root, file)

            frame = cv2.imread(img_path)

            if frame is None:
                continue

            start_time = time.time()

            # Preprocess
            img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)

            # Feature Extraction (MobileNet)
            features = base_model.predict(img, verbose=0)

            # Random Forest Prediction
            prediction = rf.predict(features)[0]

            label = "Faulty" if prediction == 1 else "Healthy"

            latency = time.time() - start_time

            print(f"{file} → {label} | Latency: {latency:.4f} sec")

            # Display result
            display = frame.copy()

            color = (0,255,0) if label=="Healthy" else (0,0,255)

            cv2.putText(display,
                        label,
                        (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        color,
                        2)

            cv2.imshow("Thermal Fault Detection System", display)

            if cv2.waitKey(400) & 0xFF == ord('q'):
                break

cv2.destroyAllWindows()

print("\nLive inspection completed.")