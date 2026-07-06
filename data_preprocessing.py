import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

DATASET_PATH = "thermal_data"

# -----------------------------
# Data Generator
# -----------------------------
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(224,224),
    batch_size=8,
    class_mode="binary",
    subset="training",
    classes=["Healthy", "Faulty"],  #Healthy=0, Faulty=1
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(224,224),
    batch_size=8,
    class_mode="binary",
    subset="validation",
    classes=["Healthy", "Faulty"], 
    shuffle=False
)

print("Class Mapping:", train_gen.class_indices)

# -----------------------------
# Visualize Augmented Images
# -----------------------------
images, labels = next(train_gen)

plt.figure(figsize=(12,6))

for i in range(8):
    plt.subplot(2,4,i+1)
    plt.imshow(images[i])
    plt.title(f"Label: {int(labels[i])}")
    plt.axis("off")

plt.suptitle("Week 1 - Visual Review of Augmented Images")
plt.show()
