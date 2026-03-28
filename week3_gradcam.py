import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model

# ----------------------------
# SETTINGS
# ----------------------------
DATASET_PATH = "thermal_data"
IMG_SIZE = 224

# ----------------------------
# Automatically find an image
# ----------------------------
img_path = None

for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        if file.endswith(".bmp") or file.endswith(".bmp"):
            img_path = os.path.join(root, file)
            break
    if img_path:
        break

if img_path is None:
    raise ValueError("No image found inside dataset")

print("Using image:", img_path)

# ----------------------------
# Load Image
# ----------------------------
img = cv2.imread(img_path)

if img is None:
    raise ValueError("Image could not be loaded")

img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img_normalized = img / 255.0
img_array = np.expand_dims(img_normalized, axis=0)

# ----------------------------
# Load MobileNetV2
# ----------------------------
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

# Last convolution layer
last_conv_layer = base_model.get_layer("Conv_1")

# Create GradCAM model
grad_model = Model(
    inputs=base_model.inputs,
    outputs=[last_conv_layer.output, base_model.output]
)

# ----------------------------
# Compute Gradients
# ----------------------------
with tf.GradientTape() as tape:

    conv_outputs, predictions = grad_model(img_array)

    loss = tf.reduce_mean(predictions)

grads = tape.gradient(loss, conv_outputs)

pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

conv_outputs = conv_outputs[0]

heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
heatmap = tf.squeeze(heatmap)

heatmap = np.maximum(heatmap, 0)

if np.max(heatmap) != 0:
    heatmap /= np.max(heatmap)

heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))

# ----------------------------
# Convert Heatmap
# ----------------------------
heatmap = np.uint8(255 * heatmap)

heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

overlay = heatmap_color * 0.4 + img

# ----------------------------
# Plot Results
# ----------------------------
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Grad-CAM Heatmap")
plt.imshow(heatmap_color)
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Overlay")
plt.imshow(cv2.cvtColor(overlay.astype("uint8"), cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.tight_layout()
plt.show()