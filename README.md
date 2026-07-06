VisionSpec-QC

Automated Visual Quality Control for SSDs using Thermal Imaging & Transfer Learning

VisionSpec-QC is a computer vision system that automatically classifies SSDs as Healthy or Faulty from thermal images, removing the need for slow, inconsistent manual inspection.

The system uses a hybrid two-stage architecture rather than a single end-to-end deep learning model. A MobileNetV2 network, pretrained on ImageNet, is used as a frozen feature extractor — its convolutional layers are never fine-tuned, and instead convert each thermal image into a compact deep feature vector via global average pooling. These feature vectors are then passed to a Random Forest classifier, which learns the actual decision boundary between healthy and faulty components. This design choice was deliberate: training a full CNN end-to-end on a small thermal-image dataset caused severe overfitting, so freezing the pretrained backbone and pairing it with a classical ML classifier produced a far more robust and generalizable model.

To make the system trustworthy in a real quality-control setting, Grad-CAM visual explainability was added, highlighting the exact regions of a thermal image that drove each prediction — giving inspectors visual justification rather than a black-box output. The system also includes a simulated real-time inference mode, running new images through the full pipeline with OpenCV and measuring per-image latency to validate feasibility for live deployment.

The model was evaluated using accuracy, a confusion matrix, and ROC-AUC analysis, achieving 98% classification accuracy. Overall, VisionSpec-QC demonstrates a complete, deployment-aware computer vision workflow: augmented data preprocessing, transfer-learning-based feature extraction, classical ML classification, explainable AI, and real-time inference simulation — built specifically for industrial-style visual quality control.
