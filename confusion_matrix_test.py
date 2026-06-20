import os
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image as keras_image
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading MobileNetV2...")
vision_model = MobileNetV2(weights='imagenet')

CAR_KEYWORDS = ['car', 'convertible', 'jeep', 'limousine', 'minivan', 'pickup',
                'sports_car', 'beach_wagon', 'racer', 'station_wagon',
                'model_t', 'amphibian']

def is_car_match(label):
    label = label.lower()
    return any(
        kw == label or label.startswith(kw + '_') or label.endswith('_' + kw)
        for kw in CAR_KEYWORDS
    )

def check_is_car(image_path):
    img = Image.open(image_path).convert('RGB').resize((224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    preds = vision_model.predict(img_array, verbose=0)
    decoded = decode_predictions(preds, top=5)[0]

    is_car = any(
        is_car_match(label) and confidence > 0.15
        for (_, label, confidence) in decoded
    )
    top_label = decoded[0][1]
    return is_car, top_label

# ─────────────────────────────────────────
# RUN TEST
# ─────────────────────────────────────────
y_true = []
y_pred = []

car_folder = "test_images/cars"
not_car_folder = "test_images/not_cars"

print("\nTesting CAR images...")
for fname in os.listdir(car_folder):
    path = os.path.join(car_folder, fname)
    is_car, label = check_is_car(path)
    y_true.append(1)
    y_pred.append(1 if is_car else 0)
    print(f"  {fname:30s} -> Predicted: {'Car' if is_car else 'Not Car':10s} (top label: {label})")

print("\nTesting NOT-CAR images...")
for fname in os.listdir(not_car_folder):
    path = os.path.join(not_car_folder, fname)
    is_car, label = check_is_car(path)
    y_true.append(0)
    y_pred.append(1 if is_car else 0)
    print(f"  {fname:30s} -> Predicted: {'Car' if is_car else 'Not Car':10s} (top label: {label})")

# ─────────────────────────────────────────
# CONFUSION MATRIX
# ─────────────────────────────────────────
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["Not Car", "Car"]))
import json

cm_results = {
    "confusion_matrix": cm.tolist(),
    "labels": ["Not Car", "Car"],
    "total_images": len(y_true),
    "accuracy": float((cm[0][0] + cm[1][1]) / cm.sum())
}

with open("outputs/confusion_matrix_results.json", "w") as f:
    json.dump(cm_results, f, indent=4)

print("\nConfusion matrix data saved to outputs/confusion_matrix_results.json")

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Not Car", "Car"], yticklabels=["Not Car", "Car"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Car Image Verification (MobileNetV2)")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=150)
print("\nConfusion matrix image saved to outputs/confusion_matrix.png")
plt.show()