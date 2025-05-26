# YOLOv8 License Plate Detection 🚘📸

This project focuses on fine-tuning **YOLOv8** for detecting license plates from custom datasets. The model has been trained on annotated images of vehicles with license plates and is capable of running inference with high accuracy and speed.

---

## 📂 Dataset

The dataset used for training is available on Kaggle:

🔗 [Bangladeshi Vehicle License Plate Dataset](https://www.kaggle.com/datasets/sifatkhan69/bangladeshi-vehicle-license-plate)

It contains images of vehicles from Bangladesh with corresponding license plate annotations in YOLO format.

---

## 🧠 Load the Model Weights

1. First, upload your trained weights (`best.pt`) to your Kaggle Notebook.
2. Then, use the following code to load the trained YOLOv8 model:

```python
from models.experimental import attempt_load
import torch

# Path to the saved weights
weights_path = '/kaggle/working/yolo_license_plate/yolov11_license_plate/weights/best.pt'  # <-- Update this with your Kaggle path if different

# Load the model
model = attempt_load(weights_path, map_location='cpu')  # Use 'cuda' if running on GPU
model.eval()

# Dummy input for inference
img = torch.zeros((1, 3, 640, 640))  # Replace with actual image tensor
pred = model(img)

print("✅ Inference completed. Predictions:")
print(pred)
```
