# YOLOv8 License Plate Detection 🚘📸

This project focuses on fine-tuning **YOLOv8** for detecting license plates from custom datasets. The model has been trained on annotated images of vehicles with license plates and is capable of running inference with high accuracy and speed.

---

##  Load the Model Weights
At first upload the output weights on your kaggle notebook then
Use the following code to load the trained YOLOv8 model:
```
python
Copy
Edit
from models.experimental import attempt_load
import torch

# Path to the saved weights
weights_path = '/kaggle/working/yolo_license_plate/yolov11_license_plate/weights/best.pt' #GIVE YOUR KAGGLE PATH

# Load the model
model = attempt_load(weights_path, map_location='cpu')  # Change to 'cuda' if using GPU
model.eval()

# Dummy input for inference
img = torch.zeros((1, 3, 640, 640))  # Replace with actual image tensor
pred = model(img)

print("✅ Inference completed. Predictions:")
print(pred)
```


