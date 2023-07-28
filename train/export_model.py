from ultralytics import YOLO

# Load a model
#model = YOLO('yolov8n.pt')  # load an official model
model = YOLO('../models/yolov8s_best.pt')  # load a custom trained

# Export the model
model.export(format='onnx', opset=15)
