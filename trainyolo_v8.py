from ultralytics import YOLO
import cv2
from overlays import *

# load a pretrained model
model = YOLO("yolov8n.pt")

model.train(data="/Users/louis.skowronek/aiss_yolo.yaml",
            pretrained=True,
            epochs=2,
            name='yolov8n_custom')