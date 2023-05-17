from ultralytics import YOLO
import cv2
from overlays import *

# load a pretrained model

def train_yolo_v8():
    model = YOLO("yolov8n.pt")

    model.train(data="/Users/louis.skowronek/object-detection-project/aiss_yolo.yaml",
                pretrained=True,
                epochs=5,
                name='yolov8n_custom')


if __name__ == '__main__':
    train_yolo_v8()
    