def train_yolo_v8():
    """
    Train a YOLO v8 model using Ultralytics.

    This function loads a pretrained YOLO v8 model and trains it further using the specified parameters.
    """
    try:
        # Import the YOLO class from the ultralytics module
        from ultralytics import YOLO

        # Load a pretrained YOLO v8 model
        model = YOLO("yolov8n.pt")

        # Train the model with specified parameters
        model.train(
            data="/Users/louis.skowronek/object-detection-project/aiss_yolo.yaml",
            pretrained=True,
            epochs=100,
            device='0',  # use GPU
            name='yolov8n_custom',
            flipud=0.5,
            degrees=0.8
        )

        print("Training completed successfully.")

    except ImportError:
        print("Please make sure you have the ultralytics library installed.")
    except Exception as e:
        print("An error occurred:", e)


if __name__ == '__main__':
    train_yolo_v8()


