def main():
    """
    Main function to load a YOLO model and export it to ONNX format.
    """
    try:
        # Import the YOLO class from the ultralytics module
        from ultralytics import YOLO

        # Load a YOLO model (replace the model path with your own)
        #model = YOLO('yolov8n.pt')  # load an official model
        model = YOLO('../models/yolov8s_best.pt')  # load a custom trained model

        # Export the YOLO model to ONNX format with specific opset version
        model.export(format='onnx', opset=15)

        print("Model exported to ONNX format successfully.")

    except ImportError:
        print("Please make sure you have the ultralytics library installed.")
    except Exception as e:
        print("An error occurred:", e)


if __name__ == "__main__":
    main()
