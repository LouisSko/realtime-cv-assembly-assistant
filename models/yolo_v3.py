import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import seaborn as sns


url = 'http://192.168.0.202:4747/video'
#cap = cv2.VideoCapture(url)
cap = cv2.VideoCapture('/Users/louis.skowronek/IMG_4358.MOV')

# Load YOLOv3 weights and configuration files
net = cv2.dnn.readNet("/Users/louis.skowronek/yolov3.weights", "/Users/louis.skowronek/yolov3.cfg")

# Load classes
classes = []
with open("/Users/louis.skowronek/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# specify color for each class
dict_colors = {}
it = 0
palette = sns.color_palette("deep", len(classes))
palette = np.round(np.array(palette) * 255).astype(float)
for i, cat in enumerate(classes):
    dict_colors[cat] = palette[i]

#print(dict_colors)

# Define the input and output layers
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

# Load image
#img = cv2.imread("image.jpg")

it = 0
while True:

    ret, img = cap.read()

    if it % 5 == 0:

        height, width, channels = img.shape

        # Preprocess image
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)

        # Set the input
        net.setInput(blob)

        # Run the forward pass
        outs = net.forward(ln)

        # Initialize variables
        class_ids = []
        confidences = []
        boxes = []

        # Loop over each detection
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply non-max suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Draw the bounding boxes and labels
        font = cv2.FONT_HERSHEY_PLAIN

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = classes[class_ids[i]]
                color = dict_colors[label]
                conf = np.round(confidences[i], 2)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, f'{label} - {conf}', (x, y + 30), font, 3, color, 3)

        # Show the image
        cv2.imshow("Test", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    it += 1

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()

#cv2.waitKey(0)
#cv2.destroyAllWindows()



#labeling: RoboFlow
