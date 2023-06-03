from ultralytics import YOLO
import cv2
from overlays import *


# load a pretrained model
model = YOLO("/Users/louis.skowronek/Downloads/test/yolov8n_custom/weights/best.pt")


# url = 'http://192.168.0.202:4747/video'
#url = 'http://172.17.46.21/video'
# cap = cv2.VideoCapture(url)

cap = cv2.VideoCapture('/Users/louis.skowronek/Downloads/IMG_4546.MOV')

# Get the video's width, height, and frames per second (fps)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec for the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_file = '/Users/louis.skowronek/Downloads/output_4546_reduced.mp4'

bitrate = 500_000
# Create a VideoWriter object
out = cv2.VideoWriter(output_file, fourcc, bitrate, fps, (width, height))

#cap = cv2.VideoCapture('/Users/louis.skowronek/Downloads/IMG_4534.MOV')

it = 0
while cap.isOpened():

    success, img = cap.read()

    if success:# and it % 3 == 0:
        # predict on an image
        results = model(img)
        res_plotted = results[0].plot(boxes=True)

        # overlay_highest_score(cv2, res_plotted, results)
        # overlay_id(cv2, res_plotted, results)

        # Show the image
        # cv2.imshow("result", res_plotted)

        # Write the frame to the output video
        out.write(res_plotted)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    it += 1
# When everything done, release the video capture object
cap.release()
out.release()

# Closes all the frames
cv2.destroyAllWindows()

# how to train a model?
# how to generate training data for the model? -> Roboflow
