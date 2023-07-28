from ultralytics import YOLO
import cv2

# load a pretrained model
model = YOLO("/Users/louis.skowronek/Downloads/test/yolov8n_custom/weights/yolov8s_best.pt")


url = 'http://192.168.0.202:4747/video'
#url = 'http://172.17.46.21/video'
cap = cv2.VideoCapture(url)

#qcap = cv2.VideoCapture('/Users/louis.skowronek/Downloads/IMG_4378.MOV')

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
        cv2.imshow("result", res_plotted)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    it += 1
# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()

# how to train a model?
# how to generate training data for the model? -> Roboflow
