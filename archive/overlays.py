def overlay_highest_score(cv2, res_plotted, results):
    '''
    get the object with the highest score
    '''
    if len(results[0].boxes.data) > 0:
        index = results[0].boxes.conf.argmax()

        xy = results[0].boxes[index].xyxy.numpy().astype(int).flatten()

        x1 = xy[0]
        y1 = xy[1]
        x2 = xy[2]
        y2 = xy[3]

        #cv2.rectangle(res_plotted, (x1, y1), (x2, y2), (0,0,0), -1)

        cv2.circle(res_plotted, center=((x2 + x1) // 2, (y2 + y1) // 2), radius=30, color=(0, 0, 255),
                   thickness=-1)

def overlay_id(cv2, res_plotted, results):
    """
    add the id to the rectangle
    """
    for res in results[0].boxes:

        number = str(int(res.cls.item()))

        # Define the text, font, and font scale
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 3.0

        # Define the bounding box
        bbox = res.xyxy.numpy().astype(int).flatten()  # (x1, y1, x2, y2)

        # Get the width and height of the text
        text_size, _ = cv2.getTextSize(number, font, font_scale, thickness=3)
        text_width, text_height = text_size

        # Calculate the center point of the bounding box
        center_x = (bbox[0] + bbox[2]) // 2
        center_y = (bbox[1] + bbox[3]) // 2

        # Calculate the starting point of the text
        start_x = center_x - text_width // 2
        start_y = center_y + text_height // 2

        # Put the text centered in the bounding box
        cv2.putText(res_plotted, f'{number}', (start_x, start_y), font, font_scale, (0, 0, 255), thickness=3, lineType=cv2.LINE_AA)



