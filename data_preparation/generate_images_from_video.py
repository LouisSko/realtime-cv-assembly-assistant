import cv2
import os


def generate_images_from_video(dir_video, file, dir_save=None, frames_per_second=1):

    if dir_save is None:
        dir_save=dir_video
    cap = cv2.VideoCapture(os.path.join(dir_video, file))

    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # make image directory if necessary
    img_directory = os.path.join(dir_save, 'images')
    if not os.path.exists(img_directory):
        os.makedirs(img_directory)

    it = 0
    f = 0

    print(f'Generate images. Check {img_directory}')
    # get one image per second
    while cap.isOpened():
        success, img = cap.read()

        if success and (it % (fps*frames_per_second) == 0):
            file_name = file.split('.')[0]+'_'+str(f)+'.jpeg'
            cv2.imwrite(os.path.join(img_directory, file_name), img, [cv2.IMWRITE_JPEG_QUALITY, 100])
            f += 1
        it += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the video capture object
    cap.release()
    print(f'images generated')


if __name__ == "__main__":
    file_video ='grey_axle_short_and_grey_axle_long.mp4'
    dir_video = '/Users/louis.skowronek/object-detection-project/videos/'
    dir_save = '/Users/louis.skowronek/AISS/generate_images/grey_axle_short_and_grey_axle_long.mp4'
    generate_images_from_video(dir_video, file_video, dir_save)
