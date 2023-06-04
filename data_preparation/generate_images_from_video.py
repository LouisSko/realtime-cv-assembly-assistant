import cv2
import os


def generate_images_from_video(dir_video, file):

    cap = cv2.VideoCapture(os.path.join(dir_video, file))

    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # make image directory if necessary
    img_directory = os.path.join(dir_video, 'images')
    if not os.path.exists(img_directory):
        os.makedirs(img_directory)

    it = 0
    f = 0

    print(f'Generate images. Check {img_directory}')
    # get one image per second
    while cap.isOpened():
        success, img = cap.read()

        if success and (it % fps == 0):
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
    file_video ='IMG_4580.MOV'
    dir_video = '/Users/louis.skowronek/Downloads/generate_images'
    generate_images_from_video(dir_video, file_video)
