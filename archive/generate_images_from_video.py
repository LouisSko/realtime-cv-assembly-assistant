import cv2
import os


def extract_frames(path_video, dir_save=None, frames_per_second=1):
    """
    Generate images from a given video file at a specified rate.

    Args:
        dir_video (str): Directory where the video file is located.
        file (str): Video file name.
        dir_save (str, optional): Directory to save the generated images.
            If None, images will be saved in the same directory as the video. Defaults to None.
        frames_per_second (int, optional): Number of frames to capture per second of video. Defaults to 1.
    """

    # Create a video capture object
    cap = cv2.VideoCapture(path_video)

    # extract video file name
    file = path_video.split('/')[-1]

    # Get frames per second of the video
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Prepare the images directory, create if it doesn't exist
    img_directory = dir_save
    if not os.path.exists(img_directory):
        os.makedirs(img_directory)

    it = 0  # Frame iterator
    f = 0  # Image file count

    print(f'Generate images. Check {img_directory}')

    # Loop to capture images from video
    while cap.isOpened():
        success, img = cap.read()  # Read a frame from the video

        # If the frame read is not successful, break the loop
        if not success:
            break

        # Write the image to file every 'frames_per_second' frames
        if success and (it % (fps*frames_per_second) == 0):
            file_name = file.split('.')[0] + '_' + str(f) + '.jpeg'  # Create image file name
            # Write the image to file
            cv2.imwrite(os.path.join(img_directory, file_name), img, [cv2.IMWRITE_JPEG_QUALITY, 100])
            f += 1  # Increment image file count

        it += 1  # Increment frame iterator

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the video capture object
    cap.release()

    print(f'images generated')



if __name__ == "__main__":
    path_video = '/videos/black_axle_and_black_beam.mp4'
    dir_save = '/Users/louis.skowronek/AISS/generate_images/data/black_axle_and_black_beam/images'
    extract_frames(path_video, dir_save)
