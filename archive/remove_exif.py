from PIL import Image
import piexif
import os
import pandas as pd



if __name__ == '__main__':
    dir_images = "/Users/louis.skowronek/aiss_images/images"
    remove_exif(dir_images)
