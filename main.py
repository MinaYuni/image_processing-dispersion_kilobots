# Importing all necessary libraries
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage import data
from scipy import misc, ndimage
from PIL import Image, ImageFilter


def read_video(name_video):
    already_read = False
    frame_path = './data/frames/' + name_video

    # Read the video from specified path
    cam = cv2.VideoCapture("./data/videos/" + name_video + ".mp4")

    try:
        # creating a folder named data
        if not os.path.exists('data'):
            os.makedirs('data')
        if not os.path.exists('./data/frames'):
            os.makedirs('./data/frames')
        if not os.path.exists(frame_path):
            os.makedirs(frame_path)
        else:
            already_read = True
            print("Vidéo déjà découpée : " + frame_path)
    except OSError:  # if not created then raise error
        print('Error: Creating directory of data')

    if not already_read:
        print("CREATING FRAMES")

        currentframe = 0

        while True:
            # reading from frame
            ret, frame = cam.read()

            if ret:
                # if video is still left continue creating images
                name = frame_path + '/' + name_video + '_frame' + str(currentframe) + '.jpg'
                print('Creating...' + name)

                # writing the extracted images
                cv2.imwrite(name, frame)

                # increasing counter so that it will
                # show how many frames are created
                currentframe += 1
            else:
                break

    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()

    return frame_path


def edges_detection(name_image):
    # Read the original image
    path_dir = "./data/images/"
    path = path_dir + name_image + ".jpg"
    img = cv2.imread(path)

    # Convert to graycsale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)

    # Sobel Edge Detection
    sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)  # Sobel Edge Detection on the X axis
    sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)  # Sobel Edge Detection on the Y axis
    sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)  # Combined X and Y Sobel Edge Detection

    # Canny Edge Detection
    edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)  # Canny Edge Detection

    # save image
    path_save = path_dir + name_image + "_edges.jpg"
    cv2.imwrite(path_save, edges)
    print("Image with edges detection save to : " + path_save)

    cv2.destroyAllWindows()

    return path_save


def black_edges(name_image):
    # load image
    path_dir = "./data/images/"
    path = path_dir + name_image + ".jpg"
    img = cv2.imread(path)

    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # threshold
    thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)[1]

    # write result to disk
    path_save = path_dir + name_image + "_black_edges.jpg"
    cv2.imwrite(path_save, thresh)
    print("Image with edges detection save to : " + path_save)

    cv2.destroyAllWindows()

    return path_save


if __name__ == '__main__':
    # video = "test02"
    # path_save_frame = read_video(video)
    # list_files = os.listdir(path_save_frame)

    nom_image = "anneau02"
    path_save_image_e = edges_detection(nom_image)
    path_save_image_be = black_edges(nom_image)
