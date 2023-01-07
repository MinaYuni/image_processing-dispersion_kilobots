# Importing all necessary libraries
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import math
from skimage import data
from scipy import misc, ndimage
from PIL import Image, ImageFilter


def video_to_images(name_video, frames_per_second=1):
    frame_path = './data/frames/' + name_video

    try:
        # creating a folder named data
        if not os.path.exists('data'):
            os.makedirs('data')
        if not os.path.exists('./data/frames'):
            os.makedirs('./data/frames')
        if not os.path.exists(frame_path):
            os.makedirs(frame_path)
    except OSError:  # if not created then raise error
        print('Error: Creating directory of data')

    # Read the video from specified path
    cam = cv2.VideoCapture("./data/videos/" + name_video + ".mp4")

    frame_list = []
    frame_rate = cam.get(cv2.CAP_PROP_FPS)  # video frame rate

    # frame
    current_frame = 0
    num_frame = 0

    if frames_per_second > frame_rate or frames_per_second == -1:
        frames_per_second = frame_rate

    while True:
        # reading from frame
        ret, frame = cam.read()

        if ret:
            if current_frame % (math.floor(frame_rate / frames_per_second)) == 0:
                # if video is still left continue creating images
                name = frame_path + '/' + name_video + '_frame' + str(num_frame) + '.jpg'
                print('Creating...' + name)

                # adding frame to list
                frame_list.append(frame)

                # writing selected frames to images_path
                cv2.imwrite(name, frame)

                num_frame += 1

            current_frame += 1
        else:
            break

    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()

    print("Nombre d'images enregistrées :", len(frame_list))

    return frame_path, frame_list


def edges_detection(name_image, path_image=None):
    # Read image
    path_dir = "./data/images/"
    if path_image is None:
        path = path_dir + name_image + ".jpg"
    else:
        path = path_image + "/" + name_image + ".jpg"
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


def black_edges(name_image, path_image=None):
    # Read image
    path_dir = "./data/images/"
    if path_image is None:
        path = path_dir + name_image + ".jpg"
    else:
        path = path_image + "/" + name_image + ".jpg"
    img = cv2.imread(path)

    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # threshold
    thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)[1]

    # write result to disk
    new_name = name_image + "_be"
    path_save = path_dir + new_name + ".jpg"
    cv2.imwrite(path_save, thresh)
    print("Image with black edges detection save to : " + path_save)

    cv2.destroyAllWindows()

    return new_name


def remove_shadow(name_image, path_image=None):
    # Read image
    path_dir = "./data/images/"
    if path_image is None:
        path = path_dir + name_image + ".jpg"
    else:
        path = path_image + "/" + name_image + ".jpg"
    img = cv2.imread(path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Equalize the histogram
    enhanced = cv2.equalizeHist(gray)

    # Save the result
    new_name = name_image + "_rs"
    path_save = path_dir + new_name + ".jpg"
    cv2.imwrite(path_save, enhanced)
    print("Image with shadows removed save to : " + path_save)

    return new_name


def round_object(name_image, nb_kilobots, p1, p2, path_image=None):
    # Read image
    path_dir = "./data/images/"
    if path_image is None:
        path = path_dir + name_image + ".jpg"
    else:
        path = path_image + "/" + name_image + ".jpg"
    img = cv2.imread(path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use the HoughCircles function to detect circles
    # liste des objets ronds détectés
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=p1, param2=p2, minRadius=8, maxRadius=15)

    params = None
    # Loop over the circles and draw them on the image
    if circles is not None:
        if len(circles[0]) == nb_kilobots:
            params = (p1, p2)

            for (x, y, radius) in circles[0]:
                cv2.circle(img, (x, y), radius, (0, 255, 0), 2)

            # write result to disk
            path_save = path_dir + name_image + "_ro_" + str(p1) + "_" + str(p2) + ".jpg"
            cv2.imwrite(path_save, img)
            print("Image with round objects detection save to : " + path_save)

            cv2.destroyAllWindows()
    else:
        return [], params

    return circles[0], params


def if_not_list(list_of_lists, sub_list):
    is_not_in_list_of_lists = True
    for lst in list_of_lists:
        if all(elem in lst for elem in sub_list):
            is_not_in_list_of_lists = False
            break

    if is_not_in_list_of_lists:
        return True

    return False


def common_elements(list_of_lists, list_of_elements):
    common = []

    for element in list_of_lists[0]:
        if all(element in x for x in list_of_lists):
            common.append(element)

    return common


def count_elements(list_of_lists, dict_of_elements):
    for element in dict_of_elements.keys():
        dict_of_elements[element] = sum(x.count(element) for x in list_of_lists)

    return dict_of_elements


def add_cross_to_video(name_video, path_video=None):
    frame_path = './data/frames/' + name_video

    # Read the video from specified path
    cap = cv2.VideoCapture("./data/videos/" + name_video + ".mp4")

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

    currentframe = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        center = (width // 2, height // 2)
        cv2.line(frame, center, (center[0], 0), (255, 255, 255), 2)
        cv2.line(frame, center, (center[0], height), (255, 255, 255), 2)
        cv2.line(frame, center, (0, center[1]), (255, 255, 255), 2)
        cv2.line(frame, center, (width, center[1]), (255, 255, 255), 2)

        # cv2.imshow('Frame', frame)

        # if video is still left continue creating images
        name = frame_path + '/' + name_video + '_cross_frame' + str(currentframe) + '.jpg'
        print('Creating...' + name)

        # writing the extracted images
        cv2.imwrite(name, frame)

        # increasing counter so that it will
        # show how many frames are created
        currentframe += 1

    cap.release()
    cv2.destroyAllWindows()


def duration_video(name_video):
    # Load the video
    cap = cv2.VideoCapture("./data/videos/" + name_video + ".mp4")

    # Get the number of frames and the frame rate
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate the duration of the video in seconds
    duration = num_frames / fps

    cap.release()

    return duration


def analyse(name_video, nb_kilobots):
    path_frames, liste_frames = video_to_images(name_video, 2)
    list_files = os.listdir(path_frames)
    # add_cross_to_video(video)

    liste_images = []
    for file in list_files:
        liste_images.append(file[:-4])

    all_params = []
    all_circles = []
    for image in liste_images:
        # image_rs = remove_shadow(image, path_frames)
        # image_be = black_edges(image_rs)
        image_be = black_edges(image, path_frames)

        liste_params = []
        liste_circles = []

        flag_detection = False
        i = 1
        j = 1

        while not flag_detection:
            while not flag_detection:
                pos_circles, tuple_params = round_object(image_be, nb_kilobots, i, j)

                # si on a réussi à détecter le bon nombre de cercles (ça veut dire que la liste n'est pas nulle)
                if tuple_params is not None:
                    liste_params.append(tuple_params)
                    liste_circles.append(pos_circles)

                    flag_detection = True

                if j == 100:
                    break
                else:
                    j += 1

            if i == 100:
                break
            else:
                i += 1

        # print(all_circles[0][:nb_k], liste_params)
        all_params.append(liste_params)
        all_circles.append(liste_circles)

    return all_params, all_circles


if __name__ == '__main__':
    # Secondes = temps simulation / len(dictionnaire  positions récoltes) sur kilombo
    # Nombre frames = len(dictionnaire  positions récoltes) sur kilombo

    nb_k = 15
    video = "disque04_cut"

    p, c = analyse(video, nb_k)

    # params_possible = {}
    # for i in range(10, 21, 2):
    #     for j in range(10, 21, 2):
    #         params_possible[(i, j)] = 0
    # liste_test = [[(12, 12), (14, 12)], [(12, 12), (14, 12), (16, 12), (18, 12), (20, 12)], [], [(10, 14), (10, 16), (10, 18), (12, 14), (12, 16), (12, 18), (14, 14), (14, 16), (14, 18), (16, 14), (16, 16), (18, 14), (18, 16), (20, 14), (20, 16)], [(10, 14), (10, 16), (12, 14), (14, 14), (16, 14), (18, 14), (20, 14)], [(10, 14), (10, 16), (12, 14), (12, 16), (14, 14), (14, 16), (16, 14), (16, 16), (18, 14), (18, 16), (20, 14), (20, 16)], [(10, 14), (10, 16), (12, 12), (12, 14), (12, 16), (14, 12), (14, 14), (14, 16), (16, 12), (16, 14), (16, 16), (18, 12), (18, 14), (18, 16), (20, 12), (20, 14)], [(10, 16), (12, 16), (14, 16), (16, 16), (18, 16), (20, 16)], [(10, 14), (12, 14), (14, 14), (16, 14), (18, 14), (20, 14)], [], [(10, 14), (10, 16), (10, 18), (12, 14), (12, 16), (14, 14), (14, 16), (16, 14), (16, 16), (18, 14), (18, 16), (20, 14), (20, 16)], [(10, 14), (10, 16), (12, 14), (14, 14), (16, 14), (18, 14), (20, 14)], [], [(10, 14)], [(10, 14), (10, 16), (12, 14), (12, 16), (14, 14), (14, 16), (16, 14), (16, 16), (18, 14), (18, 16), (20, 14), (20, 16)], [(10, 16), (12, 16), (14, 16), (16, 16), (18, 16), (20, 16)], [(10, 14), (10, 16), (10, 18), (12, 14), (12, 16), (12, 18), (14, 14), (14, 16), (14, 18), (16, 14), (16, 16), (16, 18), (18, 14), (18, 16), (18, 18), (20, 14), (20, 16)], [(10, 14), (10, 16), (10, 18), (12, 12), (12, 14), (12, 16), (12, 18), (14, 12), (14, 14), (14, 16), (14, 18), (16, 12), (16, 14), (16, 16), (16, 18), (18, 12), (18, 14), (18, 16), (18, 18), (20, 12), (20, 14), (20, 16), (20, 18)], [(10, 16), (16, 14), (18, 14), (20, 14)], [(10, 16), (12, 16), (14, 16), (20, 14)], [(10, 14), (12, 14), (14, 12), (14, 14), (16, 12), (16, 14), (18, 12), (18, 14), (20, 12), (20, 14)], [], [(10, 14), (12, 14), (14, 14), (16, 14), (18, 14), (20, 12), (20, 14)], [(10, 14), (12, 14), (14, 14), (16, 14), (18, 14), (20, 14)], [(10, 14), (10, 16), (12, 14), (12, 16), (14, 14), (14, 16), (16, 14), (18, 14), (20, 14)], [(10, 14)], [(14, 12), (16, 12), (18, 12)], [(10, 14), (10, 16), (12, 14), (12, 16), (14, 14), (14, 16), (16, 14), (16, 16), (18, 14), (18, 16), (20, 14), (20, 16)], [], [(10, 14), (12, 14), (14, 14), (16, 14), (18, 14), (20, 14)], [(10, 14), (10, 16), (12, 14), (12, 16), (14, 14), (14, 16), (16, 14), (16, 16), (18, 14), (18, 16), (20, 14), (20, 16)]]
    # print(len(liste_test))
    # filtered_list = list(filter(None, liste_test))
    # print(len(filtered_list))
    # commun = count_elements(filtered_list, params_possible)
    # filtered_dict = {k: v for k, v in commun.items() if v != 0}
    # for key, value in filtered_dict.items():
    #     print(f"{key}: {value}")
    # sorted_dict = sorted(filtered_dict)
    # print(sorted_dict)