import time
import cv2
import os
import math
import shutil
# import matplotlib.pyplot as plt
# import numpy as np
# from skimage import data
# from scipy import misc, ndimage
# from PIL import Image, ImageFilter


def video_to_images(name_video, name_file=None, frames_per_second=1):
    print("----------------------------------------------------")
    print("VIDEO : " + name_video.upper())
    print("----------------------------------------------------")

    if name_file is None:
        path_video = "./data/videos/" + name_video + ".mp4"
        frame_path = './data/frames/' + name_video
    else:
        path_video = "./data/videos/" + name_file + "/" + name_video + ".mp4"
        frame_path = "./data/frames/" + name_file + "/" + name_video

    try:
        # creating a folder named data
        if not os.path.exists("data"):
            os.makedirs("data")
        if not os.path.exists("./data/frames"):
            os.makedirs("./data/frames")
        if name_file is not None:
            if not os.path.exists("./data/frames/" + name_file):
                os.makedirs("./data/frames/" + name_file)
        if not os.path.exists(frame_path):
            os.makedirs(frame_path)
        else:
            shutil.rmtree(frame_path)
            os.mkdir(frame_path)
    except OSError:  # if not created then raise error
        print('Error: Creating directory of data')

    # Read the video from specified path
    cam = cv2.VideoCapture(path_video)

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
                if len(str(num_frame)) == 1:
                    num_frame_str = "000" + str(num_frame)
                elif len(str(num_frame)) == 2:
                    num_frame_str = "00" + str(num_frame)
                elif len(str(num_frame)) == 3:
                    num_frame_str = "0" + str(num_frame)
                else:
                    num_frame_str = str(num_frame)

                name = frame_path + '/' + name_video + '_frame' + num_frame_str + '.jpg'
                # print('Creating...' + name)

                # writing selected frames to images_path
                cv2.imwrite(name, frame)

                num_frame += 1

            current_frame += 1
        else:
            break

    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()

    return frame_path


def add_cross_to_image(name_video, name_image):
    path_data_image = "./data/images/"

    img = cv2.imread(name_image+".jpg")

    # add cross
    height, width, _ = img.shape
    center = (width // 2, height // 2)
    cv2.line(img, center, (center[0], 0), (255, 255, 255), 2)
    cv2.line(img, center, (center[0], height), (255, 255, 255), 2)
    cv2.line(img, center, (0, center[1]), (255, 255, 255), 2)
    cv2.line(img, center, (width, center[1]), (255, 255, 255), 2)

    # save result
    new_name = name_image + "_cross"
    path_dir = path_data_image + "/" + name_video
    path_save = path_dir + "/" + new_name + ".jpg"
    cv2.imwrite(path_save, img)
    print("Image with shadows removed save to : " + path_save)

    return new_name, path_dir


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


def detect_black_edges(name_video, name_image, path_image=None):
    # Read image
    path_data_image = "./data/images/"
    if path_image is None:
        path = path_data_image + name_image + ".jpg"
    else:
        path = path_image + "/" + name_image + ".jpg"
    img = cv2.imread(path)

    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # threshold
    thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)[1]

    # write result to disk
    new_name = name_image + "_be"
    path_dir = path_data_image + name_video + "/be"
    path_save = path_dir + "/" + new_name + ".jpg"
    cv2.imwrite(path_save, thresh)
    # print("Image with black edges detection save to : " + path_save)

    cv2.destroyAllWindows()

    return new_name, path_dir


def remove_shadow(name_video, name_image, path_image=None):
    # Read image
    path_data_image = "./data/images/"
    if path_image is None:
        path = path_data_image + name_image + ".jpg"
    else:
        path = path_image + "/" + name_image + ".jpg"
    img = cv2.imread(path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Equalize the histogram
    enhanced = cv2.equalizeHist(gray)

    # Save the result
    new_name = name_image + "_rs"
    path_dir = path_data_image + name_video + "/rs"
    path_save = path_dir + "/" + new_name + ".jpg"
    cv2.imwrite(path_save, enhanced)
    # print("Image with shadows removed save to : " + path_save)

    cv2.destroyAllWindows()

    return new_name, path_dir


def detect_round_objects(name_video, name_image, p1, p2, nb_kilobots=-1, path_image=None, obj_type="kilobots"):
    # Read image
    path_data_image = "./data/images/"
    if path_image is None:
        path = path_data_image + name_image + ".jpg"
    else:
        path = path_image + "/" + name_image + ".jpg"
    img = cv2.imread(path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use the HoughCircles function to detect circles
    if obj_type == "arena":
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=p1, param2=p2, minRadius=100, maxRadius=140)
    else:  # detection de kilobots par défaut
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=p1, param2=p2, minRadius=8, maxRadius=15)

    params = None
    # Loop over the circles and draw them on the image
    if circles is not None:
        if (obj_type == "kilobots" and len(circles[0]) == nb_kilobots) or (obj_type == "arena" and len(circles[0]) == 1):
            params = (p1, p2)

            for (x, y, radius) in circles[0]:
                cv2.circle(img, (x, y), radius, (0, 255, 0), 2)

            # save result
            if obj_type == "arena":
                new_name = name_image + "_arena_" + str(p1) + "_" + str(p2)
                path_dir = path_data_image + name_video
                path_save = path_dir + "/" + new_name + ".jpg"
                cv2.imwrite(path_save, img)
                # print("Image with arena detection save to : " + path_save)
            else:
                new_name = name_image + "_kb_" + str(p1) + "_" + str(p2)
                path_dir = path_data_image + name_video + "/kb"
                path_save = path_dir + "/" + new_name + ".jpg"
                cv2.imwrite(path_save, img)
                # print("Image with kilobots detection save to : " + path_save)
        else:
            return [], params, ""
    else:
        return [], params, ""

    return circles[0], params, path_save


def if_not_list(list_of_lists, sub_list):
    is_not_in_list_of_lists = True

    for lst in list_of_lists:
        if all(elem in lst for elem in sub_list):
            is_not_in_list_of_lists = False
            break

    if is_not_in_list_of_lists:
        return True

    return False


def count_elements(list_of_lists, dict_of_elements):
    for element in dict_of_elements.keys():
        dict_of_elements[element] = sum(x.count(element) for x in list_of_lists)

    return dict_of_elements


def point_in_circle(point_x, point_y, center_x, center_y, radius):
    # Calculate the distance between the point and the center of the circle
    distance = math.sqrt((point_x - center_x) ** 2 + (point_y - center_y) ** 2)

    # Check if the distance is less than or equal to the radius
    if distance <= radius:
        return True
    else:
        return False


def get_position_arena(name_video, path_images):
    x_circle = -1
    y_circle = -1
    radius_circle = -1
    flag_arena_detected = False

    liste_images_test = os.listdir(path_images)
    for img_test in liste_images_test:
        for i in range(30, 36):
            liste_circles, parameters, path_save = detect_round_objects(name_video, img_test[:-4], 10, i, path_image=path_images, obj_type="arena")
            if parameters is not None:
                x_circle = liste_circles[0][0]
                y_circle = liste_circles[0][1]
                radius_circle = liste_circles[0][2]
                flag_arena_detected = True
                break
        if flag_arena_detected:
            break

    return x_circle, y_circle, radius_circle


def analyse(name_video, nb_kilobots, name_file=None):
    f = open("./data/"+name_video+".txt", "w")
    f.write(str(nb_kilobots)+"\n")

    if name_file is None:
        path_frames = video_to_images(name_video, frames_per_second=2)
    else:
        path_frames = video_to_images(name_video, name_file=name_file, frames_per_second=2)

    # name of all the frames of the video
    name_frames = []
    liste_frames = os.listdir(path_frames)
    for frame in liste_frames:
        name_frames.append(frame[:-4])

    # detection of black edges for all frames od the video
    for frame in name_frames:
        # image_rs = remove_shadow(name_video, image, path_frames)
        # image_be = black_edges(name_video, image_rs)
        image_be, path_img_be = detect_black_edges(name_video, frame, path_frames)

    # name of all the images of the black edges detection
    name_images = []
    liste_images = os.listdir(path_img_be)
    for img in liste_images:
        name_images.append(img[:-4])

    x_arena, y_arena, radius_arena = get_position_arena(name_video, path_img_be)

    all_params = []
    all_circles = []

    # kilobots detection
    for image in name_images:
        liste_params = []
        liste_circles = []

        flag_detection = False
        i = 10
        j = 10

        while not flag_detection:
            while not flag_detection:
                pos_circles, tuple_params, path_image_saved = detect_round_objects(name_video, image, i, j, nb_kilobots=nb_kilobots, path_image=path_img_be)

                # if we managed to detect the right number of circles
                if tuple_params is not None:
                    # check that all detected circles are in the arena
                    nb_correct = 0
                    for circle in pos_circles:
                        if point_in_circle(circle[0], circle[1], x_arena, y_arena, radius_arena):
                            nb_correct += 1
                    if nb_correct == nb_kilobots:
                        flag_detection = True
                        liste_params.append(tuple_params)
                        liste_circles.append(pos_circles)

                        # save kilobot position
                        for circle in pos_circles:
                            line = f"{image[25:29]},{circle[0]},{circle[1]}\n"
                            f.write(line)
                    else:
                        os.remove(path_image_saved)
                        # print("False-positive, remove " + path_image_saved)

                if j == 100:  # timeout
                    break
                else:
                    j += 1

            if i == 100:  # timeout
                break
            else:
                i += 1

        # print(all_circles[0][:nb_k], liste_params)
        all_params.append(liste_params)
        all_circles.append(liste_circles)

    f.close()

    return all_circles, all_params


def create_folder(name_video):
    try:
        # creating a folder named data
        if not os.path.exists("data"):
            os.makedirs("data")
        if not os.path.exists("./data/images"):
            os.makedirs("./data/images")
        if not os.path.exists("./data/images/" + name_video):
            os.makedirs("./data/images/" + name_video)
        else:  # reset folder
            shutil.rmtree("./data/images/" + name_video)
            os.makedirs("./data/images/" + name_video)
        if not os.path.exists("./data/images/" + name_video + "/be"):
            os.makedirs("./data/images/" + name_video + "/be")
        if not os.path.exists("./data/images/" + name_video + "/rs"):
            os.makedirs("./data/images/" + name_video + "/rs")
        if not os.path.exists("./data/images/" + name_video + "/kb"):
            os.makedirs("./data/images/" + name_video + "/kb")
    except OSError:  # if not created then raise error
        print('Error: Creating directory of data')


if __name__ == '__main__':
    # Secondes = temps simulation / len(dictionnaire  positions récoltes) sur kilombo
    # Nombre frames = len(dictionnaire  positions récoltes) sur kilombo

    nb_k = 15
    path_video = "./data/videos/"
    video = ""
    file = "2023-01-06"

    if file != "":
        liste_videos = os.listdir(path_video + file)
        for v in liste_videos:
            start = time.perf_counter()
            create_folder(v[:-4])
            c, p = analyse(v[:-4], nb_k, name_file=file)
            end = time.perf_counter()
            tps = end - start
            print(f"Temps d'exécution : {tps} s")
    else:
        c, p = analyse(video, nb_k)
