# Importing all necessary libraries
import cv2
import os


def read_video(name_video):
    # Read the video from specified path
    cam = cv2.VideoCapture("./data/videos/" + name_video)

    try:
        # creating a folder named data
        if not os.path.exists('data'):
            os.makedirs('data')

    # if not created then raise error
    except OSError:
        print('Error: Creating directory of data')

    # frame
    currentframe = 0

    while (True):

        # reading from frame
        ret, frame = cam.read()

        if ret:
            # if video is still left continue creating images
            name = './data/frames/' + name_video + '-frame' + str(currentframe) + '.jpg'
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


if __name__ == '__main__':
    name_video = "video_test01.mp4"
    read_video(name_video)
