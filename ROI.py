import cv2
import time
from drew import *
import __setting__

max_auxiliary_number = __setting__.max_auxiliary_number


THICKNESS = 3
COLOR_MAP = {
    "white": (255, 255, 255),
    "green": (0, 255, 0),
    "red": (0, 0, 255),
    "blue": (255, 0, 0)
}

varThreshold = 100

def on_mouse_select_rectangle(event, x, y, flags, param):
    """
    This function aims for set the mouse call back to select a rectangle
    """
    global img, point1, point2, left_button_up
    img2 = img.copy()
    if event == cv2.EVENT_LBUTTONDOWN:
        # when left mouse button is pressed, get a point
        point1 = (x, y)
        cv2.circle(img2, point1, 10, (0, 255, 0), 5)
        cv2.imshow('image', img2)

    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
        # when hold left mouse button, draw a rectangle
        cv2.rectangle(img2, point1, (x, y), (255, 0, 0), thickness=THICKNESS)
        cv2.imshow('image', img2)

    elif event == cv2.EVENT_LBUTTONUP:
        # when release the left mouse button, get a rectangle
        point2 = (x, y)
        cv2.rectangle(img2, point1, point2, (0, 0, 255), thickness=THICKNESS)
        left_button_up = False # notice that it has been released the left mouse button


def get_monitored_area(capture, number, frame_width, frame_height):
    _, frame = capture.read()
    time.sleep(0.5)  # waiting for opening the camera
    _, frame = capture.read()
    frame = cv2.resize(frame, (frame_width, frame_height))

    frame_width = int(capture.get(3))
    frame_height = int(capture.get(4))

    global img, left_button_up, min_x, min_y, width, height

    img = frame
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', on_mouse_select_rectangle)
    cv2.imshow('image', frame)
    cv2.moveWindow("image", 0, 0)

    left_button_up = True

    monitored_areas = list()
    occupied_list = [max_auxiliary_number for i in range(0,number)]
    for i in range(0, number):
        while left_button_up:
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        if point1 != point2:
            min_x = min(point1[0], point2[0])
            if min_x > frame_width:
                min_x = frame_width
            min_y = min(point1[1], point2[1])
            if min_y > frame_height:
                min_y = frame_height
            width = abs(point1[0] - point2[0])
            height = abs(point1[1] - point2[1])

        monitored_areas.append(((min_x, min_y), (min_x + width, min_y + height), cv2.createBackgroundSubtractorMOG2(history=50,varThreshold=varThreshold, detectShadows=False)))
        draw_frame_box(frame, monitored_areas, occupied_list)
        left_button_up = True

    cv2.destroyAllWindows()
    return monitored_areas, occupied_list