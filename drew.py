import numpy as np
import cv2
import __setting__

max_auxiliary_number = __setting__.max_auxiliary_number

COLOR_MAP = {
    "white": (255, 255, 255),
    "green": (0, 255, 0),
    "red": (0, 0, 255),
    "blue": (255, 0, 0),
    "yellow": (255, 255, 0),
    "black": (0, 0, 0)
}

LINE_THICKNESS = 3
TEXT_THICKNESS = 2
BOXES_THICKNESS = 1


def draw_frame_box(img, monitored_areas, occupied_list):
    for i,area in enumerate(monitored_areas):
        if occupied_list[i] >= max_auxiliary_number/2:
            cv2.rectangle(img=img, pt1=area[0], pt2=area[1], color=COLOR_MAP["red"], thickness=LINE_THICKNESS)
        else:
            cv2.rectangle(img=img, pt1=area[0], pt2=area[1], color=COLOR_MAP["green"], thickness=LINE_THICKNESS)