import cv2
import time
from drew import *
from ROI import *
from model import *
import __setting__

max_auxiliary_number = __setting__.max_auxiliary_number

VIDEO_SOURCE = "data/test.mp4"


FRAME_WIDTH = int(1280/4*2)
FRAME_HEIGHT = int(720/4*2)

frame_width = FRAME_WIDTH
frame_height = FRAME_HEIGHT


def filter_mask(img):
    """
        This filters are hand-picked just based on visual tests
    """


    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
    #img = cv2.medianBlur(img, 5)

    # Fill any small holes
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    # Remove noise
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

    # Dilate to merge adjacent blobs
    dilation = cv2.dilate(opening, kernel, iterations=2)

    return dilation


def detect_moving_object(fg_mask):
    wight, height = fg_mask.shape
    fg_mask[fg_mask < 200] = 0
    fg_mask = filter_mask(fg_mask)
    _, thresh = cv2.threshold(fg_mask, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

    for (i, contour) in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        contour_valid = (w <= .5 * wight) and (h <= .5 *height)

        if not contour_valid:
            return True

    return False


def main():
    number = int(input("How many parking Spaces need to be monitored："))

    capture = cv2.VideoCapture(VIDEO_SOURCE)
    capture.set(cv2.CAP_PROP_FPS, 25)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    monitored_areas, occupied_list = get_monitored_area(capture, number, frame_width, frame_height)

    cv2.namedWindow('capture')
    cv2.moveWindow("capture", 0, 0)
    current_time = int(time.time())
    while True:
        _, frame = capture.read()
        frame = cv2.resize(frame, (frame_width, frame_height))

        current_fps = round(1/(time.time() - current_time),2)
        current_time = time.time()

        for i,area in enumerate(monitored_areas):
            x_0 = area[0][0]
            y_0 = area[0][1]
            x_1 = area[1][0]
            y_1 = area[1][1]
            img = frame[y_0:y_1,x_0:x_1]
            backSub = area[2]
            fg_mask = backSub.apply(img, None, 0.001)

            if(detect_moving_object(fg_mask)):
                print(i, " :not do predict")
                continue

            print(i, " :do predict")
            out = predict(img)
            occupied_list[i] += out
            if occupied_list[i] > max_auxiliary_number:
                occupied_list[i] = max_auxiliary_number
            elif occupied_list[i] < 0:
                occupied_list[i] = 0

        total_org = (int(frame_width / 2 - 70), 30)
        draw_frame_box(frame, monitored_areas,occupied_list)
        cv2.putText(img=frame, text="FPS: " + str(current_fps), org=(total_org[0], total_org[1] + 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=COLOR_MAP["red"], thickness=TEXT_THICKNESS)
        cv2.imshow("capture", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()