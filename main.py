import threading
import time
from queue import Queue
import xboxcont

from djitellopy import Tello, TelloException
import numpy as np
import cv2

# PCのwebcamで動かすときはFalseにする
IS_TELLO =False
TELLO_STOP = True
TELLO_STOP = TELLO_STOP or not IS_TELLO
SHOW_UN_PROCESSED_PICTURE = False
USE_CONTROLLER = False
USE_CONTROLLER = USE_CONTROLLER and TELLO_STOP

isError = threading.Event()

aruco = cv2.aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
param = aruco.DetectorParameters()


class TelloControl:
    def __init__(self):
        self.height = None
        self.width = None
        if IS_TELLO:
            self.tello = Tello()
            while True:
                try:
                    self.tello.connect()
                    break
                except TelloException as e:
                    print(e)
                time.sleep(2)
            self.tello.streamon()
            if not TELLO_STOP:
                self.tello.takeoff()
            else:
                self.tello.land()
                if USE_CONTROLLER:
                    self.joy = xboxcont.XboxController()
            self.frame_read = self.tello.get_frame_read()
        else:
            self.cap = cv2.VideoCapture(0)

        self.picture = Queue(512)

        self.isError = False
        self.img_prev = None
        self.t_img = threading.Timer(1 / 30, self.image_proc)
        self.t_img.start()
        self.t_batt = threading.Timer(1 / 30, self.tello_check_battery)
        self.t_batt.start()
        self.t_flight = threading.Timer(1 / 30, self.take_image)
        self.t_flight.start()

    def tello_check_battery(self):
        if self.isError:
            return
        if IS_TELLO:
            if self.tello.get_battery() < 20:
                print("Battery is low at {}.".format(self.tello.get_battery()))
                self.isError = True
                self.tello.land()
            t = threading.Timer(1 / 30, self.tello_check_battery)
            t.start()

    def image_proc(self):
        if self.isError:
            return
        img = self.get_pict()

        # ここにimgの処理をかく
        kernel_size = 5
        self.height, self.width, _ = img.shape[:3]
        _, img_gray, _ = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        img_gray_blur = cv2.GaussianBlur(img_gray, (kernel_size, kernel_size), 0)
        img_blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        img_sobel_x = cv2.Sobel(img_blurred, cv2.CV_64F, 1, 0, ksize=3)
        img_sobel_x = cv2.convertScaleAbs(img_sobel_x)
        img_sobel_y = cv2.Sobel(img_blurred, cv2.CV_64F, 0, 1, ksize=3)
        img_sobel_y = cv2.convertScaleAbs(img_sobel_y)
        img_sobel_combined = cv2.addWeighted(img_sobel_x, 0.5, img_sobel_y, 0.5, 0)
        img_sobel_combined_gray = cv2.cvtColor(img_sobel_combined, cv2.COLOR_BGR2GRAY)
        img_canny = cv2.Canny(img_gray, 100, 200, apertureSize=3, L2gradient=True)
        thr, img_bin = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY)
        img_bin = img_bin.astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernel)


        kernel_size = 3
        img_canny_dilated = cv2.dilate(img_canny,
                                       kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
                                       iterations=1)
        img_canny_inverted = cv2.bitwise_not(img_canny_dilated)

        _, img_binarized = cv2.threshold(img_sobel_combined_gray, 20, 255, cv2.THRESH_BINARY_INV)
        img_binarized = img_binarized.astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        img_binarized_morphed = cv2.morphologyEx(img_binarized, cv2.MORPH_OPEN, kernel)

        contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(img, contours, -1, (0, 0, 255), 2)
        circles = []
        img_disp = np.copy(img)




        if self.img_prev is not None:
            kernel = np.ones((5, 5), np.uint8)
            img_abs = np.abs(img_gray_blur.astype(np.float32) - self.img_prev.astype(np.float32)).astype(np.uint8)

            #self.picture.put((img_gray_blur,))
        for cnt in contours:
            area = cv2.contourArea(cnt)
            length = cv2.arcLength(cnt, True)
            length = length if length > 0 else 1
            x_c, y_c, w_c, h_c = cv2.boundingRect(cnt)
            cv2.drawContours(img_disp, [cnt], -1, (0, 0, 255), 2)
            circularity = 4 * np.pi * area / length / length
            if circularity > 0.7 and length > 150:
                cv2.putText(img_disp, f"{circularity:.3f} {length:.3f}", (x_c, y_c - 10), cv2.FONT_HERSHEY_PLAIN, 2,
                            (0, 255, 0), 1,
                            cv2.LINE_AA)

                cv2.rectangle(img_disp, (x_c, y_c), (x_c + w_c, y_c + h_c), (0, 255, 0), 2)
                circles.append((x_c, y_c, w_c, h_c))
        circles.sort(key= lambda cir: cir[2])
        # ここまでimgの処理
        if IS_TELLO:
            # 移動の関数
            self.flight_control(circles)

        self.picture.put((img_disp,))
        self.img_prev = img_gray_blur
        t = threading.Timer(1 / 30, self.image_proc)
        t.start()

    def get_pict(self):
        if IS_TELLO:
            img = self.frame_read.frame
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            _, img = self.cap.read()
        return img

    def take_image(self):
        if self.isError:
            return
        if SHOW_UN_PROCESSED_PICTURE:
            img = self.get_pict()
            self.picture.put((img,))
            t = threading.Timer(1 / 30, self.take_image)
            t.start()

    def get_image(self):
        return self.picture.get()

    def flight_control(self, circles):
        if self.isError:
            return
        # ここにTelloを動かす処理を書く
        if len(circles) > 0:
            target = circles[0]
            x, y = (target[0] + target[2] / 2, target[1] + target[3] / 2)
            print(f"x:{x - self.width / 2} y:{y - self.height / 2}")

            if not TELLO_STOP:
                if abs(x - self.width / 2) > 40:
                    if (x - self.width / 2) < 0:
                        self.tello.send_rc_control(-20, 0, 0, 0)
                    else:
                        self.tello.send_rc_control(20, 0, 0, 0)
                else:
                    if abs(y - self.height / 2) > 40:
                        if y < self.height / 2:
                            self.tello.send_rc_control(0, 0, 20, 0)
                        else:
                            self.tello.send_rc_control(0, 0, -20, 0)
                    else:
                        self.tello.send_rc_control(0, 0, 0, 0)
                        #self.tello.move_forward(100)

            elif USE_CONTROLLER:
                (x, y, a, b, rb) = self.joy.read()
                print((x, y, a, b, rb))
        # ここまで
        else:
            self.tello.send_rc_control(0, 0, 0, 0)


if __name__ == '__main__':
    tello_control = TelloControl()

    while True:
        img = tello_control.get_image()[0]
        cv2.imshow('frame', img)

        key = cv2.waitKey(int(1000 / 300))
