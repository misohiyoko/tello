import threading
import time
from queue import Queue
import xboxcont

from djitellopy import Tello, TelloException
import numpy as np
import cv2

# PCのwebcamで動かすときはFalseにする
IS_TELLO = True
TELLO_STOP = True
TELLO_STOP = TELLO_STOP or not IS_TELLO
SHOW_UN_PROCESSED_PICTURE = False
USE_CONTROLLER = False
USE_CONTROLLER = USE_CONTROLLER and TELLO_STOP

isError = threading.Event()

aruco = cv2.aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_7X7_50)
param = aruco.DetectorParameters()
tello = Tello()
tello.connect()
tello.land()
