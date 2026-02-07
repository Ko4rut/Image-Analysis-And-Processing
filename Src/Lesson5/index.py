import numpy as np
import cv2 as cv
from HelperFunc import showImg
import os 

BASE_DIR = os.path.dirname(__file__)
image_path = os.path.join(BASE_DIR, "..", "..", "Images", "Head_Film.png")


image = cv.imread(image_path)
image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)





showImg(image)