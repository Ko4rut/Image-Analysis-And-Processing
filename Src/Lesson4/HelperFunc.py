import cv2 as cv
import numpy as np

def showImg(img):
    cv.imshow("Image",img)
    cv.waitKey(0)
    cv.destroyAllWindows()

a = np.array([[1,2,3],
              [4,5,6]])
b = np.array([[2,2,2],
               [2,2,2]])
# a = np.ones((3,3),dtype=np.uint8)
# print(a)