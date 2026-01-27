import cv2 as cv


def showImg(img):
    cv.imshow("Image",img)
    cv.waitKey(0)
    cv.destroyAllWindows()
