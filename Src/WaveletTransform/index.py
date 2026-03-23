import cv2 as cv
import numpy as np
import os

BASE_DIR = os.path.dirname(__file__)
image_path = os.path.join(BASE_DIR, "..", "..", "Images", "Head_Film.png")

image = cv.imread(image_path, cv.IMREAD_GRAYSCALE).astype(np.float32)

def showImg(img):
    cv.imshow("Image", img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def normalize_for_display(img):
    img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX)
    return img.astype(np.uint8)

def haar_1d(signal):
    n = len(signal)
    half = n // 2
    output = np.zeros(n, dtype=np.float32)

    for i in range(half):
        output[i] = (signal[2*i] + signal[2*i+1]) / 2
        output[half + i] = (signal[2*i] - signal[2*i+1]) / 2

    return output

def haar_2d(img):
    temp = np.apply_along_axis(haar_1d, 1, img)   # rows
    result = np.apply_along_axis(haar_1d, 0, temp) # cols
    return result

def haar_2d_multilevel(image, levels=3):
    result = image.copy().astype(np.float32)
    h, w = result.shape

    for level in range(levels):
        current_h = h // (2 ** level)
        current_w = w // (2 ** level)

        region = result[:current_h, :current_w]
        result[:current_h, :current_w] = haar_2d(region)

    return result

wl3 = haar_2d_multilevel(image, levels=1)
showImg(normalize_for_display(wl3))


