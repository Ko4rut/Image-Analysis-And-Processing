import cv2 as cv
import numpy as np
import os

BASE_DIR = os.path.dirname(__file__)
image_path = os.path.join(BASE_DIR, "..", "..", "Images", "lena_std.tif")

image = cv.imread(image_path)
# print(image.shape)

def transferGrayScale(img) -> np.ndarray | None:
    if img is None:
        return None
    else:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        return img

def detectTheBorderImg(img) -> np.ndarray | None:
    if img is None:
        return None

    if len(img.shape) == 3:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        gray = img

    # Sobel theo x và y
    gx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
    gy = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)

    # Độ lớn gradient
    mag = np.sqrt(gx**2 + gy**2)

    # Chuẩn hóa về 0–255
    mag = np.clip(mag, 0, 255).astype(np.uint8)

    return mag

def detectNegativeImg(img):
    if img.ndim == 2:  
        print("grayscale image")
        return 255 - img

    elif img.ndim == 3 and img.shape[2] == 3: 
        print("color image")
        L = np.max(img)
        return (L - 1 - img).astype(np.uint8)

    else:
        raise ValueError("Invalid image")

    
def showImage(img) -> None:
    cv.imshow("Image", img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def gamma_correction(img, gamma: float):
    img_norm = img / 255.0
    corrected = np.power(img_norm, gamma)
    return (corrected * 255).astype(np.uint8)

# imageEdge = detectTheBorderImg(image)
# showImage(imageEdge)
# image_transferred = transferGrayScale(image)
# showImage(image_transferred)
# image_transferred = gamma_correction(image_transferred,2)
# image_transferred = detectNegativeImg(image)
# showImage(image_transferred)
