import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt

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
    print(img[:5,0])
    img_norm = img / 255.0
    print(img_norm[:5,0])
    corrected = np.power(img_norm, gamma)
    print(corrected[:5,0])
    img = (corrected * 255).astype(np.uint8)
    print(img[:5,0])
    return img


def log_transform(img):
    print("Original:")
    print(img[:5, 0])

    img = img.astype(np.float32)

    log_img = np.log1p(img)   # log(1 + r)
    print("\nAfter log:")
    print(log_img[:5, 0])

    log_img = log_img / np.max(log_img) * 255
    log_img = log_img.astype(np.uint8)

    print("\nFinal image:")
    print(log_img[:5, 0])

    return log_img

def bit_plane_slicing(img):
    planes = []

    for k in range(8):
        print(f"\nBit plane {k}:")

        # 1. Tách bit thứ k
        plane = (img >> k) & 1
        print(plane[:5, 0])   # debug giống style bạn

        # 2. Đưa về ảnh hiển thị (0 hoặc 255)
        plane = (plane * 255).astype(np.uint8)

        planes.append(plane)

    return planes

def drawHis(img):
    plt.figure()
    plt.hist(img.flatten(), bins=256, range=(0,256))
    plt.title("Histogram of grayscale image")
    plt.xlabel("Gray level")
    plt.ylabel("Number of pixels")
    plt.show()


def histogram_equalization(img):
    # 1. In pixel gốc (debug)
    print("Original pixels:")
    print(img[:5, 0])

    # 2. Tính histogram
    hist = np.zeros(256, dtype=int)
    for value in img.flatten():
        hist[value] += 1

    # 3. Tính PDF
    num_pixels = img.size
    pdf = hist / num_pixels

    # 4. Tính CDF
    cdf = np.cumsum(pdf)

    print("\nCDF (first 10 values):")
    print(cdf[:10])

    # 5. Ánh xạ mức xám
    img_eq = np.floor(255 * cdf[img]).astype(np.uint8)

    # 6. In pixel sau cân bằng
    print("\nEqualized pixels:")
    print(img_eq[:5, 0])

    return img_eq

def histogram_specification(img, target_hist):
    # 1. Histogram & CDF ảnh gốc
    hist, _ = np.histogram(img.flatten(), 256, [0,256])
    pdf = hist / hist.sum()
    cdf_src = np.cumsum(pdf)

    # 2. CDF histogram mục tiêu
    target_hist = target_hist / np.sum(target_hist)
    cdf_target = np.cumsum(target_hist)

    # 3. Tạo bảng ánh xạ
    mapping = np.zeros(256, dtype=np.uint8)

    for i in range(256):
        diff = np.abs(cdf_src[i] - cdf_target)
        mapping[i] = np.argmin(diff)

    # 4. Ánh xạ ảnh
    img_spec = mapping[img]

    return img_spec

target_hist = np.zeros(256)
target_hist[231:256] = 1

image_transferred = transferGrayScale(image)    
image_transferred = bit_plane_slicing(image_transferred)
# drawHis(image_transferred)
# image_transferred = histogram_specification(image_transferred,target_hist=target_hist)
# drawHis(image_transferred)
# image_transferred = bit_plane_slicing(image_transferred)
# image_transferred= cv.equalizeHist(image_transferred)

showImage(image_transferred)