import numpy as np
import cv2 as cv
from HelperFunc import showImg
import os 

BASE_DIR = os.path.dirname(__file__)
image_path = os.path.join(BASE_DIR, "..", "..", "Images", "lena_std.tif")


image = cv.imread(image_path)
image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)



# Hàm làm nhiễu
def add_salt_pepper_noise(image, prob=0.02):
    noisy = image.copy()
    h, w = image.shape

    # Salt
    num_salt = int(prob * h * w / 2)
    coords = [np.random.randint(0, i, num_salt) for i in image.shape]
    noisy[coords[0], coords[1]] = 255

    # Pepper
    num_pepper = int(prob * h * w / 2)
    coords = [np.random.randint(0, i, num_pepper) for i in image.shape]
    noisy[coords[0], coords[1]] = 0

    return noisy

def add_gaussian_noise(image, mean=0, var=200):
    image = image.astype(np.float64)
    sigma = np.sqrt(var)
    noise = np.random.normal(mean, sigma, image.shape)
    noisy = image + noise
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)


# Lọc trung bình
def Mean_Filter(image, N):
    if N%2==0:
        return image

    h, w = image.shape
    pad = N // 2

    # Padding để không bị lỗi biên
    padded = np.pad(image, pad_width=pad, mode='edge')
    # print(padded.shape)
    # Ảnh output
    output = np.zeros_like(image,dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            window = padded[i:i+N, j:j+N]
            output[i, j] = np.mean(window)

    return output 

# Lọc trung vị
def Meadian_Filter(image,windowSize):
    if windowSize %2 == 0:
        return None
    h,w = image.shape
    pad = windowSize//2
    padded = np.pad(image, pad_width= pad, mode='edge')
    for i in range(h):
        for j in range(w):
            window = padded[i:i+windowSize,j:j+windowSize]
            image[i,j] = np.median(window)
    return image
            
# Lọc băng thông cao cơ bản
# def high_pass(image,Kernel):
#     h,w = Kernel.shape
#     if w %2 == 0:
#         print("Kernel is odd, Pls let kernel is even")
#         return None
#     pad = Kernel//2
#     padded = np.pad(image, pad_width= pad, mode='edge')
#     for i in range(h):
#         for j in range(w):
#             window = padded[i:i+pad,j:j+pad]
#             target = Kernel*window
#             image[i][j] = 




# image = add_salt_pepper_noise(image,prob= 0.2)
# image = Meadian_Filter(image,3)
# print(image.shape)
# print(image)
showImg(image)