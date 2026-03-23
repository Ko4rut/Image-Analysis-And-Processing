import numpy as np
import cv2 as cv
from HelperFunc import showImg
import os 

BASE_DIR = os.path.dirname(__file__)
image_path = os.path.join(BASE_DIR, "..", "..", "Images", "Head_Film.png")


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


# Lọc trung bình (lowpass)
def Mean_Filter(image, N):
    if N%2==0:
        return image

    h, w = image.shape
    pad = N // 2
    kernel = (1/N**2)*np.ones((N,N))
    # Padding để không bị lỗi biên
    padded = np.pad(image, pad_width=pad, mode='edge')
    # print(padded.shape)
    # Ảnh output
    # kernel = np.zeros(N,dtype=)
    output = np.zeros_like(image,dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            window = padded[i:i+N, j:j+N]
            output[i, j] = np.sum(window * kernel)
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
def high_pass(image,Kernel):
    Kernel = Kernel.astype(np.float32)
    h,w = Kernel.shape
    if w %2 == 0:
        print("Kernel size must be odd")
        return None
    pad = w//2
    padded = np.pad(image, pad_width= pad, mode='edge')
    output = np.zeros_like(image, dtype=np.float32)
    H,W= image.shape
    for i in range(H):
        for j in range(W):
            window = padded[i:i+h,j:j+w]
            target = np.sum(window*Kernel)
            output[i][j] = target
    return np.clip(output, 0, 255).astype(np.uint8)

def high_boost(img,A):
    img = image.astype(np.float32)
    blur = Mean_Filter(image, 17).astype(np.float32)
    high_boost = A * img - blur
    high_boost = np.clip(high_boost, 0, 255).astype(np.uint8)
    return high_boost


#Lọc theo đạo hàm
def gradien_filter(img, kernel):
    img = img.astype(np.float32)
    output = np.zeros_like(img, dtype=np.float32)
    h,w = kernel.shape
    padding = w//2
    padded = np.pad(img, padding,mode="edge")
    h_img, w_img = img.shape
    for i in range(h_img):
        for j in range(w_img):
            window = padded[i:i+h, j:j+w]
            output[i,j]=np.sum(window*kernel)
    return np.clip(output,0,255).astype(np.uint8)

    
 # thử lọc theo Prewitt

def prewitt_filter(img):
    img = img.astype(np.float32)

    kx = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ], dtype=np.float32)

    ky = np.array([
        [-1, -1, -1],
        [ 0,  0,  0],
        [ 1,  1,  1]
    ], dtype=np.float32)

    pad = 1
    padded = np.pad(img, pad, mode='edge')

    H, W = img.shape
    output = np.zeros((H, W), dtype=np.float32)

    for i in range(H):
        for j in range(W):
            window = padded[i:i+3, j:j+3]
            Gx = np.sum(window * kx)
            Gy = np.sum(window * ky)
            output[i, j] = np.sqrt(Gx*Gx + Gy*Gy)

    return np.clip(output, 0, 255).astype(np.uint8)
   
# image = Mean_Filter(image,7)
# image = add_salt_pepper_noise(image,prob= 0.2)
# image = Meadian_Filter(image,3)
# kernel = 4*np.array([[-1,-1,-1],
#                   [-1,8,-1],
#                   [-1,-1,-1]])
# image = high_pass(image,kernel)
## Sài công thức như thầy vẫn ổn
# image = image - Mean_Filter(image,3)
# image = np.clip(image,0,255).astype(np.uint8)

## High-boost theo công thức slide 
image = high_boost(image,3)


# kx = np.array([
#     [-1, 1]
# ])

# ky = np.array([
#     [-1],
#     [ 1]
# ])

# kx = np.array([
#     [-1, 0, 1],
#     [-2, 0, 2],
#     [-1, 0, 1]
# ])
# image = gradien_filter(image,kx)
# image = prewitt_filter(image)
# print(image.shape)
# print(image)
showImg(image)