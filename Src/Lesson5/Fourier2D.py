# import numpy as np
# import cv2 as cv
# from HelperFunc import showImg
# import os 

# BASE_DIR = os.path.dirname(__file__)
# image_path = os.path.join(BASE_DIR, "..", "..", "Images", "Head_Film.png")


# image = cv.imread(image_path)
# image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# def dft_2d(img):
#     img = np.asarray(img, dtype=np.complex128)
#     w, h = img.shape
#     X = np.zeros((w, h), dtype=np.complex128)
#     count = 0
#     for i in range(w):          # tần số theo x
#         for j in range(h):      # tần số theo y
#             s = 0j
#             for x in range(w):  # chạy qua pixel x
#                 for y in range(h):  # chạy qua pixel y
#                     s += img[x, y] * np.exp(
#                         -1j * 2*np.pi * ((i*x)/w + (j*y)/h)
#                     )
#                     count +=1
#                     print(count)
#             X[i, j] = s
#     return X
  
# # print(dft_2d(image).shape)

# F = np.fft.fft2(image)          # Fourier 2D
# F_shift = np.fft.fftshift(F)    # đưa DC về giữa (để nhìn đẹp)
# mag = np.abs(F_shift)           # độ lớn phổ
# print(mag)

