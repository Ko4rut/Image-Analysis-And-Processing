import cv2 as cv
import os

BASE_DIR = os.path.dirname(__file__)
image_path = os.path.join(BASE_DIR, "..", "..", "Images", "FishingBoat.jpg")
image = cv.imread(image_path)
aa= image[100:300,200:400,2]
b = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# cv.imshow("Original", a)
# cv.imshow("Gray", b)
cv.imshow("Sliced", aa)
cv.waitKey(0)
cv.destroyAllWindows()