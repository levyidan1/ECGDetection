import cv2
import numpy as np

image = cv2.imread('images/ecg_2.png')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
gray = clahe.apply(gray)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)


cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    area = cv2.contourArea(c)
    if area < 50:
        cv2.drawContours(opening, [c], -1, 0, -1)


result = 255 - opening
# result = cv2.GaussianBlur(result, (3,3), 0)
cv2.imshow('', image)
cv2.waitKey()
cv2.imshow('', result)
cv2.waitKey()
# cv2.imwrite('images/ecg_3_result.png', result)


equ_image = cv2.equalizeHist(gray)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(gray)

# res = np.hstack((gray, equ_image))

# cv2.imshow(segmented_image)
# cv2.waitKey()
# cv2.imshow('', equ_image)
# cv2.waitKey()
# cv2.imshow('', res)
# cv2.waitKey()

# cv2.imshow('', cl1)
# cv2.waitKey()