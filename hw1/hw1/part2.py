import cv2

image = cv2.imread("lena.bmp")

cv2.imshow("orignal meow", image)

# rotate 45 degrees clockwise
height, width = image.shape[0], image.shape[1]
center = (height/2, width/2)

M = cv2.getRotationMatrix2D(center, -45, 1.0)
rotated45 = cv2.warpAffine(image, M, (height, width))

smol = cv2.resize(image,None,fx=0.5,fy=0.5)

ret, bw_img = cv2.threshold(image,128,255,cv2.THRESH_BINARY)

cv2.imshow("smol meow meow", smol)
cv2.imshow("45 meow meow", rotated45)
cv2.imshow("binary meow meow", bw_img)

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("fortyfive.bmp", rotated45)
cv2.imwrite("smol.bmp", smol)
cv2.imwrite("binary.bmp", bw_img)
