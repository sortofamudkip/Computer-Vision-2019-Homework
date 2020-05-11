import cv2

image = cv2.imread("lena.bmp")

flipped_image    = image[::-1] # to flip it on its head

right_left_image = image.copy()
for i in range(image.shape[0]):
    right_left_image[i] = right_left_image[i][::-1]

diag_image = image.copy()

for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        diag_image[i][j] = image[j][i]



cv2.imshow("orignal meow", image)
cv2.imshow("flipped meow", flipped_image)
cv2.imshow("right left meow", right_left_image)
cv2.imshow("diag image meow", diag_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("flipped.bmp", flipped_image)
cv2.imwrite("reversed.bmp", right_left_image)
cv2.imwrite("diag.bmp", diag_image)
