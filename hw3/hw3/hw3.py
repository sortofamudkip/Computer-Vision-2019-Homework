import cv2
import matplotlib.pyplot as plt
import numpy as np


image = cv2.imread("lena.bmp")

## A, get histogram

def binarize(image):
    binary_image = image.copy()
    shape = binary_image.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            if image[i, j][0] < 128:
                binary_image[i, j] = [0, 0, 0]
            else: 
                binary_image[i, j] = [255, 255, 255]
    return binary_image

def get_frequency(image):
    shape = image.shape
    y = [0] * 256
    for i in range(shape[0]):
        for j in range(shape[1]):
            y[int(image[i, j][0])] += 1
    return y

def histogramize(image, frequency, title="", file_title="meow.jpg"):
    x = np.arange(256)
    y = frequency
    fig, ax = plt.subplots()
    ax.set_title(title)
    plt.bar(x, np.array(y))
    plt.savefig(file_title)

y_original = get_frequency(image)
histogramize(image, y_original, "B05902100 - HW3 part 1", "part1.png")

## B. divide all by 3

three = image.copy()
for i in range(three.shape[0]):
    for j in range(three.shape[1]):
        three[i, j] = [int(three[i, j][0] / 3), int(three[i, j][1] / 3), int(three[i, j][2] / 3)]

y_three = get_frequency(three)
cv2.imwrite("part2.bmp", three)
histogramize(three, y_three, "B05902100 - HW3 part 2", "part2.png")


## C. histogram equaliztion
S_k = [0] * 256
n = three.shape[0] * three.shape[1]
n_cumulative = np.cumsum(y_three)

imhe = three.copy()
for i, s in enumerate(S_k):
    S_k[i] = 255/n * n_cumulative[i]

for i in range(three.shape[0]):
    for j in range(three.shape[1]):
        imhe[i, j] = [int(S_k[imhe[i, j][0]])] * 3

histy_y = get_frequency(imhe)
cv2.imwrite("part3.bmp", imhe)
histogramize(imhe, histy_y, "B05902100 - HW3 part 3", "part3.png")



cv2.waitKey(0)
cv2.destroyAllWindows()
