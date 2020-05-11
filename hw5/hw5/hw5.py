import cv2
import numpy as np
from itertools import product

BLACK = 0
WHITE = 1

def operate(img, kernel, f):
    new_img = img.copy()
    h, w = img.shape[0], img.shape[1]
    height, width = range(img.shape[0]), range(img.shape[1])
    coords = product(height, width)
    for ij in coords:
        i, j = ij
        # for each pixel, collect all the pixels created by it and the kernel
        pixel_kernel = [ (i+k[0], j+k[1]) for k in kernel if 0 <= i+k[0] < h and 0 <= j+k[1] < w]
        new_img[i, j] = f([ img[p[0], p[1]] for p in pixel_kernel]) if pixel_kernel else new_img[i, j]
    return new_img

def dilate(img, kernel):
    return operate(img, kernel, max)

def erode(img, kernel):
    return operate(img, kernel, min)

if __name__ == "__main__":
    image = cv2.imread("lena.bmp", cv2.IMREAD_GRAYSCALE)
    # imshow_binary("binary", binary_image)

    ## create kernel for A B C D
    octagon = np.array([
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        ], dtype=np.uint8)
    oct_kernel = np.argwhere(octagon == WHITE) # get all points 
    # oct_kernel = set(tuple(map(tuple, oct_kernel))) # turn it into a set 

    ## A. Dilated

    dilated_image = dilate(image, oct_kernel)
    cv2.imwrite("dilated.bmp", dilated_image)

    # ## B. Erosion

    eroded_image = erode(image, oct_kernel)
    cv2.imwrite("eroded.bmp", eroded_image)

    # ## C. Opening 

    opened_image = dilate(eroded_image, oct_kernel)
    cv2.imwrite("opened.bmp", opened_image)

    # ## D. Closing

    closed_image = erode(dilated_image, oct_kernel)
    cv2.imwrite("closed.bmp", closed_image)
