import cv2
import numpy as np

BLACK = 0
WHITE = 1

## get histogram

def binarize(image):
    binary_image = image.copy()
    shape = binary_image.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            if image[i, j] < 128:
                binary_image[i, j] = 0
            else: 
                binary_image[i, j] = 1
    return binary_image

def convert_for_imshow(binary_image):
    shape = binary_image.shape
    image = np.zeros((shape[0], shape[1], 3), np.uint8) # create completely black image
    for i in range(shape[0]):
        for j in range(shape[1]):
            image[i, j] = [0, 0, 0] if not binary_image[i, j] else [255, 255, 255]
    return image

def fill_img(height, width, val, points):
    image = np.full((height, width), val, dtype=np.uint8)
    for point in points:
        image[point[0], point[1]] = WHITE
    return image

def dilate(image, kernel):
    A = np.argwhere(image == WHITE)
    A_dil_B = set()
    for a in A:
        for b in kernel:
            new_point = a + b
            if 0 <= new_point[0] < image.shape[0] and 0 <= new_point[1] < image.shape[1]:
                A_dil_B.add((new_point[0], new_point[1]))

    new_image = fill_img(image.shape[0], image.shape[1], 0, A_dil_B)
    return new_image

def erode(image, kernel):
    A = np.argwhere(image == WHITE)
    A = set(tuple(map(tuple, A)))
    A_erode_B = set()

    for r in range(-5, 513):
        for c in range(-5, 513):
            good = True
            for b in kernel:
                if (b[0]+r, b[1]+c) not in A: good = False; break
            if good: A_erode_B.add((r,c))

    new_image = fill_img(image.shape[0], image.shape[1], 0, A_erode_B)
    return new_image


def imshow_binary(name, image):
    cv2.imshow(name, convert_for_imshow(image))

def write_binary(name, image):
    cv2.imwrite(name, convert_for_imshow(image))


if __name__ == "__main__":
    image = cv2.imread("lena.bmp", cv2.IMREAD_GRAYSCALE)
    binary_image = binarize(image)
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


    ## A. Dilated

    dilated_image = dilate(binary_image, oct_kernel)
    write_binary("dilated.bmp", dilated_image)

    ## B. Erosion

    eroded_image = erode(binary_image, oct_kernel)
    write_binary("eroded.bmp", eroded_image)

    ## C. Opening 

    opened_image = dilate(eroded_image, oct_kernel)
    write_binary("opened.bmp", opened_image)

    ## D. Closing

    closed_image = erode(dilated_image, oct_kernel)
    write_binary("closed.bmp", closed_image)

    ## E. Hit-or-miss
    J = [(0,0), (0,-1), (1,0)]
    K = [(-1,0), (-1,1), (0,1)]
    A_c = 1 - binary_image

    A_erode_J = erode(binary_image, J)
    A_erode_K = erode(A_c, K)
    hit_miss_image = binary_image.copy()
    for i in range(binary_image.shape[0]):
        for j in range(binary_image.shape[1]):
            hit_miss_image[i, j] = 1 if A_erode_J[i, j] and A_erode_K[i, j] else 0
    write_binary("hit_or_miss.bmp", hit_miss_image)

    print("done")