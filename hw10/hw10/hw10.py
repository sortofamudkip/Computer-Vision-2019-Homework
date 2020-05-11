import cv2
import numpy as np  
import math

def laplace(img, mask, threshold: int):
    def has_neg_1(gradient, i, j):
        neighbours = np.zeros((3, 3))
        for delta_i in range(-1, 1+1):
            for delta_j in range(-1, 1+1):
                neighbours[1+delta_i, 1+delta_j] = gradient[i+delta_i, j+delta_j] if (delta_i, delta_j) != (0,0) else 87
        return np.any(neighbours == -1)

    extended = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REPLICATE).astype(np.float)
    gradient = img.copy().astype(np.float); new_img = img.copy(); h, w = new_img.shape[:2]
    for i in range(h):
        for j in range(w):
            n, m = i+1, j+1
            total = np.sum(extended[n-1:n+2,m-1:m+2] * mask)
            gradient[i,j] = 1 if total >= threshold else -1 if total <= -threshold else 0
    gradient = cv2.copyMakeBorder(gradient, 1, 1, 1, 1, cv2.BORDER_REPLICATE) 
    for i in range(h):
        for j in range(w):
            n, m = i+1, j+1
            new_img[i,j] = 0 if gradient[n,m] == 1 and has_neg_1(gradient, n, m) else 255
    return new_img

def biglaplace(img, mask, threshold: int):
    def has_neg_1(gradient, i, j):
        neighbours = np.zeros((3, 3))
        for delta_i in range(-1, 1+1):
            for delta_j in range(-1, 1+1):
                neighbours[1+delta_i, 1+delta_j] = gradient[i+delta_i, j+delta_j] if (delta_i, delta_j) != (0,0) else 87
        return np.any(neighbours == -1)

    extended = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_REPLICATE).astype(np.float)
    gradient = img.copy().astype(np.float); new_img = img.copy(); h, w = new_img.shape[:2]
    for i in range(h):
        for j in range(w):
            n, m = i+5, j+5
            total = np.sum(extended[n-5:n+6,m-5:m+6] * mask)
            gradient[i,j] = 1 if total >= threshold else -1 if total <= -threshold else 0
    gradient = cv2.copyMakeBorder(gradient, 1, 1, 1, 1, cv2.BORDER_REPLICATE) 
    for i in range(h):
        for j in range(w):
            n, m = i+1, j+1
            new_img[i,j] = 0 if gradient[n,m] == 1 and has_neg_1(gradient, n, m) else 255
    return new_img


if __name__ == '__main__':
    lena = cv2.imread("lena.bmp", cv2.IMREAD_GRAYSCALE)

    stuffs = [
        {
            "name": "laplace1.bmp",
            "detector": laplace,
            "mask": np.array([[0, 1, 0],
                             [1, -4, 1],
                             [0, 1, 0]]),
            "threshold": 15
        },
        {
            "name": "laplace2.bmp",
            "detector": laplace,
            "mask": np.array([[1, 1, 1],
                              [1, -8, 1],
                              [1, 1, 1]]) / 3,
            "threshold": 15
        },
        {
            "name": "minvarLaplace.bmp",
            "detector": laplace,
            "mask": np.array([[2, -1, 2],
                              [-1, -4, -1],
                              [2, -1, 2]]) / 3,
            "threshold": 20
        },  
        {
            "name": "LaplaceGaussian.bmp",
            "detector": biglaplace,
            "mask": np.array(
               [[0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0],
                [0, 0,-2, -4, -8, -9, -8, -4,-2, 0, 0],
                [0,-2,-7,-15,-22,-23,-22,-15,-7,-2, 0],
                [-1,-4,-15,-24,-14,-1,-14,-24,-15,-4,-1],
                [-1,-8,-22,-14,52,103,52,-14,-22,-8,-1],
                [-2,-9,-23,-1,103,178,103,-1,-23,-9,-2],
                [-1,-8,-22,-14,52,103,52,-14,-22,-8,-1],
                [-1,-4,-15,-24,-14,-1,-14,-24,-15,-4,-1],
                [0,-2,-7,-15,-22,-23,-22,-15,-7,-2, 0],
                [0, 0,-2, -4, -8, -9, -8, -4,-2, 0, 0],
                [0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0]]
                ),
            "threshold": 3000
        },
        {
            "name": "DiffOfGaussian.bmp",
            "detector": biglaplace,
            "mask": np.array(
               [[-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1],
                [-3, -5, -8,-11,-13,-13,-13,-11, -8, -5, -3],
                [-4, -8,-12,-16,-17,-17,-17,-16,-12, -8, -4],
                [-6,-11, -16,-16,0,  15,  0,-16,-16,-11, -6],
                [-7,-13,-17,  0, 85,160, 85,  0,-17,-13, -7],
                [-8,-13,-17, 15,160,283,160, 15,-17,-13, -8],
                [-7,-13,-17,  0, 85,160, 85,  0,-17,-13, -7],
                [-6,-11, -16,-16,0,  15,  0,-16,-16,-11, -6],
                [-4, -8,-12,-16,-17,-17,-17,-16,-12, -8, -4],
                [-3, -5, -8,-11,-13,-13,-13,-11, -8, -5, -3],
                [-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1]]),
            "threshold": 1
        },
    ]

    for stuff in stuffs:
        image = stuff["detector"](lena.copy(), stuff["mask"], stuff["threshold"])
        cv2.imwrite(stuff["name"], image)
        print("finished writing", stuff["name"])


    # test = np.array([[153,166,85,90,96,102,101,98,73,64,55,40,172,163,155],
    #               [161,164,161,166,171,173,174,176,180,178,183,174,180,174,180],
    #               [103,105,97,97,97,103,101,98,111,111,104,48,54,49,57],
    #               [126,123,117,115,145,86,113,146,96,73,110,98,43,53,95],
    #               [135,129,110,114,107,113,114,66,84,34,120,130,73,51,79],
    #               [134,128,130,134,139,129,94,39,45,197,48,43,41,57,46],
    #               [123,167,142,183,168,110,45,141,61,85,81,61,57,118,127],
    #               [130,131,170,194,170,69,171,100,173,150,136,128,134,142,137],
    #               [133,127,211,201,172,127,181,115,141,131,148,150,151,147,140],
    #               [108,110,95,227,172,192,203,169,163,149,42,51,189,171,159],
    #               [163,149,146,150,203,127,69,37,49,64,85,78,98,215,202],
    #               [155,152,144,209,182,205,66,46,45,147,156,112,146,125,101],
    #               [209,143,149,46,52,55,134,159,157,150,143,136,122,148,109],
    #               [123,93,54,55,144,156,164,155,145,188,215,207,209,98,91],
    #               [96,51,67,163,162,154,157,145,191,210,213,48,91,85,89]])

    # mask = np.array([[0, 1, 0],
    #                [1, -4, 1],
    #                [0, 1, 0]])
    # meow = laplace(test, mask, 15)


    cv2.waitKey(0)
    cv2.destroyAllWindows()
