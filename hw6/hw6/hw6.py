import cv2
import numpy as np

Q = "q"
R = "r"
S = "s"
BLACK = 1
WHITE = 0


def convert_for_imshow(binary_image):
    shape = binary_image.shape
    image = np.zeros((shape[0], shape[1], 3), np.uint8) # create completely black image
    for i in range(shape[0]):
        for j in range(shape[1]):
            image[i, j] = [binary_image[i, j]] * 3
    return image

def boo(a): return 1 if a else 0

def f(a, b, c, d):
    return 5 if a == b == c == d == R else (boo(a == Q) + boo(b == Q) + boo(c == Q) + boo(d == Q))

def H(x, a, b, c):
    if x != a: return S
    if x == a == b == c: return R
    if x == a and (x != b or x != c): return Q
    assert 0

def yokoi(image, i, j):
    rmax, cmax = image.shape[0], image.shape[1]
    def c(r, c):
        return image[r, c] if 0 <= r < rmax and 0 <= c < cmax else -1
    a1 = H(c(i,j), c(i,j+1), c(i-1,j+1), c(i-1,j))
    a2 = H(c(i,j), c(i-1,j), c(i-1,j-1), c(i,j-1))
    a3 = H(c(i,j), c(i,j-1), c(i+1,j-1), c(i+1,j))
    a4 = H(c(i,j), c(i+1,j), c(i+1,j+1), c(i,j+1))
    f_ = f(a1, a2, a3, a4)
    return f_


def binarize(image):
    binary_image = image.copy()
    shape = binary_image.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            if image[i, j] < 128:
                binary_image[i, j] = BLACK
            else: 
                binary_image[i, j] = WHITE
    return binary_image

if __name__ == "__main__":
    image = cv2.imread("lena.bmp", cv2.IMREAD_GRAYSCALE)
    h, w = image.shape[0], image.shape[1]
    sixfour = np.zeros((64,64), dtype=np.short)

    for i in range(64):
        for j in range(64):
            sixfour[i, j] = image[i*8, j*8]
    

    # exit()
    # cv2.imshow("sixfour", convert_for_imshow(sixfour))
    sixfour = binarize(sixfour)
    # for i in range(sixfour.shape[0]):
    #     for j in range(sixfour.shape[1]):
    #         print("{}".format(sixfour[i,j]), end=" ")
    #     print()
    # print("========")

    temp = sixfour.copy()

    h, w = sixfour.shape[0], sixfour.shape[1]
    for i in range(h):
        for j in range(w):
            if sixfour[i,j] == WHITE:
                sixfour[i, j] = yokoi(temp, i, j)
            else:
                sixfour[i, j] = 0

    for i in range(sixfour.shape[0]):
        for j in range(sixfour.shape[1]):
            print("{}".format(sixfour[i,j]) if sixfour[i,j] else " ", end="")
        print()

    cv2.waitKey(0)
    cv2.destroyAllWindows()
