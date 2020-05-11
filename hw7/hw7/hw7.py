import cv2
import numpy as np

BLACK = 1
WHITE = 0

def convert_for_imshow(binary_image):
    shape = binary_image.shape
    image = np.zeros((shape[0], shape[1], 3), np.uint8) # create completely black image
    for i in range(shape[0]):
        for j in range(shape[1]):
            image[i, j] = [0, 0, 0] if binary_image[i, j] == BLACK else [255, 255, 255]
    return image

def boo(a): return 1 if a else 0

class Yokoi_:

    Q = "q"
    R = "r"
    S = "s"

    def f(self, a, b, c, d):
        return 5 if a == b == c == d == self.R else (boo(a == self.Q) + boo(b == self.Q) + boo(c == self.Q) + boo(d == self.Q))

    def H(self, x, a, b, c):
        if x != a: return self.S
        if x == a == b == c: return self.R
        if x == a and (x != b or x != c): return self.Q
        assert 0

    def yokoi_value(self, image, i, j):
        rmax, cmax = image.shape[0], image.shape[1]
        def c(r, c):
            return image[r, c] if 0 <= r < rmax and 0 <= c < cmax else -1
        a1 = self.H(c(i,j), c(i,j+1), c(i-1,j+1), c(i-1,j))
        a2 = self.H(c(i,j), c(i-1,j), c(i-1,j-1), c(i,j-1))
        a3 = self.H(c(i,j), c(i,j-1), c(i+1,j-1), c(i+1,j))
        a4 = self.H(c(i,j), c(i+1,j), c(i+1,j+1), c(i,j+1))
        f_ = self.f(a1, a2, a3, a4)
        return f_

    def yokoi(self, sixfour):
        temp = sixfour.copy()
        h, w = sixfour.shape[0], sixfour.shape[1]
        for i in range(h):
            for j in range(w):
                if sixfour[i,j] == WHITE:
                    sixfour[i, j] = self.yokoi_value(temp, i, j)
                else:
                    sixfour[i, j] = 0
        return sixfour

Yokoi = Yokoi_()

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


def pair_relationship(yokoi_img): # yokoi_img is an array NUMBERS, not an image
    marked = set()
    r, c = yokoi_img.shape[0], yokoi_img.shape[1]
    def h(i, j):
        return 1 if 0 <= i < r and 0 <= j < c and yokoi_img[i, j] == 1 else 0
    for i in range(r):
        for j in range(c):
            if yokoi_img[i, j] > 0:
                if (h(i+1, j) + h(i-1, j) + h(i, j+1) + h(i, j-1) >= 1) and yokoi_img[i, j] == 1:
                    marked.add((i,j))
    return marked

def mark(image) -> set:
    yokoi_img = Yokoi.yokoi(image) # get yokoi connectivity matrix
    # print(yokoi_img)
    marked = pair_relationship(yokoi_img)
    return marked

class Shrink_:

    def h(self, x, a, b, c):
        return 1 if x == a == WHITE and (x != b or x != c) else 0

    def f(self, a1, a2, a3, a4, x0):
        return (boo(a1 == 1) + boo(a2 == 1) + boo(a3 == 1) + boo(a4 == 1))  == 1

    def shrink(self, image):
        rmax, cmax = image.shape[0], image.shape[1]
        def c(r, c):
            if 0 <= r < rmax and 0 <= c < cmax:
                if image[r,c] == BLACK: return BLACK
                elif image[r,c] == WHITE or image[r,c] == MARKED: return WHITE
                assert 0
            else: return -1

        for i in range(rmax):
            for j in range(cmax):
                if image[i, j] == MARKED:
                    a1 = self.h(c(i,j), c(i,j+1), c(i-1,j+1), c(i-1,j))
                    a2 = self.h(c(i,j), c(i-1,j), c(i-1,j-1), c(i,j-1))
                    a3 = self.h(c(i,j), c(i,j-1), c(i+1,j-1), c(i+1,j))
                    a4 = self.h(c(i,j), c(i+1,j), c(i+1,j+1), c(i,j+1))
                    if self.f(a1, a2, a3, a4, c(i,j)): # true, delete pixel (turn to black)
                        image[i, j] = BLACK
                    else: # else, keep it
                        image[i, j] = WHITE 
                    # print(f"at ({i}, {j}): a1 = {a1}, a2 = {a2}, a3 = {a3}, a4 = {a4}, set img to {image[i,j]}")

        return image


    def old_shrink(self, image) -> set:
        rmax, cmax = image.shape[0], image.shape[1]
        def c(r, c):
            return image[r, c] if 0 <= r < rmax and 0 <= c < cmax else -1

        ans = set()
        for i in range(rmax):
            for j in range(cmax):
                if image[i, j] == WHITE:
                    a1 = self.h(c(i,j), c(i,j+1), c(i-1,j+1), c(i-1,j))
                    a2 = self.h(c(i,j), c(i-1,j), c(i-1,j-1), c(i,j-1))
                    a3 = self.h(c(i,j), c(i,j-1), c(i+1,j-1), c(i+1,j))
                    a4 = self.h(c(i,j), c(i+1,j), c(i+1,j+1), c(i,j+1))
                    if self.f(a1, a2, a3, a4, image[i, j]): # true, delete pixel (turn to black)
                        image[i, j] = BLACK
                        ans.add((i,j))

        ## collect a set of all remaining (white) pixels?
        # points = np.argwhere(image == WHITE)
        # ans = set([tuple(x) for x in points]) 
        return ans

Shrink = Shrink_()

def delete_from_image(image, intersected):
    for point in intersected:
        image[point[0], point[1]] == BLACK 

MARKED = 87
def thin_once(image):
    new_img = image.copy()
    marked = mark(image) # return a set of marked pixels
    for p in marked: new_img[p[0], p[1]] = MARKED
    # print("marked:"); print(new_img)
    shrinked = Shrink.shrink(new_img) # return the shrinked image
    return shrinked

def is_identical(A, B):
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i,j] != B[i,j]: return False
    return True

def thin(image):
    old_img = image.copy()
    while True:
        new_img = thin_once(old_img.copy())
        if is_identical(old_img, new_img): break
        old_img = new_img.copy()
    return old_img

if __name__ == "__main__":
    image = cv2.imread("lena.bmp", cv2.IMREAD_GRAYSCALE)
    h, w = image.shape[0], image.shape[1]
    sixfour = np.zeros((64,64), dtype=np.short)

    for i in range(64):
        for j in range(64):
            sixfour[i, j] = image[i*8, j*8]
    
    sixfour = binarize(sixfour)
    result = thin(sixfour)

    # result = cv2.resize(result, (512, 512))

    cv2.imwrite("meow.bmp", convert_for_imshow(result))

    large = np.zeros((512, 512))
    for i in range(512):
        for j in range(512):
            large[i,j] = result[int(i/8),int(j/8)]


    cv2.imwrite("meow_large.bmp", convert_for_imshow(large))

    # ans = thin(sixfour)

    # points = np.array([
    #     [0,0], [0,1],
    #     [1,1], [1,2],
    #     [2,2], [2,3],
    #     [3,2], [3,3],
    #     [4,2], [4,3]
    # ])
    # A = np.full((5,5), BLACK)
    # for p in points: A[p[0], p[1]] = WHITE
    # print(thin(A))

    # print(A)
    # testest = mark(A.copy())
    # print(testest)

    # print("----")
    # tettt = Shrink.shrink(A.copy())
    # print(tettt)

    # print("----")
    # result = testest & tettt
    # A = np.full((5,5), BLACK)
    # for p in result: A[p[0], p[1]] = WHITE
    # print(A)

    # print("---test shrinking---")

    # b = BLACK
    # w = WHITE
    # A = np.array([
    #     [b, b, w, w, b, b],
    #     [b, w, w, w, w, b],
    #     [w, w, w, w, w, w],
    #     [b, w, w, b, w, w],
    #     [b, b, w, b, w, w],
    #     [b, b, w, b, w, w],
    #     [b, b, w, b, w, w]
    # ])

    # ew = Shrink.shrink(A)
    # print(ew)



    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
