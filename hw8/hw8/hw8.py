import numpy as np
import cv2

def salt_pepper_noise(img, noise=0.05):
    new = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            u = np.random.uniform()
            if u < noise: new[i, j] = 0
            elif u > 1 - noise: new[i, j] = 255
            else: new[i, j] = img[i, j]
    return new

def gaussian_noise(img, amp=10):
    new = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            u = img[i, j] + amp * np.random.normal()
            u = min(255, max(0, u))
            new[i, j] = int(u)
    return new

def get_neighbours(I, i, j, r=1):
    ans = [I[m, n] for m in range(i-r, i+r+1) for n in range(j-r, j+r+1) ]# if (m, n) != (0, 0)
    return ans

def med(radius):
    def medddddd(img): # returns the median 3x3 image.
        ans = img.copy()
        I = cv2.copyMakeBorder(ans, radius, radius, radius, radius, cv2.BORDER_REFLECT) 
        newr, newc = I.shape[0], I.shape[1]
        for i in range(radius, newr-radius):
            for j in range(radius, newc-radius):
                neighbours = sorted(get_neighbours(I, i, j, r=radius))
                ans[i - radius, j - radius] = neighbours[int(len(neighbours)/2)] # -1 because index starts from 0
        return ans
    return medddddd

def box(radius):
    def boxxxx(img):
        ans = img.copy()
        I = cv2.copyMakeBorder(ans, radius, radius, radius, radius, cv2.BORDER_REFLECT)
        newr, newc = I.shape[0], I.shape[1]
        for i in range(radius, newr-radius):
            for j in range(radius, newc-radius):
                ans[i - radius, j - radius] = np.sum(I[i-radius:i+radius+1, j-radius:j+radius+1]) / ((2*radius+1)**2)
        return ans
    return boxxxx

def SNR(noise, original, VS):
    area = original.shape[0] * original.shape[1]
    correct = original/ 255
    noisy_img = noise / 255
    mu_N = np.sum(noisy_img-correct) / area
    VN = np.sum((noisy_img - correct - mu_N)**2)/area
    ans = 20 * np.log10( np.sqrt(VS) / np.sqrt(VN) )
    return ans

def get_VS(image):
    area = image.shape[0] * image.shape[1]
    correct = image/255
    mu = np.sum(correct) / area
    VS = np.sum((correct - mu)**2) / area
    return VS

def complete_SMR(noise, original):
    area = original.shape[0] * original.shape[1]
    correct = original/255
    mu = np.sum(correct) / area
    VS = np.sum((correct - mu)**2) / area

    noisy_img = noise / 255
    mu_N = np.sum(noisy_img - correct) / area
    VN = np.sum((noisy_img - correct - mu_N)**2)/area
    ans = 20 * np.log10( np.sqrt(VS) / np.sqrt(VN) )
    return ans

octagon = np.array([
    [0, 1, 1, 1, 0],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [0, 1, 1, 1, 0],
    ], dtype=np.uint8)
kernel = np.argwhere(octagon == 1) - 2 # get all points 

# change to use border extension
def operate(img, f):
    new_img = img.copy()
    h, w = img.shape[0], img.shape[1]
    for i in range(h):
        for j in range(w):
            # for each pixel, collect all the pixels created by it and the kernel
            pixel_kernel = [ (i+k[0], j+k[1]) for k in kernel if 0 <= i+k[0] < h and 0 <= j+k[1] < w]
            neighbours = [img[p[0], p[1]] for p in pixel_kernel]
            new_img[i, j] = f(neighbours) 
            # if len(neighbours) - 1 != len(kernel): print(f"at {(i,j)} from {neighbours}, chose {new_img[i,j]}")
    return new_img


def dilate(img): return operate(img, max)

def erode(img): return operate(img, min)

def open(img): 
    return dilate(erode(img.copy()))

def close(img):
    return erode(dilate(img.copy()))

def openclose(img):
    return close(open(img.copy()))

def closeopen(img):
    return open(close(img.copy()))

if __name__ == "__main__":

    np.random.seed(55)
    image = cv2.imread("lena.bmp", cv2.IMREAD_GRAYSCALE)
    VS = get_VS(image)

    images = {
        "gauss10" : gaussian_noise(image, 10),
        "gauss30": gaussian_noise(image, 30),
        "salt005": salt_pepper_noise(image, 0.05),
        "salt010": salt_pepper_noise(image, 0.1)
    }

    filters = {
        # "box3" : box(1),
        # "box5" : box(2),
        # "med3" : med(1),
        # "med5" : med(2),
        "openclose": openclose,
        "closeopen": closeopen
    }

    # print the SNR for each image
    original_image = image
    for image_name, noisy_image in images.items():
        print("{} SNR: {}".format(image_name + ".bmp", SNR(noisy_image, original_image, VS)))
        cv2.imwrite(image_name + ".bmp", noisy_image)

    for image_name, noisy_image in images.items():
        for fname, f in filters.items():
            name = "{}_{}.bmp".format(image_name, fname)
            filtered = f(noisy_image)
            print("{} SNR: {}".format(name, SNR(filtered, original_image, VS)))
            cv2.imwrite(name, filtered)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
