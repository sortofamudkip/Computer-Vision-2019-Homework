import cv2
import numpy as np 	
import math
from time import time

def roberty(img, threshold: int=30):
	extended = cv2.copyMakeBorder(img, 0, 1, 0, 1, cv2.BORDER_REPLICATE).astype(np.short)
	new_img = img.copy(); h, w = new_img.shape[:2]
	r1_meow = np.array([[-1, 0], [0, 1]])
	r2_meow = np.array([[0, -1], [1, 0]])
	for i in range(h):
		for j in range(w):
			r1 = np.sum(extended[i:i+2,j:j+2] * r1_meow)
			r2 = np.sum(extended[i:i+2,j:j+2] * r2_meow)
			new_img[i,j] = 0 if math.sqrt(r1**2 + r2**2) >= threshold else 255
	return new_img


def preuwuy(img, threshold: int=24):
	extended = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REPLICATE).astype(np.short)
	new_img = img.copy(); h, w = new_img.shape[:2]
	p1_meow = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
	p2_meow = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
	for i in range(h):
		for j in range(w):
			n, m = i+1, j+1
			p1 = np.sum(extended[n-1:n+2,m-1:m+2] * p1_meow)
			p2 = np.sum(extended[n-1:n+2,m-1:m+2] * p2_meow)
			new_img[i,j] = 0 if math.sqrt(p1**2 + p2**2) >= threshold else 255
	return new_img

def sobbbbby(img, threshold: int=38):
	extended = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REPLICATE).astype(np.short)
	new_img = img.copy(); h, w = new_img.shape[:2]
	s1_meow = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
	s2_meow = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
	for i in range(h):
		for j in range(w):
			n, m = i+1, j+1
			s1 = np.sum(extended[n-1:n+2,m-1:m+2] * s1_meow)
			s2 = np.sum(extended[n-1:n+2,m-1:m+2] * s2_meow)
			new_img[i,j] = 0 if math.sqrt(s1**2 + s2**2) >= threshold else 255
	return new_img

def freddychan(img, threshold: int=30):
	extended = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REPLICATE).astype(np.short)
	new_img = img.copy(); h, w = new_img.shape[:2]
	f1_meow = np.array([[-1, -math.sqrt(2), -1], [0, 0, 0], [1, math.sqrt(2), 1]])
	f2_meow = np.array([[-1, 0, 1], [-math.sqrt(2), 0, math.sqrt(2)], [-1, 0, 1]])
	for i in range(h):
		for j in range(w):
			n, m = i+1, j+1
			f1 = np.sum(extended[n-1:n+2,m-1:m+2] * f1_meow)
			f2 = np.sum(extended[n-1:n+2,m-1:m+2] * f2_meow)
			new_img[i,j] = 0 if math.sqrt(f1**2 + f2**2) >= threshold else 255
	return new_img


def generate_kmasks(amount: int):
	t = np.roll(np.array([-3, -3, -3, -3, 5, 5, 5, -3]), amount)
	return np.array([[t[0], t[7], t[6]], [t[1], 0, t[5]], [t[2], t[3], t[4]]])

def kirschhhhh(img, threshold=135):
	k_masks = [generate_kmasks(k) for k in range(8)]
	extended = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REPLICATE).astype(np.short)
	new_img = img.copy(); h, w = new_img.shape[:2]
	for i in range(h):
		for j in range(w):
			n, m = i+1, j+1
			new_img[i,j] = 0 if max([np.sum(extended[n-1:n+2,m-1:m+2] * k) for k in k_masks]) >= threshold else 255
	return new_img

def generate_rmasks(amount: int):
	t = np.roll(np.array([-1, -2, -1, 0, 1, 2, 1, 0]), amount)
	return np.array([[t[0], t[7], t[6]], [t[1], 0, t[5]], [t[2], t[3], t[4]]])

def robinhood(img, threshold=43):
	k_masks = [generate_rmasks(k) for k in range(8)]
	extended = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REPLICATE).astype(np.short)
	new_img = img.copy(); h, w = new_img.shape[:2]
	for i in range(h):
		for j in range(w):
			n, m = i+1, j+1
			new_img[i,j] = 0 if max([np.sum(extended[n-1:n+2,m-1:m+2] * k) for k in k_masks]) >= threshold else 255
	return new_img

def nevatiababayetu(img, threshold=12500):
	k_masks = [
		np.array([[100] * 5, [100] * 5, [0] * 5, [-100] * 5, [-100] * 5]),
		np.array([[100] * 5, [100] * 3 + [78, -32], [100, 92, 0, -92, -100], [32, -78] + [-100] * 3, [-100] * 5]),
		np.array([[100]*3+[32,-100], 
					[100, 100, 92, -78, -100], 
					[100, 100, 0, -100, -100], 
					[100, 78, -92, -100, -100],
					[100, -32, -100, -100, -100]]),
		np.array([[-100, -100, 0, 100, 100]]*5),
		np.array([[-100, 32, 100, 100, 100],
					[-100, -78, 92, 100, 100],
					[-100, -100, 0, 100, 100],
					[-100, -100, -92, 78, 100],
					[-100]*3 + [-32,100]]),
		np.array([[100]*5,
					[-32, 78] + [100]*3,
					[-100, -92, 0, 92, 100],
					[-100] * 3 + [-78, 32],
					[-100] * 5])]
	extended = cv2.copyMakeBorder(img, 2, 2, 2, 2, cv2.BORDER_REPLICATE).astype(np.short)
	new_img = img.copy(); h, w = new_img.shape[:2]
	for i in range(h):
		for j in range(w):
			n, m = i+2, j+2
			new_img[i,j] = 0 if max([np.sum(extended[n-2:n+3,m-2:m+3] * k) for k in k_masks]) >= threshold else 255
	return new_img

if __name__ == '__main__':
	lena = cv2.imread("lena.bmp", cv2.IMREAD_GRAYSCALE)
	detectors = [roberty, preuwuy, sobbbbby, freddychan, kirschhhhh, robinhood, nevatiababayetu]
	names =     ["robert", "prewitt", "sobel", "frei", "kirsch", "robinson", "nevatiababu"]
	before = time()
	ans_images = [f(lena) for f in detectors]
	for name, img in zip(names, ans_images):
		# save it with a name
		cv2.imwrite(name+".bmp", img)
	# test = np.array([
	# 	[169, 146, 153, 145, 137, 151, 112, 98],
	# 	[104, 104, 97, 100, 115, 40, 42, 63],
	# 	[130, 120, 95, 120, 130, 212, 115, 128],
	# 	[124, 157, 162, 45, 87, 77, 75, 101],
	# 	[124, 201, 177, 176, 136, 113, 150, 137],
	# 	[162, 155, 193, 46, 52, 87, 126, 203],
	# 	[141, 149, 38, 54, 155, 145, 132, 57],
	# 	[87, 64, 156, 161, 180, 210, 99, 79],
	# ])

	# print(nevatiababayetu(test))
	print("time:", time() - before)

	cv2.waitKey(0)
	cv2.destroyAllWindows()
