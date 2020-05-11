import cv2
import matplotlib.pyplot as plt
import numpy as np


image = cv2.imread("lena.bmp")

binary_image = image.copy()

shape = binary_image.shape


## A, binarize
for i in range(shape[0]):
    for j in range(shape[1]):
        if image[i, j][0] < 128:
            binary_image[i, j] = [0, 0, 0]
        else: 
            binary_image[i, j] = [255, 255, 255]

print("finished binarizing")
# cv2.imshow("binarized", binary_image)



## B, histogram
x = np.arange(256)
y = [0] * 256

for i in range(shape[0]):
    for j in range(shape[1]):
        y[int(image[i, j][0])] += 1

fig, ax = plt.subplots()
ax.set_title("B05902100 - HW2 graph")
plt.bar(x, np.array(y))
plt.show()

## C, connected components
rows, cols = shape[0], shape[1]
cur_label = 0

# define functions
def get_min_neighbours(Labels, i, j):
    r, c = Labels.shape[0], Labels.shape[1]
    label_list = [Labels[i, j]]
    if 0 <= i+1 < r: label_list.append(Labels[i+1, j]) if Labels[i+1, j] else 0
    if 0 <= i-1 < r: label_list.append(Labels[i-1, j]) if Labels[i-1, j] else 0
    if 0 <= j+1 < c: label_list.append(Labels[i, j+1]) if Labels[i, j+1] else 0
    if 0 <= j-1 < c: label_list.append(Labels[i, j-1]) if Labels[i, j-1] else 0
    return min(label_list)

class Region:
    def __init__(self, point):
        self.points = [point]
        self.top_left = point
        self.bot_right = point
        self.num_pixels = 1
    def update(self, new_point):
        # self.top_left[0] = min(self.top_left[0], new_point[0])  # smallest r
        # self.top_left[1] = min(self.top_left[1], new_point[1])  # smallest c
        # self.bot_right[0] = max(self.bot_right[0], new_point[0])  # largest r
        # self.bot_right[1] = max(self.bot_right[0], new_point[1])  # largest c
        self.num_pixels += 1
        self.points.append(new_point)
    def process(self):
        top_left_r = self.points[0][0]
        top_left_c = self.points[0][1]
        bot_rght_r = self.points[0][0]
        bot_rght_c = self.points[0][1]
        centroid_r = 0
        centroid_c = 0
        for p in self.points:
            top_left_r = min(top_left_r, p[0])
            top_left_c = min(top_left_c, p[1])
            bot_rght_r = max(bot_rght_r, p[0])
            bot_rght_c = max(bot_rght_c, p[1])
            ## find centroid
            centroid_r += p[0] 
            centroid_c += p[1] 
        centroid = (int(centroid_c / self.num_pixels), int(centroid_r / self.num_pixels))
        return (top_left_c, top_left_r), (bot_rght_c, bot_rght_r), centroid

# initialize labels
Labels = np.zeros((rows,cols), dtype=int)
for i in range(rows):
    for j in range(cols):
        if binary_image[i, j][0] == 255:
            cur_label += 1
            Labels[i, j] = cur_label

# the actual thingy
change = True
while change == True:
    """Top down pass"""
    change = False
    for i in range(rows):
        for j in range(cols):
            if Labels[i, j] != 0:
                M = get_min_neighbours(Labels, i, j)
                if M != Labels[i, j]: change = True
                Labels[i, j] = M
    """Bottom up pass"""
    for i in range(rows-1, -1, -1):
        for j in range(cols-1, -1, -1):
            if Labels[i, j] != 0:
                M = get_min_neighbours(Labels, i, j)
                if M != Labels[i, j]: change = True
                Labels[i, j] = M
print("finished the step thingy")

known_regions = dict()

for i in range(rows):
    for j in range(cols):
        l = Labels[i, j]
        if l > 0: # if it's a region
            if int(l) in known_regions: # update the extremities
                known_regions[l].update([i,j])
            else: # create new one
                known_regions[l] = Region([i,j])

print("found {} regions".format(len(known_regions)))

binary_box = binary_image.copy()

for r in known_regions:
    region = known_regions[r]
    if region.num_pixels >= 500:
        top_left, bot_right, centroid = region.process()
        print("region {} has {} pixels (rectangle: {} and {}, centroid: {})".format(r, region.num_pixels, top_left, bot_right, centroid))
        cv2.rectangle(binary_box, top_left, bot_right, (255, 255, 0), 2)
        cv2.circle(binary_box, centroid, 2, (0,255,0), 4)

# cv2.imshow("doctors without binding boxes", binary_box)
## overhead

# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.imwrite("binarized.bmp", binary_image)
cv2.imwrite("boxed.bmp", binary_box)
