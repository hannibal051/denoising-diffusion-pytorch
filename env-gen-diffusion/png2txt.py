import numpy as np
import open3d as o3d
import laspy
import glob
import scipy
from PIL import Image

im_frame = Image.open('/home/yy/Downloads/Untitled1.png').convert('L')
img = np.array(im_frame.getdata()).reshape(1000, 1000)

def generate_points_in_circle(center_x, center_y, radius):
    points = []

    for x in range(center_x - radius, center_x + radius + 1):
        for y in range(center_y - radius, center_y + radius + 1):
            if (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2:
                points.append((x, y, 1))
    return points

radius = int(40 + 30 * 0.4)

out = []
circles = set()
for i in range(len(img)):
    for j in range(len(img[0])):
        if img[i][j] < 10:
            circles.add((i, j, 0))
            circle = generate_points_in_circle(i, j, radius)
            for c in circle:
                circles.add(c)
            

out = np.array(list(circles), dtype=int)
# print(out)

np.savetxt("/home/yy/Downloads/map2.txt", out, fmt='%i')