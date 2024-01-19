import numpy as np
import open3d as o3d
import laspy
import glob
import scipy
from PIL import Image

im_frame = Image.open('/home/yy/Downloads/1.png')
_observation_np = np.array(im_frame.getdata()).reshape(480, 640) / 255.0 * 8.0

fu = 1.897929002282757
fv = 1.423446751712068
cam_width = 640
cam_height = 480
cam_z = -(0.095 + 0.3)
centerU = cam_width / 2
centerV = cam_height / 2


depth_buffer = np.zeros_like(_observation_np)
points = []

for i in range(cam_width):
    for j in range(cam_height):
        u = (i-centerU)/(cam_width)  # image-space coordinate
        v = (j-centerV)/(cam_height)  # image-space coordinate
        d = _observation_np[j, i]  # depth buffer value
        if d < 0.1 or d > 5.:
            continue
        X2 = [d*fu*u, d*fv*v, d]  # deprojection vector
        # if abs(-X2[1] - cam_z) < 0.05:
        points.append([X2[2], X2[0], -X2[1]])
points = np.array(points)
print(np.min(points[:, 0]), np.max(points[:, 0]))
print(np.min(points[:, 1]), np.max(points[:, 1]))
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# pcd.colors = o3d.utility.Vector3dVector(rgb/255.)
o3d.visualization.draw_geometries([pcd])