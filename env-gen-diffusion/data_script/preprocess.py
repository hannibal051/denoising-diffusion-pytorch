import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
from PIL.TiffTags import TAGS


def crop_tiff_image(tiff_image_path, num_crops, crop_size=(128, 128), nodata_value=-9999):
    tiff_image = Image.open(tiff_image_path)
    image_array = np.array(tiff_image)
    image_array = np.ma.masked_equal(image_array, nodata_value)

# Open the TIFF file

# tiff_tags = tiff_image.tag_v2
# # Iterate through each tag and extract its human-readable name
# print("Resolution", tiff_image.info['dpi'])
# for tag_id, value in tiff_tags.items():
#     tag_name = TAGS.get(tag_id)
#     print(f"{tag_name}: {value}")



# # randomly crop the image
# start_x = 15
# start_y = 100
# crop_size = 128
# image_array = image_array[start_y:start_y+crop_size, start_x:start_x+crop_size]
# height, width = image_array.shape

# # Create a meshgrid of the pixel coordinates
# x = np.arange(0, width)
# y = np.arange(0, height)
# x, y = np.meshgrid(x, y)

# # Plot the 2.5D elevation
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(x, y, image_array, cmap='viridis')

# plt.show()
# # plt.imshow(image_array, cmap='gray')
# # plt.show()