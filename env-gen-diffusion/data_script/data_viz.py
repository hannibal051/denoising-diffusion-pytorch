import os
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

root_folder = "/media/hdd/env-gen-diffusion/elevation"
rand_indices = [0, 1, 2, 3, 4, 5] 

for i in rand_indices:
    img_path = os.path.join(root_folder, f"{i}.png")
    image = imageio.imread("/media/hdd/env-gen-diffusion/elevation/{}.png".format(i))
    if len(image.shape) == 3 and image.shape[2] == 4:
        image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    # Use the grayscale image as height data
    height_data = image 
    print(height_data)

    # Create a meshgrid for X and Y coordinates
    x = np.arange(height_data.shape[1])
    y = np.arange(height_data.shape[0])
    X, Y = np.meshgrid(x, y)

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    ax.plot_surface(X, Y, height_data, cmap='viridis', linewidth=0)

    # Set labels for the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Height')

    # Display the plot
    plt.savefig(f"elevation_{i}.png", bbox_inches='tight', dpi=200)
    plt.show()