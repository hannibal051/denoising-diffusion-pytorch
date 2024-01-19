import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d

# Set random seed for reproducibility
np.random.seed(42)

# Define parameters
grid_size = 100
block_size = 25 

# Generate grid points
x = np.linspace(-5, 5, grid_size)
y = np.linspace(-5, 5, grid_size)
x, y = np.meshgrid(x, y)

# Create block-wise height values
height_blocks = np.random.rand(grid_size // block_size, grid_size // block_size) * 10
height = np.repeat(height_blocks, block_size, axis=0)
height = np.repeat(height, block_size, axis=1)

# Visualize the elevation map
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
surface = ax.plot_surface(x, y, height, cmap="viridis", linewidth=0, antialiased=True)

# Add a color bar to show height values
cbar = fig.colorbar(surface, shrink=0.5, aspect=5)
cbar.set_label("Height")

# Set labels and title
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("2.5D Block-like Elevation Map with Random Obstacles")

plt.show()
