
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib import cm
from matplotlib import animation
import matplotlib.colors as colors
import numpy as np
import torch
from scipy import interpolate
import scipy
import chart_studio.plotly as py
import plotly.graph_objects as go
from  plotly.offline import plot
from collections import deque

plt.rcParams['text.usetex'] = True


# utility function that visualize the sampling process via a video
def save_sampling_video(sample_seq, output_path, num_forward_steps, steps=None):
    """Given a sequence of DDPM reverse process samples, save a 15s video of 
    the sampling process.
    """
    # todo: we need to keep the total number of frames fixed
    def update_colorbar_limits_and_cmap(surface, z):
#         cbar = fig.colorbar(surface, shrink=0.5, aspect=10)
        cbar.mappable.set_clim(vmin=np.min(z), vmax=np.max(z))
        cbar.update_normal(surface)

    def set_camera_angle(ax, elev, azim):
        ax.view_init(elev=elev, azim=azim)
    
    def elevation_map(t):
        return sample_seq[t]
    
    def update(t):
        ax.clear()
        i = t // rendered_diffusion_interval
        if i < num_forward_steps:
            z = elevation_map(i)
            if steps:
                ax.set_title('Step ' + str(steps[i]))
        else:
            z = elevation_map(num_forward_steps - 1)
            if steps:
                ax.set_title('Step ' + str(steps[num_forward_steps - 1]))
            
        surface = ax.plot_surface(x, y, z * 2, cmap='turbo', linewidth=0, antialiased=False, alpha=1, rcount=1280, ccount=1280, edgecolor='none')
        ax.set_aspect('equal')
        azim = -np.pi# + t * dtheta
        set_camera_angle(ax, elev=45, azim=np.degrees(azim))
#         update_colorbar_limits_and_cmap(fig, surface, z)
        return surface,

    # video-related parameters
    video_time = 15 # 15 second
    fps = 60 # 60 frame per second
    dt = 1.0 / fps * 1000 # in ms
    total_num_frames = video_time * fps
    rendered_diffusion_interval = int(total_num_frames / num_forward_steps)
    dtheta = 2 * np.pi / total_num_frames 

    sample_seq = np.squeeze(sample_seq, axis=1)
    print(sample_seq.shape)
    T, H, W = sample_seq.shape
    resolution = 0.1
    x = np.arange(0, H*resolution, resolution)
    y = np.arange(0, W*resolution, resolution)
    x, y = np.meshgrid(x, y)
    first_sample = sample_seq[-1].squeeze()    
    fig = plt.figure(figsize=(32, 18))
    ax = fig.add_subplot(projection='3d')
    set_camera_angle(ax, elev=45, azim=-90)
    surf = ax.plot_surface(x, y, first_sample, cmap='terrain', linewidth=1, antialiased=False)
    # Add a colorbar to represent the elevation values
    cbar = fig.colorbar(surf, shrink=0.5, aspect=5)
    update_colorbar_limits_and_cmap(surf, first_sample)
    # Set labels for axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Elevation')
    
    # Show the plot
    anim = FuncAnimation(fig, update, frames=np.arange(0, total_num_frames), interval=dt, blit=True)

    # Save as mp4 (requires ffmpeg)
    anim.save(output_path, writer='ffmpeg')
    plt.show()


# utility function that visualizes the elevation map in 3D
def visualize_elevation_map(viz_samples, n_rows=2, n_cols=3, azim=-45, title=None, masks=None, origin=None, subtitles=None, save_dir=None):
    # we choose the first six samples to show
    num_viz = n_rows * n_cols
    if isinstance(viz_samples, torch.Tensor):
    # vis_samples = viz_samples.squeeze().to('cpu').detach().numpy()
        vis_samples = viz_samples.view(num_viz, 128, 128).to('cpu').detach().numpy()
    else:
        vis_samples = viz_samples.reshape(num_viz, 128, 128)
    B, H, W = viz_samples.shape
    resolution = 0.1
    x = np.arange(0, H*resolution, resolution)
    y = np.arange(0, W*resolution, resolution)
    x, y = np.meshgrid(x, y)

    xnew = np.arange(0, H*resolution, resolution * 0.1)
    ynew = np.arange(0, W*resolution, resolution * 0.1)
    xnew, ynew = np.meshgrid(xnew, ynew)
    
    fig = plt.figure(figsize=(16*n_rows, 9*n_cols))
    
    # Normalize the colormap
    norm = colors.Normalize(vmin=np.min(vis_samples), vmax=np.max(vis_samples))
    # Create a ScalarMappable with the desired colormap
    sm = plt.cm.ScalarMappable(cmap='terrain', norm=norm)
    sm.set_array([])  # Important: This line is required to make the colorbar work
    fig.suptitle(title, fontsize=16)
    for i in range(n_rows):
        for j in range(n_cols):
            index = i*n_cols+j
            ax = fig.add_subplot(n_rows, n_cols, index+1, projection='3d')
            if subtitles != None:
                ax.set_title(subtitles[index])
            ax.view_init(elev=50, azim=azim)            

            if isinstance(masks, np.ndarray) and isinstance(origin, np.ndarray):
                if index == 0:
                    z1 = origin
                else:
                    z1 = vis_samples[index - 1].copy()
                
                ax.plot_surface(x, y, z1 * 2, cmap='turbo', linewidth=0, antialiased=False, alpha=1, rcount=1280, ccount=1280, edgecolor='none')

                for ii in range(128):
                    for jj in range(128):
                        if (vis_samples[index][ii][jj] - z1[ii][jj]) < 0.2:
                        # if masks[ii][jj] == 0:
                            vis_samples[index][ii][jj] = np.nan
                            pass
                cmap = cm.viridis
                # surf = ax.plot_surface(x, y, vis_samples[index] * 2., color='g', linewidth=0, antialiased=False, alpha=0.8, rcount=1280, ccount=1280, edgecolor='none')
            else:
                surf = ax.plot_surface(x, y, vis_samples[index] * 2., color='g', linewidth=0, antialiased=False, alpha=1, rcount=1280, ccount=1280, edgecolor='none')
                ax.set_title('Step ' + str(((index + 1) % 19) * 50) + '  lambda ' + str(((index + 1) // 19) * 0.1))
            
            ax.grid(False)
            ax.axis('off')
            ax.set_aspect('equal')
    # Create the colorbar outside the plot
    cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
    fig.colorbar(sm, cax=cbar_ax)
    plt.subplots_adjust(left=0, top=0.95, wspace=0.1, hspace=0.1, right=0.9)
    plt.show()
    if save_dir != None:
        fig.savefig(save_dir, dpi=300, format='svg', bbox_inches='tight')


# utility function that visualizes the elevation map in 3D
def visualize_leveled_elevation_map(viz_samples, n_rows=2, n_cols=3, azim=-45, title=None, subtitles=None, save_dir=None):
    # we choose the first six samples to show
    num_viz = n_rows * n_cols 
    if isinstance(viz_samples, torch.Tensor):
    # vis_samples = viz_samples.squeeze().to('cpu').detach().numpy()
        vis_samples = viz_samples.view(num_viz, 128, 128).to('cpu').detach().numpy()
    else:
        vis_samples = viz_samples.reshape(num_viz, 128, 128)
    B, H, W = viz_samples.shape
    resolution = 0.1
    x = np.arange(0, H*resolution, resolution)
    y = np.arange(0, W*resolution, resolution)
    x, y = np.meshgrid(x, y)

    xnew = np.arange(0, H*resolution, resolution * 0.1)
    ynew = np.arange(0, W*resolution, resolution * 0.1)
    xnew, ynew = np.meshgrid(xnew, ynew)
    
    fig = plt.figure(figsize=(16, 9))
    
    # Normalize the colormap
    norm = colors.Normalize(vmin=np.min(vis_samples), vmax=np.max(vis_samples))
    # Create a ScalarMappable with the desired colormap
    sm = plt.cm.ScalarMappable(cmap='terrain', norm=norm)
    sm.set_array([])  # Important: This line is required to make the colorbar work
    fig.suptitle(title, fontsize=16)

    ax = fig.add_subplot(1, 1, 1, projection='3d')
    pre_max_height = 0
    for i in range(n_rows):
        for j in range(n_cols):
            index = i*n_cols+j
            if subtitles != None:
                ax.set_title(subtitles[index])
            ax.view_init(elev=50, azim=azim)

            surf = ax.plot_surface(x, y, vis_samples[index] + pre_max_height * 2, cmap='terrain', linewidth=0, antialiased=False, alpha=1-0.1*index, rcount=128, ccount=128)
            pre_max_height += np.max(vis_samples[index])
    ax.set_aspect('equal')
    # Create the colorbar outside the plot
    cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
    fig.colorbar(sm, cax=cbar_ax)
    plt.subplots_adjust(left=0, top=0.95, wspace=0.1, hspace=0.1, right=0.9)
    plt.show()
    if save_dir != None:
        fig.savefig(save_dir, dpi=300, format='svg', bbox_inches='tight')


def visualize_elevation_map_pyploy(viz_samples, n_cols=1, azim=-45, title=None, subtitles=None, save_dir=None, masks=None):
    # we choose the first six samples to show
    num_viz = n_cols + 1
    if isinstance(viz_samples, torch.Tensor):
    # vis_samples = viz_samples.squeeze().to('cpu').detach().numpy()
        vis_samples = viz_samples.view(num_viz, 128, 128).to('cpu').detach().numpy()
    else:
        vis_samples = viz_samples.reshape(num_viz, 128, 128)
    B, H, W = viz_samples.shape
    resolution = 0.1
    x = np.arange(0, H*resolution, resolution)
    y = np.arange(0, W*resolution, resolution)
    x, y = np.meshgrid(x, y)

    xnew = np.arange(0, H*resolution, resolution * 0.1)
    ynew = np.arange(0, W*resolution, resolution * 0.1)
    xnew, ynew = np.meshgrid(xnew, ynew)

    fig = go.Figure()
    pre_max_height = 0.
    for index in range(n_cols):
        make_int = np.vectorize(int)
        cmap = plt.get_cmap("Pastel1")
        colorscale = [[0, 'rgb' + str(cmap(2)[0:3])], [1, 'rgb' + str(cmap(3)[0:3])]]
        fig.add_trace(go.Surface(z=vis_samples[index], x=x, y=y, surfacecolor=vis_samples[index], colorscale=colorscale))

        if isinstance(masks, np.ndarray):
            z1 = vis_samples[index + 1].copy()
            for ii in range(128):
                for jj in range(128):
                    if np.abs(z1[ii][jj] - vis_samples[index][ii][jj]) < 0.05:
                        z1[ii][jj] = vis_samples[index][ii][jj]
            cmap = plt.get_cmap("Pastel1")
            colorscale = [[0, 'rgb' + str(cmap(5)[0:3])], [1, 'rgb' + str(cmap(5)[0:3])]]
            fig.add_trace(go.Surface(z=z1, x=x, y=y, opacity=0.5, surfacecolor=vis_samples[index], colorscale=colorscale))
    
    # fig.update_traces(contours_z=dict(show=True, usecolormap=True,
    #                               highlightcolor="limegreen", project_z=True))
    # fig = go.Figure(data=[go.Surface(z=vis_samples[0], x=x, y=y)])
    fig.update_layout(title='Mt Bruno Elevation', autosize=False,
                    width=1300, height=1300,
                    margin=dict(l=65, r=50, b=65, t=90))
    fig.update_layout(scene_aspectmode='manual',
                  scene_aspectratio=dict(x=1, y=1, z=(np.max(vis_samples[0]) - np.min(vis_samples[0]))/12.8))
    plot(fig, filename='jupyter-parametric_plot')