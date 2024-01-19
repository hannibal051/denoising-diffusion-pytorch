from pyvista import examples
import numpy as np
import open3d as o3d
import laspy
import glob
import scipy
import math
from PIL import Image
import pyvista as pv
from scipy import interpolate
from cupyx.scipy.interpolate import RegularGridInterpolator
import cupy as cp

def h5_to_vtp(xcoordinates, ycoordinates, vals, surf_name='train_loss', log=False, zmax=-1, interp=0.2):
    #set this to True to generate points
    show_points = False
    #set this to True to generate polygons
    show_polys = True

    x_array = xcoordinates[:].ravel()
    y_array = ycoordinates[:].ravel()
    z_array = vals[:].ravel()

    # Interpolate the resolution up to the desired amount
    if interp > 0:
        # m = interpolate.interp2d(xcoordinates[0,:], ycoordinates[:,0], vals, kind='cubic')
        # x_array = np.linspace(min(x_array), max(x_array), interp)
        # y_array = np.linspace(min(y_array), max(y_array), interp)
        # z_array = m(x_array, y_array).ravel()

        # x_array, y_array = np.meshgrid(x_array, y_array)
        # x_array = x_array.ravel()
        # y_array = y_array.ravel()

        x = cp.arange(0, len(vals[0]), 1)
        y = cp.arange(0, len(vals), 1)
        interp_ = RegularGridInterpolator((x, y), cp.array(vals), bounds_error=False, fill_value=None, method='linear')
        xnew = cp.arange(0, len(vals[0]), interp)
        ynew = cp.arange(0, len(vals), interp)
        Xnew, Ynew = cp.meshgrid(xnew, ynew, indexing='ij')
        Znew = interp_((Xnew, Ynew))
        x_array = Xnew[:].ravel()
        y_array = Ynew[:].ravel()
        z_array = Znew[:].ravel()

    vtp_file = "/home/fish/vtk"
    if zmax > 0:
        z_array[z_array > zmax] = zmax
        vtp_file +=  "_zmax=" + str(zmax)

    if log:
        z_array = np.log(z_array + 0.1)
        vtp_file +=  "_log"
    vtp_file +=  ".vtp"
    print("Here's your output file:{}".format(vtp_file))

    number_points = len(z_array)
    print("number_points = {} points".format(number_points))

    matrix_size = int(math.sqrt(number_points))
    print("matrix_size = {} x {}".format(matrix_size, matrix_size))

    poly_size = matrix_size - 1
    print("poly_size = {} x {}".format(poly_size, poly_size))

    number_polys = poly_size * poly_size
    print("number_polys = {}".format(number_polys))

    min_value_array = [min(x_array), min(y_array), min(z_array)]
    max_value_array = [max(x_array), max(y_array), max(z_array)]
    min_value = min(min_value_array)
    max_value = max(max_value_array)

    averaged_z_value_array = []

    # poly_count = 0
    # for column_count in range(poly_size):
    #     stride_value = column_count * matrix_size
    #     for row_count in range(poly_size):
    #         temp_index = stride_value + row_count
    #         averaged_z_value = (z_array[temp_index] + z_array[temp_index + 1] +
    #                             z_array[temp_index + matrix_size]  +
    #                             z_array[temp_index + matrix_size + 1]) / 4.0
    #         averaged_z_value_array.append(averaged_z_value)
    #         poly_count += 1

    # avg_min_value = min(averaged_z_value_array)
    # avg_max_value = max(averaged_z_value_array)
    avg_min_value = min_value
    avg_max_value = max_value

    output_file = open(vtp_file, 'w')
    output_file.write('<VTKFile type="PolyData" version="1.0" byte_order="LittleEndian" header_type="UInt64">\n')
    output_file.write('  <PolyData>\n')

    if (show_points and show_polys):
        output_file.write('    <Piece NumberOfPoints="{}" NumberOfVerts="{}" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="{}">\n'.format(number_points, number_points, number_polys))
    elif (show_polys):
        output_file.write('    <Piece NumberOfPoints="{}" NumberOfVerts="0" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="{}">\n'.format(number_points, number_polys))
    else:
        output_file.write('    <Piece NumberOfPoints="{}" NumberOfVerts="{}" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="">\n'.format(number_points, number_points))

    # <PointData>
    output_file.write('      <PointData>\n')
    output_file.write('        <DataArray type="Float32" Name="zvalue" NumberOfComponents="1" format="ascii" RangeMin="{}" RangeMax="{}">\n'.format(min_value_array[2], max_value_array[2]))
    for vertexcount in range(number_points):
        if (vertexcount % 6) == 0:
            output_file.write('          ')
        output_file.write('{}'.format(z_array[vertexcount]))
        if (vertexcount % 6) == 5:
            output_file.write('\n')
        else:
            output_file.write(' ')
    if (vertexcount % 6) != 5:
        output_file.write('\n')
    output_file.write('        </DataArray>\n')
    output_file.write('      </PointData>\n')

    # <CellData>
    output_file.write('      <CellData>\n')
    if (show_polys and not show_points):
        output_file.write('        <DataArray type="Float32" Name="averaged zvalue" NumberOfComponents="1" format="ascii" RangeMin="{}" RangeMax="{}">\n'.format(avg_min_value, avg_max_value))
        for vertexcount in range(number_polys):
            if (vertexcount % 6) == 0:
                output_file.write('          ')
            output_file.write('{}'.format(averaged_z_value_array[vertexcount]))
            if (vertexcount % 6) == 5:
                output_file.write('\n')
            else:
                output_file.write(' ')
        if (vertexcount % 6) != 5:
            output_file.write('\n')
        output_file.write('        </DataArray>\n')
    output_file.write('      </CellData>\n')

    # <Points>
    output_file.write('      <Points>\n')
    output_file.write('        <DataArray type="Float32" Name="Points" NumberOfComponents="3" format="ascii" RangeMin="{}" RangeMax="{}">\n'.format(min_value, max_value))
    for vertexcount in range(number_points):
        if (vertexcount % 2) == 0:
            output_file.write('          ')
        output_file.write('{} {} {}'.format(x_array[vertexcount], y_array[vertexcount], z_array[vertexcount]))
        if (vertexcount % 2) == 1:
            output_file.write('\n')
        else:
            output_file.write(' ')
    if (vertexcount % 2) != 1:
        output_file.write('\n')
    output_file.write('        </DataArray>\n')
    output_file.write('      </Points>\n')

    # <Verts>
    output_file.write('      <Verts>\n')
    output_file.write('        <DataArray type="Int64" Name="connectivity" format="ascii" RangeMin="0" RangeMax="{}">\n'.format(number_points - 1))
    if (show_points):
        for vertexcount in range(number_points):
            if (vertexcount % 6) == 0:
                output_file.write('          ')
            output_file.write('{}'.format(vertexcount))
            if (vertexcount % 6) == 5:
                output_file.write('\n')
            else:
                output_file.write(' ')
        if (vertexcount % 6) != 5:
            output_file.write('\n')
    output_file.write('        </DataArray>\n')
    output_file.write('        <DataArray type="Int64" Name="offsets" format="ascii" RangeMin="1" RangeMax="{}">\n'.format(number_points))
    if (show_points):
        for vertexcount in range(number_points):
            if (vertexcount % 6) == 0:
                output_file.write('          ')
            output_file.write('{}'.format(vertexcount + 1))
            if (vertexcount % 6) == 5:
                output_file.write('\n')
            else:
                output_file.write(' ')
        if (vertexcount % 6) != 5:
            output_file.write('\n')
    output_file.write('        </DataArray>\n')
    output_file.write('      </Verts>\n')

    # <Lines>
    output_file.write('      <Lines>\n')
    output_file.write('        <DataArray type="Int64" Name="connectivity" format="ascii" RangeMin="0" RangeMax="{}">\n'.format(number_polys - 1))
    output_file.write('        </DataArray>\n')
    output_file.write('        <DataArray type="Int64" Name="offsets" format="ascii" RangeMin="1" RangeMax="{}">\n'.format(number_polys))
    output_file.write('        </DataArray>\n')
    output_file.write('      </Lines>\n')

    # <Strips>
    output_file.write('      <Strips>\n')
    output_file.write('        <DataArray type="Int64" Name="connectivity" format="ascii" RangeMin="0" RangeMax="{}">\n'.format(number_polys - 1))
    output_file.write('        </DataArray>\n')
    output_file.write('        <DataArray type="Int64" Name="offsets" format="ascii" RangeMin="1" RangeMax="{}">\n'.format(number_polys))
    output_file.write('        </DataArray>\n')
    output_file.write('      </Strips>\n')

    # <Polys>
    output_file.write('      <Polys>\n')
    output_file.write('        <DataArray type="Int64" Name="connectivity" format="ascii" RangeMin="0" RangeMax="{}">\n'.format(number_polys - 1))
    if (show_polys):
        polycount = 0
        for column_count in range(poly_size):
            stride_value = column_count * matrix_size
            for row_count in range(poly_size):
                temp_index = stride_value + row_count
                if (polycount % 2) == 0:
                    output_file.write('          ')
                output_file.write('{} {} {} {}'.format(temp_index, (temp_index + 1), (temp_index + matrix_size + 1), (temp_index + matrix_size)))
                if (polycount % 2) == 1:
                    output_file.write('\n')
                else:
                    output_file.write(' ')
                polycount += 1
        if (polycount % 2) == 1:
            output_file.write('\n')
    output_file.write('        </DataArray>\n')
    output_file.write('        <DataArray type="Int64" Name="offsets" format="ascii" RangeMin="1" RangeMax="{}">\n'.format(number_polys))
    if (show_polys):
        for polycount in range(number_polys):
            if (polycount % 6) == 0:
                output_file.write('          ')
            output_file.write('{}'.format((polycount + 1) * 4))
            if (polycount % 6) == 5:
                output_file.write('\n')
            else:
                output_file.write(' ')
        if (polycount % 6) != 5:
            output_file.write('\n')
    output_file.write('        </DataArray>\n')
    output_file.write('      </Polys>\n')

    output_file.write('    </Piece>\n')
    output_file.write('  </PolyData>\n')
    output_file.write('</VTKFile>\n')
    output_file.write('')
    output_file.close()

    print("Done with file:{}".format(vtp_file))

terrain_path = "/home/fish/terrain/denoising-diffusion-pytorch/env-gen-diffusion/test_imgs/test/7.txt"
# terrain_path = "/home/fish/isaacgym/sim-to-real-offroad/assets/tif/output_hh.tif"

if terrain_path[-3:] == 'tif':
    elevation_image = np.array(Image.open(terrain_path), dtype=np.float32)[:128*4, :128*4]
elif terrain_path[-3:] == 'txt':
    elevation_image = np.loadtxt(terrain_path, dtype=np.float32)

if terrain_path[-3:] == 'tif':
    elevation_image = (elevation_image - np.min(elevation_image)) / (np.max(elevation_image) - np.min(elevation_image)) * 15.
elif terrain_path[-3:] == 'txt':
    elevation_image = (elevation_image - np.min(elevation_image)) * 5.

x = cp.arange(0, len(elevation_image[0]), 1) * 0.1
y = cp.arange(0, len(elevation_image), 1) * 0.1
xg, yg = cp.meshgrid(x, y, indexing='ij')

# h5_to_vtp(xg.get(), yg.get(), elevation_image)

# interp = RegularGridInterpolator((x, y), cp.array(elevation_image), bounds_error=False, fill_value=None, method='linear')
# xnew = cp.arange(0, len(elevation_image[0]), 1) * 0.1
# ynew = cp.arange(0, len(elevation_image), 1) * 0.1
# Xnew, Ynew = cp.meshgrid(xnew, ynew, indexing='ij')
# Znew = interp((Xnew, Ynew))
# Create and plot structured grid
grid = pv.StructuredGrid(xg.get(), yg.get(), elevation_image)
grid['lidar'] = elevation_image.ravel(order='F')
grid.camera_position = 'xy'
grid.plot(scalars='lidar', notebook=False, cmap='viridis', multi_colors=True, eye_dome_lighting=True)

# pl = pv.Plotter()
# _ = pl.add_mesh(grid, smooth_shading=True, show_edges=False, show_vertices=False, cmap='viridis', multi_colors=True, show_scalar_bar=False)
# pl.enable_eye_dome_lighting()
# pl.camera_position = 'xy'
# pl.save_graphic("~/img.svg")