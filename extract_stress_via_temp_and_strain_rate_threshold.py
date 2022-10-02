# LOAD MODULES
# ------------------------------------------------------------------------------------------------ #
import os
import glob

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from skimage.color import rgba2rgb
from skimage.color import rgb2gray
from matplotlib.colors import LightSource
from scipy.interpolate import griddata
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap


# MODEL NAME
# ------------------------------------------------------------------------------------------------ #
model_name = '060_Reference_model_high_res_50_y'


# PARAMETERS
# ------------------------------------------------------------------------------------------------ #
dt = 0.1
png_res = 300
strain_rate_threshold = 1e-16
grid_res = 801
skip = 20
scale = 1


# DEFINITION OF FUNCTIONS
# ------------------------------------------------------------------------------------------------ #
def make_dir(path_model, foldername):
    ''' Makes new directory if folder does not exist yet
    Input:  path_model --> path to the model main folder (str)
    Input:  foldername --> selectable name for new folder (str)
    Output: new directory with folder
    '''
    if os.path.isdir(os.path.join(path_model, foldername)):
        print('png_folder already exists')
    else:
        os.mkdir(os.path.join(path_model, foldername))
        print('make png folder')
        
    path_pngs = os.path.join(path_model, foldername)
    return path_pngs


def prepare_topo_data(file_now):
    ''' Extract surface data from topography files
    Input:  file_now --> recent topography file at current time step (str)
    Output: x, y, z  --> arrays of coordinates
    '''
    
    data_raw = np.loadtxt(file_now)
    data_sorted = data_raw[np.argsort(data_raw[:, 0])]
    
    x = data_sorted[:, 0] - np.mean(data_sorted[:, 0])
    y = data_sorted[:, 1] - np.mean(data_sorted[:, 1])
    z = data_sorted[:, 2]
    
    return x, y, z


def setup_grid(x, y, numx, numy):
    '''' Set up grid for topography
    Input:  x, y        --> arrays of coordinates
    Input:  numx, numy  --> grid resolution
    Output: X, Y        --> 2d coordinate grid
    Output: xv, yv      --> coordinate arrays
    Output: Lx, Ly      --> physical length
    '''
    
    Lx = np.max(x) - np.min(x)
    Ly = np.max(y) - np.min(y)
    
    xv = np.linspace(-Lx / 2, Lx / 2, numx)
    yv = np.linspace(-Ly / 2, Ly / 2, numy)

    X, Y = np.meshgrid([xv], [yv], indexing='ij')
    
    return X, Y, xv, yv, Lx, Ly


def get_initial_surface_stress_points(height):
    ''' get stresses at surface for initial time step based on topography
    Input:  height --> initial position of model surface
    '''
    
    df = pd.read_csv(file_list_stress[0])
    surface_points = df.loc[df['Points:2'] == height]
    id_points = surface_points.index
    
    return id_points, surface_points
    del df, surface_points
    
    
def split_data(data_surface):
    ''' get data for stresses based on headers'''
    xcoords = data_surface[['Points:0']].to_numpy()
    ycoords = data_surface[['Points:1']].to_numpy()
    zcoords = data_surface[['Points:2']].to_numpy()
    
    xcoords = xcoords[1:, 0] - np.mean(xcoords)
    ycoords = ycoords[1:, 0] - np.mean(ycoords)
    zcoords = zcoords[1:, 0]
    
    stress1 = data_surface[[
        'surface_maximum_horizontal_compressive_stress:0']].to_numpy()
    stress2 = data_surface[[
        'surface_maximum_horizontal_compressive_stress:1']].to_numpy()
    stress3 = data_surface[[
        'surface_maximum_horizontal_compressive_stress:2']].to_numpy()
    
    stress1 = stress1[1:, 0]
    stress2 = stress2[1:, 0]
    stress3 = stress3[1:, 0]

    return xcoords, ycoords, zcoords, stress1, stress2
    del data_surface
    

def find_nearest(array, value):
    ''' find closest value'''
    dist_array = np.abs(array - value)
    min_idx = dist_array.argmin()
    closest_element = array[min_idx]
    return min_idx, closest_element
    
    
# MAKE DIRECTORIES
# ------------------------------------------------------------------------------------------------ #
path_main = os.getcwd()
path_model = os.path.join(path_main, model_name)
path_data = os.path.join(path_model, 'output')


# COLORMAP
# ------------------------------------------------------------------------------------------------ #
cm_data = np.loadtxt("broc.txt")
col_map = LinearSegmentedColormap.from_list('broc', cm_data)


# GET FILE LIST AND MAKE FOLDERS
# ------------------------------------------------------------------------------------------------ #
os.chdir(path_data)
file_list_topo = sorted(filter(os.path.isfile, glob.glob('topography.*')))
file_list_stress = sorted(filter(os.path.isfile, glob.glob('stress_via_strain_rate_new*')))

path_pngs_topo = make_dir(path_data, 'pngs_topo')
path_pngs_stress = make_dir(path_data, 'pngs_stress_new')


# GET CURRENT FILE --> START LOOP
# ------------------------------------------------------------------------------------------------ #
for time_step, item in enumerate(file_list_topo):
    file_now_topo = file_list_topo[time_step]
    file_now_stress = file_list_stress[time_step]
    
    print(file_now_stress)
    
    
# GET TOPO DATA
# ------------------------------------------------------------------------------------------------ #
    # get coordinates out of file
    x_tp, y_tp, z_tp = prepare_topo_data(file_now_topo)
    
    # extract height at t = 0
    if time_step == 0:
        height = z_tp.max()
        
    # set up grid
    X_tp, Y_tp, xv_tp, yv_tp, Lx_tp, Ly_tp = setup_grid(x_tp, y_tp, numx=grid_res, numy=grid_res)
    
    # interpolate data on structured grid
    zv_tp = griddata((x_tp, y_tp), z_tp, (X_tp, Y_tp), method='cubic') - height
    
    # get background data
    ls = LightSource(azdeg=315, altdeg=30)
    background = ls.shade(np.rot90(zv_tp, k=1, axes=(0, 1)), plt.cm.gist_gray)
    bg_gray = rgb2gray(rgba2rgb(background))
    
    
# GET STRESS DATA FOR FIRST STEP VIA COORDINATES (SURF TEMP AT T=0 SUCKS)
# ------------------------------------------------------------------------------------------------ #
    if time_step == 0:
        id_points_stress, surface_points_stress = get_initial_surface_stress_points(height)
        x_sig, y_sig, z_sig, sig1, sig2 = split_data(surface_points_stress)
        
        # set up structured grid
        X_sig, Y_sig, xv_sig, yv_sig, Lx_sig, Ly_sig = setup_grid(x_sig, y_sig, numx=grid_res, numy=grid_res)
                                                                    
        # exclude boundary area
        low_res_rim = 0.025

        xlower = find_nearest(xv_sig, -Lx_sig / 2 + low_res_rim)
        xupper = find_nearest(xv_sig, Lx_sig / 2 - low_res_rim)
        ylower = find_nearest(yv_sig, -Ly_sig / 2 + low_res_rim)
        yupper = find_nearest(yv_sig, Ly_sig / 2 - low_res_rim)

        X_clean = X_sig[xlower[0]:xupper[0], ylower[0]:yupper[0]]
        Y_clean = Y_sig[xlower[0]:xupper[0], ylower[0]:yupper[0]]
        
        # get grid for imshow background
        dx_sig = (xv_sig[1] - xv_sig[0]) / 2
        dy_sig = (yv_sig[1] - yv_sig[0]) / 2
        extent = [xv_sig[0] - dx_sig, xv_sig[-1] + dx_sig, yv_sig[0] - dy_sig, yv_sig[-1] + dy_sig]
        
        # interpolate values onto structured grid
        zv_sig = griddata((x_sig, y_sig), z_sig, (X_sig, Y_sig), method='nearest') - height
        s1 = griddata((x_sig, y_sig), sig1, (X_sig, Y_sig), method='nearest')
        s2 = griddata((x_sig, y_sig), sig2, (X_sig, Y_sig), method='nearest')
        smag = (s1 ** 2 + s2 ** 2) ** (1 / 2)
            
        # clean values (i.e., exclude boundaries)
        s1_clean = s1[xlower[0]:xupper[0], ylower[0]:yupper[0]]
        s2_clean = s2[xlower[0]:xupper[0], ylower[0]:yupper[0]]
        
        # Normalize the stress magnitude for plotting:
        s1_plot = s1_clean / np.sqrt(s1_clean**2 + s2_clean**2)
        s2_plot = s2_clean / np.sqrt(s1_clean**2 + s2_clean**2)
        
        M = np.abs(np.degrees(np.arctan2(s1_clean, s2_clean)))
        
        # get histogram distribution
        M_flat = np.ravel(M)
        M_flat = np.around(M_flat)
        M_flat = M_flat.astype(int)
        
        # Plot Background and stresses
        # ------------------------------------------------------------------------------------------- #
        fig = plt.figure(figsize=(12, 6), dpi=png_res, constrained_layout=True)
        
        gs = GridSpec(nrows=1, ncols=1, figure=fig)
        
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.set_title('Time: %1.1f Ma' % (dt * time_step))
        ax0.set_xlabel('x width (m)')
        ax0.set_ylabel('y width (m)')
        
        bg = ax0.pcolormesh(X_tp, Y_tp, zv_tp, shading='gouraud', cmap='gist_gray', alpha=1)
        
        # plt.colorbar()
        bg.set_clim(-2500, 1500)
        
        # plot contours ----------------------
        cont_levels = np.linspace(-0.0, 0.0, 1)
        
        CS = ax0.contour(X_tp, Y_tp, zv_tp, cont_levels, colors='white', linewidths=1.0, alpha=0.5)
        
        fmt = {}
        strs = ['0 m']
        for str_pos, s in zip(CS.levels, strs):
            fmt[str_pos] = s
            
        ax0.clabel(CS, CS.levels, fmt=fmt, inline=True, fontsize=10)
        # ------------------------------------
        
        q = ax0.quiver(X_clean[::skip, ::skip], Y_clean[::skip, ::skip],
                       scale * s1_plot[::skip, ::skip],
                       scale * s2_plot[::skip, ::skip],
                       color='k',
                       headlength=0, headaxislength=0, zorder=10)
        
        # ticks = np.linspace(0, 180, 10)
        # cb = fig.colorbar(q, ax=ax0, ticks=ticks, aspect=50,
        #                   label='orientation ($\\circ$)')
        # q.set_clim(0, 180)
        
        ax0.set_xticks([-75000, -50000, -25000, 0, 25000, 50000, 75000])
        ax0.set_xticklabels(['-7.5', '-5.0', '-2.5', '0', '2.5', '5.0', '7.5'])
        ax0.set_yticks([-75000, -50000, -25000, 0, 25000, 50000, 75000])
        ax0.set_yticklabels(['-7.5', '-5.0', '-2.5', '0', '2.5', '5.0', '7.5'])
        
        ax0.set_xlabel('width (km)', fontsize=12)
        ax0.set_ylabel('height (km)', fontsize=12)
        ax0.set_title('Time: %1.1f Ma' % (dt * time_step))
        
        ax0.set_aspect('equal', 'box')
        
        ax0.set(xlim=(-Lx_tp / 2, Lx_tp / 2), ylim=(-Ly_tp / 2, Ly_tp / 2))
        fig.savefig(path_pngs_stress + '/Ma_%1.1f' % (dt * time_step) + '.png', dpi=png_res, bbox_inches='tight')
        plt.show()

    else:
        df = pd.read_csv(file_list_stress[time_step])
        temp_data = df[['surface_T']].to_numpy()
        surf_temp = np.min(temp_data)
        surface_points_stress = df.loc[df['surface_T'] == surf_temp]
        
        # set stress values to zero where strain rate is below threshold
        surface_points_stress_threshold = df.loc[(df['surface_T'] == surf_temp) & (df['surface_strain_rate'] >= strain_rate_threshold)]
        index_for_zeros = surface_points_stress[~surface_points_stress.index.isin(surface_points_stress_threshold.index)].index
        surface_points_stress_with_zeros = surface_points_stress.copy(deep=True)
        
        # set stresses to negligibly small values but not zero!
        surface_points_stress_with_zeros.at[index_for_zeros, 'surface_maximum_horizontal_compressive_stress:0'] = np.nan
        surface_points_stress_with_zeros.at[index_for_zeros, 'surface_maximum_horizontal_compressive_stress:1'] = np.nan
        surface_points_stress_with_zeros.at[index_for_zeros, 'surface_maximum_horizontal_compressive_stress:2'] = np.nan
        
        x_sig, y_sig, z_sig, sig1, sig2 = split_data(surface_points_stress)
        x_thr, y_thr, z_thr, sig1_thr, sig2_thr = split_data(surface_points_stress_with_zeros)
         
        # interpolate values onto structured grid
        zv_sig = griddata((x_sig, y_sig), z_sig, (X_sig, Y_sig), method='nearest') - height
        s1 = griddata((x_sig, y_sig), sig1, (X_sig, Y_sig), method='nearest')
        s2 = griddata((x_sig, y_sig), sig2, (X_sig, Y_sig), method='nearest')
        smag = (s1 ** 2 + s2 ** 2) ** (1 / 2)
        
        # interpolate values onto structured grid with strain rate mask
        zv_thr = griddata((x_thr, y_thr), z_thr, (X_sig, Y_sig), method='nearest') - height
        s1_thr = griddata((x_thr, y_thr), sig1_thr, (X_sig, Y_sig), method='nearest')
        s2_thr = griddata((x_thr, y_thr), sig2_thr, (X_sig, Y_sig), method='nearest')
        smag_thr = (s1_thr ** 2 + s2_thr ** 2) ** (1 / 2)
        
        # clean values (i.e., exclude boundaries)
        s1_clean = s1[xlower[0]:xupper[0], ylower[0]:yupper[0]]
        s2_clean = s2[xlower[0]:xupper[0], ylower[0]:yupper[0]]
        
        # clean values (i.e., exclude boundaries)
        s1_thr_clean = s1_thr[xlower[0]:xupper[0], ylower[0]:yupper[0]]
        s2_thr_clean = s2_thr[xlower[0]:xupper[0], ylower[0]:yupper[0]]
        
        # Normalize the stress magnitude for plotting:
        s1_plot = s1_clean / np.sqrt(s1_clean**2 + s2_clean**2)
        s2_plot = s2_clean / np.sqrt(s1_clean**2 + s2_clean**2)
        
        # Normalize the stress magnitude for plotting:
        s1_thr_plot = s1_thr_clean / np.sqrt(s1_thr_clean**2 + s2_thr_clean**2)
        s2_thr_plot = s2_thr_clean / np.sqrt(s1_thr_clean**2 + s2_thr_clean**2)

        M = np.abs(np.degrees(np.arctan2(s1_clean, s2_clean)))
        M_thr = np.abs(np.degrees(np.arctan2(s1_thr_clean, s2_thr_clean)))
        
        # get histogram distribution
        M_flat = np.ravel(M)
        M_flat = np.around(M_flat)
        # M_flat = M_flat.astype(int)
        
        np.savetxt('hist_distr_all_new_%3.1f' % (dt * time_step) + '.csv', M_flat, delimiter=",")
        
        # get histogram distribution
        M_thr_flat = np.ravel(M_thr)
        M_thr_flat = np.around(M_thr_flat)
        # M_thr_flat = M_thr_flat.astype(int)
        
        np.savetxt('hist_distr_thresh_new_%3.1f' % (dt * time_step) + '.csv', M_thr_flat, delimiter=",")
        
        # Plot Background and stresses
        # ------------------------------------------------------------------------------------------- #
        fig = plt.figure(figsize=(12, 6), dpi=png_res, constrained_layout=True)
        
        gs = GridSpec(nrows=1, ncols=1, figure=fig)
        
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.set_title('Time: %1.1f Ma' % (dt * time_step))
        ax0.set_xlabel('x width (m)')
        ax0.set_ylabel('y width (m)')
        
        bg = ax0.pcolormesh(X_tp, Y_tp, zv_tp, shading='gouraud', cmap='gist_gray', alpha=1)
        
        # plt.colorbar()
        bg.set_clim(-2500, 1500)
        
        # plot contours ----------------------
        cont_levels = np.linspace(-0.0, 0.0, 1)
        
        CS = ax0.contour(X_tp, Y_tp, zv_tp, cont_levels, colors='white', linewidths=1.0, alpha=0.5)
        
        fmt = {}
        strs = ['0 m']
        for str_pos, s in zip(CS.levels, strs):
            fmt[str_pos] = s
            
        ax0.clabel(CS, CS.levels, fmt=fmt, inline=True, fontsize=10)
        # ------------------------------------
        
        q1 = ax0.quiver(X_clean[::skip, ::skip], Y_clean[::skip, ::skip],
                        scale * s1_plot[::skip, ::skip],
                        scale * s2_plot[::skip, ::skip],
                        color='k',
                        headlength=0, headaxislength=0, zorder=10)
        
        q2 = ax0.quiver(X_clean[::skip, ::skip], Y_clean[::skip, ::skip],
                        scale * s1_thr_plot[::skip, ::skip],
                        scale * s2_thr_plot[::skip, ::skip],
                        color='r',
                        headlength=0, headaxislength=0, zorder=10)
        
        # ticks = np.linspace(0, 180, 10)
        # cb = fig.colorbar(q2, ax=ax0, ticks=ticks, aspect=50,
        #                   label='orientation ($\\circ$)')
        # q.set_clim(0, 180)
        
        ax0.set_xticks([-75000, -50000, -25000, 0, 25000, 50000, 75000])
        ax0.set_xticklabels(['-7.5', '-5.0', '-2.5', '0', '2.5', '5.0', '7.5'])
        ax0.set_yticks([-75000, -50000, -25000, 0, 25000, 50000, 75000])
        ax0.set_yticklabels(['-7.5', '-5.0', '-2.5', '0', '2.5', '5.0', '7.5'])
        
        ax0.set_xlabel('width (km)', fontsize=12)
        ax0.set_ylabel('height (km)', fontsize=12)
        ax0.set_title('Time: %1.1f Ma' % (dt * time_step))
        
        ax0.set_aspect('equal', 'box')
        
        ax0.set(xlim=(-Lx_tp / 2, Lx_tp / 2), ylim=(-Ly_tp / 2, Ly_tp / 2))
        fig.savefig(path_pngs_stress + '/Ma_%1.1f' % (dt * time_step) + '.png', dpi=png_res, bbox_inches='tight')
        plt.show()
