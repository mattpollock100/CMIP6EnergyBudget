#Load libraries

import glob
import os

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.path as mpath
import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib.lines import Line2D

import statsmodels.api as sm
from scipy import stats

import iris
import iris.quickplot as qplt
import iris.analysis
import iris.coord_categorisation as icoca
from iris.util import mask_cube_from_shapefile
from iris.util import promote_aux_coord_to_dim_coord

import cartopy.io.shapereader as shpreader

import pandas as pd
import seaborn as sns

from dataclasses import dataclass

from iris.util import unify_time_units
from iris.util import equalise_attributes

import cmocean

import warnings

import pickle

#Load custom functions

from Cube_Functions import *
from Plot_Functions import *
from Climate_Functions import *
from Stats_Functions import *

#Main parameters


experiment = 'piControl' #'piControl'
#control = 'piControl'

root_path = '/gws/nopw/j04/pmip4_vol1/public/matt/data/'

sst_var = 'tos'
mld_var = 'mlotst'
T_var = 'thetao'
#output_var = 'mlt'
output_var = 'deept'

model = 'EC-Earth3'


mlotst_dict = {'EC-Earth3':{'piControl':'mlotst_Omon_EC-Earth3_piControl_r1i1p1f1_gn_225901-275912.nc',
                          'lig127k':'mlotst_Omon_EC-Earth3_lig127k_r1i1p1f1_gn_309001-328912.nc'},
             'CESM2':{'piControl':'mlotst_Omon_CESM2_lig127k_r1i1p1f1_gr_000101-070012.nc'}}

thetao_dict = {'EC-Earth3':{'piControl':'thetao_Omon_EC-Earth3_piControl_r1i1p1f1_gn_230001-239912.nc',
                          'lig127k': 'thetao_Omon_EC-Earth3_lig127k_r1i1p1f1_gn_315001-324912.nc'},
             'CESM2':{'piControl':'thetao_Omon_CESM2_lig127k_r1i1p1f1_gr_050101-060012.nc'}}


mld_file = mlotst_dict[model][experiment]
thetao_file = thetao_dict[model][experiment]


output_file = thetao_file.replace('thetao', output_var)

path = root_path + model + '/' + experiment + '/'
print('loading mixed layer depth data')
mld = iris.load_cube(path + mld_file, mld_var)

print('loading ocean temperature data')
thetao = iris.load_cube(path +thetao_file, T_var)  

print('taking first 16 layers only of ocean temp data to lower computational cost')
thetao = thetao[:,0:16,:,:]
#thetao = thetao[0:12,0:16,:,:]


"""
Compute mixed-layer temperature:
  mask each thetao level where depth <= mixed-layer thickness,
  then average in depth.
"""

print('equalizing attributes')
iris.util.equalise_attributes([thetao, mld])

print('getting time overlaps')
thetao, mld = extract_time_overlap(thetao, mld)

lat = thetao.coord('latitude')
curvilinear = lat.points.ndim > 1

if curvilinear:
    print('curvilinear coords')
else:
    print('lat lon mesh')
    
#print('broadcasting to matching shapes')
#thetao, mld = iris.util.broadcast_to_shape(thetao, mld.shape, 
#                                           broadcast_dimensions=[0,2,3], 
#                                           in_dims=[1])

print('extracting depth of cell layers')
try:
    depth_coord = thetao.coord('depth')
except iris.exceptions.CoordinateNotFoundError:
    # Fallback: grab whichever coord is tagged as the Z-axis
    depth_coord = thetao.coord(axis='Z')
depth_points = depth_coord.points  # 1D array, length = n_depth

print('building mask of mixed layer cells')
depth_3d = depth_points[:, np.newaxis, np.newaxis]
#    mld_3d: shape (time, 1, lat, lon)
mld_3d = mld.data[:, np.newaxis, :, :]

#########
if output_var == 'mlt':
    mask = depth_3d <= mld_3d   # broadcasting → (time, depth, lat, lon)
else:
    mask = depth_3d > mld_3d

print('pulling out the raw data array, filling with NaN')
data = thetao.data.filled(np.nan)

#print('masking out any “unphysical” values by turning them into NaN')
#only need this for some models
#data = np.where(data>100, np.nan, data)


#print('masking everywhere depth > MLD, again via NaN')
#data = np.where(mask, data, np.nan)



print('getting thickness of cells for weighting the average temp')
dp = depth_points                              # array of mid-depths, length = n_levels
# build the layer-boundary array, length = n_levels+1
bnd = np.empty(len(dp)+1)
bnd[0]     = 0                                  # assume surface at z=0
bnd[1:-1]  = 0.5*(dp[:-1] + dp[1:])            # mid-points between adjacent levels
# extrapolate the bottom boundary
bnd[-1]    = dp[-1] + 0.5*(dp[-1] - dp[-2])

thickness = np.diff(bnd)                       # Δz for each level
dz = thickness[np.newaxis, :, np.newaxis, np.newaxis]

print('masking everywhere depth > MLD')
# zero thickness below the mixed layer
dz_masked = np.where(mask, dz, 0.0)

print('getting mean temp over the depth‐axis, ignoring NaNs')

# numerator: sum over depth of T*Δz
num   = np.nansum(data * dz_masked, axis=1)
# denominator: sum over depth of Δz
denom = np.nansum(dz_masked,       axis=1)

mlt_data = num / denom
#mlt_data = np.nanmean(data, axis=1)  ##This version wasn't weighted correctly


if curvilinear:
    print('getting curvilinear coords')
    time = thetao.coord('time')
    depth = thetao.coord('depth')
    
    print('making index‐based dim_coords for the horizontal axes')
    nj, ni = thetao.shape[2], thetao.shape[3]
    y_index = iris.coords.DimCoord(
        np.arange(nj),
        long_name='cell index along second dimension',
        units='1', var_name='y'
    )
    x_index = iris.coords.DimCoord(
        np.arange(ni),
        long_name='cell index along first dimension',
        units='1', var_name='x'
    )
    
    print('creating mixed layer temperature cube')
    mlt_cube = iris.cube.Cube(
        mlt_data,
        long_name='mixed_layer_temperature',
        var_name=output_var,
        units=thetao.units,
        dim_coords_and_dims=[
            (time,  0),
            (y_index, 1),
            (x_index, 2),
        ],
        aux_coords_and_dims=[
            (thetao.coord('latitude'),  (1,2)),
            (thetao.coord('longitude'), (1,2)),
        ]
    )
    
else:
    print('creating co-ordinates')
    time_coord, depth_coord, lat_coord, lon_coord = thetao.dim_coords
    
    print('creating mixed layer temperature cube')
    mlt_cube = iris.cube.Cube(
        mlt_data,
        long_name='mixed_layer_temperature',
        var_name=output_var,
        units=thetao.units,
        dim_coords_and_dims=[
            (time_coord, 0),
            (lat_coord, 1),
            (lon_coord, 2),
        ],
        aux_coords_and_dims=[
            (thetao.coord('latitude'),  (1,2)),
            (thetao.coord('longitude'), (1,2)),
        ]
    )

print('saving cube to ' + path + output_file)
iris.save(mlt_cube, path + output_file)

