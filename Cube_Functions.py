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

from iris.coords import AuxCoord

from typing import List

import cartopy.io.shapereader as shpreader

import pandas as pd
import seaborn as sns

from dataclasses import dataclass

from iris.util import unify_time_units
from iris.util import equalise_attributes

import cmocean

import warnings

import cftime

from typing import List

def test():
    print('hello world')

def test2():
    print('hello world two')


def add_coords(cube, add_time = True, add_bounds = True):
    #Add time co-ordinates
    if add_time:
        try: #add month number
            icoca.add_month_number(cube, 'time', name='month_number')
        except:
            pass #print('didnt add month number')
        try: #add year
            icoca.add_year(cube, 'time', name='year')#new
        except:
            pass #print('didnt add year')
        try:#add month
            icoca.add_month(cube, 'time', name='month')#new
        except:
            pass #print('didnt add month')
        try:
            icoca.add_day_of_year(cube, 'time', name='day_of_year')
        except:
            pass #print('didnt add day')

    if add_bounds:
        #Add cell bounds
        try:
            cube.coord('latitude').guess_bounds()
        except:
            pass #print('didnt guess lat')
        try:
            cube.coord('longitude').guess_bounds()
        except:
            pass #print('didnt guess long')

    return cube

def create_path(model, experiment, var, root_path):
    file = f"{var}_*_{model}_{experiment}_*.nc"    
    if var.startswith("si") or var in ['tos', 'mlotst', 'mlt', 'deept']:
        file = f"{var}_*_{model}_{experiment}_*_regrid_*.nc"
    path = f"{root_path}{model}/{experiment}/{file}"
    return path
    
def get_cube(path, var, con, shape = None):
    cube = iris.load_cube(path, var&con)
    cube = add_coords(cube)
    if not shape == None:
        cube = mask_cube_from_shapefile(cube, shape)
    return cube

def get_iris_op(analysis):
    try:
        op = getattr(iris.analysis, analysis)
    except:
        try:
            op = getattr(iris.analysis, analysis.upper())
        except AttributeError:
            raise ValueError(f"Analysis method {analysis!r} not found in iris.analysis")
    return op
    
def get_monthly_data(cube, analysis = 'sum'):
    #requires an unweighted cube
    area_weights = iris.analysis.cartography.area_weights(cube)
    op = get_iris_op(analysis)                
    monthly_data = (cube.collapsed(['latitude', 'longitude'], op, weights=area_weights)).aggregated_by('month', iris.analysis.MEAN).data
    
    return monthly_data

def get_changes_cube(cube):

    change_cube = cube.copy()
    change_cube.data[1:] = cube.data[1:] - cube.data[:-1]
    change_cube.data[0] = change_cube.data[11] #don't have the first January, so make it equal to second january. 
    return change_cube

def get_timeseries(cube, analysis = 'sum'):
    #requires an unweighted cube
    area_weights = iris.analysis.cartography.area_weights(cube)
    op = get_iris_op(analysis)                
    timeseries = (cube.collapsed(['latitude', 'longitude'], op, weights=area_weights)).data
    
    return timeseries

def get_weighted_cube(path, var, con, shape = None):
    cube = iris.load_cube(path, var&con)
    cube = add_coords(cube)
    if not shape == None:
        cube = mask_cube_from_shapefile(cube, shape)

    #cube_dict[var] = cube
    cell_area = iris.analysis.cartography.area_weights(cube)
    cube_weighted=cube*cell_area


    return cube_weighted, cell_area

def mask_cube(cube, mask):
    cube_masked = cube.copy()
    cube_masked.data = np.ma.MaskedArray(cube.data, mask=~mask)
    return cube_masked
    

    
# def extract_time_overlap(*cubes: iris.cube.Cube) -> List[iris.cube.Cube]:
#     """
#     Extracts the overlapping time period from all provided cubes.

#     Parameters:
#         *cubes: Variable number of Iris cubes with time coordinates.

#     Returns:
#         A list of cubes constrained to the overlapping time period.

#     Raises:
#         ValueError: If fewer than 2 cubes are given or no overlap exists.
#     """
#     if len(cubes) < 2:
#         raise ValueError("At least two cubes are required to compute overlapping time.")

#     # # Determine datetime range for each cube
#     time_ranges = []
#     for cube in cubes:
#         time_coord = cube.coord('time')
#         datetimes = time_coord.units.num2date(time_coord.points)
#         time_ranges.append((datetimes[0], datetimes[-1]))

#     # Find common overlapping start and end
#     start_overlap = max(start for start, _ in time_ranges)
#     end_overlap = min(end for _, end in time_ranges)

#     if start_overlap > end_overlap:
#         raise ValueError("No overlapping time period between the provided cubes.")

#     # Apply the time constraint
#     time_constraint = iris.Constraint(time=lambda cell: start_overlap <= cell.point <= end_overlap)
#     return [cube.extract(time_constraint) for cube in cubes]



CF_DATETIME_TYPES = (
    cftime.DatetimeGregorian,
    cftime.DatetimeProlepticGregorian,
    cftime.DatetimeJulian,
    cftime.DatetimeNoLeap,
    cftime.Datetime360Day,
    cftime.DatetimeAllLeap,
)

def to_gregorian(dt):
    if isinstance(dt, CF_DATETIME_TYPES):
        return cftime.DatetimeGregorian(*dt.timetuple()[:6])
    return dt

def extract_time_overlap(*cubes: iris.cube.Cube) -> List[iris.cube.Cube]:
    """
    Extracts the overlapping time period from all provided cubes.

    Parameters:
        *cubes: Variable number of Iris cubes with time coordinates.

    Returns:
        A list of cubes constrained to the overlapping time period.

    Raises:
        ValueError: If fewer than 2 cubes are given or no overlap exists.
    """
    if len(cubes) < 2:
        raise ValueError("At least two cubes are required to compute overlapping time.")

    # Step 1: Compute overlapping period in Gregorian calendar
    time_ranges = []
    for cube in cubes:
        time_coord = cube.coord('time')
        datetimes = time_coord.units.num2date(time_coord.points)
        gregorian_datetimes = [to_gregorian(dt) for dt in datetimes]
        time_ranges.append((gregorian_datetimes[0], gregorian_datetimes[-1]))

    start_overlap = max(start for start, _ in time_ranges)
    end_overlap = min(end for _, end in time_ranges)

    if start_overlap > end_overlap:
        raise ValueError("No overlapping time period between the provided cubes.")

    # Step 2: Apply constraint using each cube’s native calendar
    constrained_cubes = []
    for cube in cubes:
        time_coord = cube.coord('time')
        native_start = time_coord.units.date2num(start_overlap)
        native_end = time_coord.units.date2num(end_overlap)
    
        # Use np.isclose to avoid floating point mismatch. Replaces commented out line below
        def is_within_bounds(cell):
            point = time_coord.units.date2num(cell.point)
            return (point >= native_start - 1e-6) and (point <= native_end + 1e-6)
        
        time_constraint = iris.Constraint(time=is_within_bounds)
        # time_constraint = iris.Constraint(
        #     time=lambda cell: native_start <= time_coord.units.date2num(cell.point) <= native_end
        #     )
        
        constrained_cubes.append(cube.extract(time_constraint))

    return constrained_cubes
