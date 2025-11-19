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

def anomaly_dict(experiment_dict, control_dict):
#requires a dictionary of dictionaries, where first key is the model, 
#second key is the variable, value is 1-d array of data
#e.g. experiment_dict = {'model1':{'var1':[1, 2, 3, 4, 5], 'var2' : [5, 6, 7, 8, 9]}, 
#                        'model2':{'var1': [.....],...}
    models = experiment_dict.keys()  
    anomaly_dict = {
    model: {
        k: experiment_dict[model][k] - control_dict[model][k]
        for k in experiment_dict[model]
        }
    for model in models
    }
    return anomaly_dict

def compute_area_albedo_cube(upwelling_cube, downwelling_cube):
    result = upwelling_cube / downwelling_cube
    result.rename("areaalbedo")
    return result
    
def compute_si_albedo_cube(upwelling_cube, downwelling_cube, siconc_cube, open_water_albedo = 0.07):
    """
    Return a new cube R such that, for each grid-cell and time-step the result is the albedo of the sea ice in that cube:

        R.data = ((100 * upwelling / downwelling) - (100 - siconc) * open_Water_albedo) / siconc

    This version handles the case where cube_c.units might be “unknown”
    by doing the (100 - siconc) step on raw numpy data and re-wrapping it as a
    dimensionless cube.  All three input cubes must lie on the same
    time/latitude/longitude grid (same shape and coordinate points).

    Parameters
    ----------
    cube_a : iris.cube.Cube
        Numerator for the first term (units must be compatible with cube_b).
    cube_b : iris.cube.Cube
        Denominator for the first term (will be converted to cube_a.units).
    cube_c : iris.cube.Cube
        Used both inside (100 - C) and as final divisor. We assume its
        data are dimensionless (or “percent,” but in practice we treat the
        raw values as unitless), so we operate on cube_c.data directly.

    Returns
    -------
    result_cube : iris.cube.Cube
        A new cube, same time/lat/lon coords, whose data are the albedo of the sea ice in that cube
        ((100*upwelling / downwelling) - (100 - siconc)*open_water_albedo) / siconc.  The final unit will be dimensionless.
    """

    if not (upwelling_cube.shape == downwelling_cube.shape == siconc_cube.shape):
        raise ValueError(
            f"Shape mismatch: a={upwelling_cube.shape}, b={downwelling_cube.shape}, c={siconc_cube.shape}"
        )

    for coord in upwelling_cube.coords(dim_coords=True):
        name = coord.name()
        pts_a = upwelling_cube.coord(name).points
        pts_b = downwelling_cube.coord(name).points
        pts_c = siconc_cube.coord(name).points
        if not (np.allclose(pts_a, pts_b) and np.allclose(pts_a, pts_c)):
            raise ValueError(f"Coordinate '{name}' differs between cubes.")

    A = upwelling_cube.copy()
    B = downwelling_cube.copy()
    C = siconc_cube.copy()

    try:
        B.convert_units(A.units)
    except Exception:
        raise ValueError(f"Cannot convert {downwelling_cube.units} into {upwelling_cube.units} to do A/B.")

    term1 = (A / B) * 100.0
 
    c_data = C.data  
    
    raw_term2_data = (100.0 - c_data) * open_water_albedo
    
    term2_cube = C.copy()
    term2_cube.data = raw_term2_data
    term2_cube.units = '1'
    term2_cube.rename("term2 = (100 - siconc) * open_water_albedo")

    numerator = term1 - term2_cube

    result = numerator / C  

    result.rename("sialbedo")

    return result

def compute_monthly_changes(monthly_data):
    """
    Estimate monthly sea ice volume growth from monthly mean values,
    assuming periodic climatology (Dec → Jan included).
    
    Parameters:
        volume_data (array-like): 12 monthly mean values (Jan to Dec)
        
    Returns:
        growth (np.ndarray): 12 monthly growth values (Jan to Dec)
                             representing volume change from start to end of each month
    """
    monthly_data = np.asarray(monthly_data)
    if monthly_data.shape[0] != 12:
        raise ValueError("Expected 12 monthly mean values (Jan to Dec)")

    # Estimate start-of-month volumes (length 12)
    # Each start-of-month value is avg of adjacent monthly means
    # e.g. start of Feb ≈ avg(Jan, Feb)
    start_volumes = 0.5 * (monthly_data + np.roll(monthly_data, 1))

    # Growth for each month = next month's start - this month's start
    growth = np.roll(start_volumes, -1) - start_volumes  # length 12, includes Dec→Jan

    return growth

def combine_variables(nested_dict, variables_to_combine, new_key):
    """
    Combine selected variables in each model's dictionary and store the result under a new key.
    
    Parameters:
    - nested_dict: dict of dicts (model -> variable -> array)
    - variables_to_combine: list of variable names to sum
    - new_key: the key under which the combined sum is stored
    
    Returns:
    - new_dict: the updated dictionary
    """
    new_dict = {}

    for model, var_dict in nested_dict.items():
        combined = sum(var_dict[var] for var in variables_to_combine)

        # Create a new inner dict with combined variables
        updated_var_dict = {k: v for k, v in var_dict.items() if k not in variables_to_combine}
        updated_var_dict[new_key] = combined

        new_dict[model] = updated_var_dict

    return new_dict
