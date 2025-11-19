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


import itertools

months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

def plot_data(data_dict, title, ax, xlabel = None, ylabel = None, color=None, plot_sum=False, labels=None):
    """Plot the monthly data stored in *data_dict* on *ax*."""
    t = np.arange(0, 12)
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    
    data_sum = np.zeros_like(t, dtype=float)
    i=0
    for var, data in data_dict.items():
        if labels:
            label = labels[i]
        else:
            label = var
        ax.plot(t, data, label=label, color=color)
        data_sum += data
        i+=1
    if plot_sum:
        ax.plot(t, data_sum, label='Total', color='k')

    ax.set_title(title)
    ax.grid()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(t)
    ax.set_xticklabels(months)


def plot_grid(*dicts, save_name=None, ylabels=None, colors=None, plot_sums=None, labels=None):
    

    if not dicts:
        raise ValueError("At least one data dictionary must be provided")

    models = list(dicts[0].keys())
    n_models = len(models)
    n_dicts = len(dicts)

    legend_adjust = -0.1 + 0.005 *  n_models
    
    if ylabels is None:
        ylabels = [''] * n_dicts
    if colors is None:
        colors = [None] * n_dicts
    if plot_sums is None:
        plot_sums = [False] * n_dicts

    fig, axes = plt.subplots(n_models, n_dicts,
                             figsize=(n_dicts * 6, n_models * 3),
                             constrained_layout=True)
    axes = np.atleast_2d(axes)

    for row, model in enumerate(models):
        for col, d in enumerate(dicts):
            ax = axes[row, col]
            if labels:
                label_list = labels[col]
            else:
                label_list=None
            plot_data(d[model], model, ax=ax, ylabel=ylabels[col],
                      color=colors[col], plot_sum=plot_sums[col], labels=label_list)
            
    
    # # Add a separate legend for each column
    for col in range(n_dicts):
        handles, labels = axes[0, col].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=min(3,len(labels)),
                   bbox_to_anchor=((col + 0.5) / n_dicts, legend_adjust),
                   bbox_transform=fig.transFigure, frameon=True)


    
    #plt.subplots_adjust(bottom=0.15)
    #plt.tight_layout()
    if save_name is not None:
        plt.savefig('/home/users/matt/plots/' + save_name + '.png',
                    bbox_inches='tight')
    plt.show()

def scatter_models(x, y, labels, xlabel=None, ylabel=None, title = None, save_name = None, x0 = None):
    #plots each data point as a different color, and creates a legend
    #Suitable for plotting different models, one data point per model
    #Plots a linear regression with 95% confidence interval.

    #if an x0 is provided, the expected y0 is calculated, and the PDF plotted

    #statistics are returned in a dictionary
    stats_dict = {}
    
    #linear regression and confidence intervals
    X = sm.add_constant(x)          
    model = sm.OLS(y, X).fit()      

    x_pred = np.linspace(x.min(), x.max(), 100)
    X_pred = sm.add_constant(x_pred)
    pred = model.get_prediction(X_pred)
    pred_summary = pred.summary_frame(alpha=0.05)  

    
    unique_labels = sorted(set(labels))
    color_list = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    color_map = dict(zip(unique_labels, color_list))
    
    point_colors = [color_map[L] for L in labels]

    
    plt.scatter(x, y, c=point_colors, s=100, edgecolor='k')
    
    # build custom legend handles
    handles = [Line2D([0], [0],
                      marker='o',
                      color='w',
                      markerfacecolor=color_map[L],
                      markeredgecolor='k',
                      markersize=10)
               for L in unique_labels]


    # regression line
    plt.plot(x_pred, pred_summary["mean"], color='black', lw=2, label="Fit")
    
    # 95% confidence band on the mean prediction
    plt.fill_between(x_pred,
                     pred_summary["mean_ci_lower"],
                     pred_summary["mean_ci_upper"],
                     color='black',
                     alpha=0.3,
                     label="95% CI")

    #plt.xlim(left=0)   
    #plt.ylim(bottom=0)
    
    plt.legend(handles,
           unique_labels,
           title="Model",
           loc='center left',
           bbox_to_anchor=(1.02, 0.5),
           borderaxespad=0.)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    if not save_name == None:
        plt.savefig('/home/users/matt/plots/'+save_name+'.png', bbox_inches='tight')
    plt.show()

        
    stats_dict['r_squared'] = model.rsquared
    stats_dict['intercept'], stats_dict['slope'] = model.params
    
    # Get fitted values (the model's prediction at each x)
    y_fitted = model.fittedvalues
    # Calculate RMSE
    rmse = np.sqrt(np.mean((y - y_fitted) ** 2))
    stats_dict['rmse'] = rmse
    
    if x0 is not None:
        # Get prediction results at x0
        X0 = np.array([[1.0, x0]])   # shape is now (1, 2): [const, x0]          
        pr = model.get_prediction(X0).summary_frame(alpha=0.05).iloc[0]

        mu = pr["mean"]                       # predicted mean
        se_mean = pr["mean_se"]               # standard error of the mean
        sigma_resid = np.sqrt(model.mse_resid)
        sigma_pred = np.sqrt(se_mean**2 + sigma_resid**2)

        # Build grid around the mean
        y_vals = np.linspace(mu - 4*sigma_pred,
                             mu + 4*sigma_pred, 200)
        
        #pdf = stats.norm.pdf(y_vals, loc=mu, scale=sigma_pred)

        df = model.df_resid
        pdf = stats.t.pdf(y_vals, df=df, loc=mu, scale=sigma_pred)
        
        # Plot
        plt.figure(figsize=(7,5))
        plt.plot(y_vals, pdf, lw=2)
        plt.title(f"PDF at {x0:.2f}")
        plt.xlabel(ylabel)
        plt.ylabel("Density")
        plt.grid(True, linestyle='--', alpha=0.5)
        if not save_name == None:
            plt.savefig('/home/users/matt/plots/'+save_name+'_pdf.png', bbox_inches='tight')
        plt.show()

        stats_dict['mean_at_x0'] = mu
        stats_dict['sd_at_x0'] = sigma_pred
        stats_dict['pdf'] = pdf


    
    return stats_dict
    
def plot_arctic_monthly_maps(monthly_data, vmin = None, vmax = None, cbar_label = None, cmap = cmocean.cm.ice, gamma = 1.0, save_name = None, labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], months_to_plot = range(12)):

    # Create a figure with 12 subplots in a 3x4 grid, each with a North Polar Stereo projection.
    
    if vmin is None:
        vmin = np.min(monthly_data.data)
    if vmax is None:
        vmax =np.max(monthly_data.data)

    norm = colors.PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)
    
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 10),
                             subplot_kw={'projection': ccrs.NorthPolarStereo()})
    axes = axes.flatten()
    
    # Create a circular path in axes coordinates.
    theta = np.linspace(0, 2 * np.pi, 100)
    x = 0.5 + 0.5 * np.cos(theta)  # center=0.5, radius=0.5
    y = 0.5 + 0.5 * np.sin(theta)
    circle_path = mpath.Path(np.column_stack([x, y]))
    
    # Loop over each month/subplot.
    for m in months_to_plot:
        ax = axes[m]
    
        # Set the geographic extent.
        ax.set_extent([-180, 180, 60, 90], crs=ccrs.PlateCarree())
    
        # Set a circular boundary for the subplot.
        ax.set_boundary(circle_path, transform=ax.transAxes)
    
        # Extract the data for the current month.
        plot_data = monthly_data[m]
    
        # Plot the data with a fixed color scale (vmin = -100, vmax = 0) and a chosen colormap.
        # The transform argument ensures that the data in PlateCarree coordinates is plotted correctly.
    
        qplt.contourf(plot_data, axes=ax, vmin=vmin, vmax=vmax, cmap=cmap, norm=norm)
    
        # Add coastlines and gridlines.
        ax.coastlines()
        ax.gridlines()
        # Add land.
        ax.add_feature(cfeature.LAND, color='#a9a9a9', zorder=4)
        # Set the subplot title to the month name.
        ax.set_title(labels[m])
    
    for a in fig.axes.copy():
        if not hasattr(a, 'projection'):
            fig.delaxes(a)
    
    # Adjust the layout so there is space at the bottom for a single common colorbar.
    plt.tight_layout(rect=[0, 0.1, 1, 1])  # Leaves a margin at the bottom (0.1 of figure height)
    
    # Create a ScalarMappable with the same colormap and normalization used in the contour plots.
    #norm = colors.Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # This line is needed for older matplotlib versions
    
    # Add a common horizontal colorbar at the bottom of the figure.
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.05, pad=0.05)
    cbar.set_label(cbar_label)

    if not save_name == None:
            plt.savefig('/home/users/matt/plots/'+save_name+'.png', bbox_inches='tight')
    plt.show()

def plot_arctic_monthly_maps_contour(
    monthly_data,
    vmin=None,
    vmax=None,
    cbar_label=None,
    cmap=cmocean.cm.ice,
    gamma=1.0,
    save_name=None,
    labels=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],
    months_to_plot=range(12),
    *,
    contour=15,                 # <— NEW: pass a numeric level to plot a single contour
    contour_kwargs=None,          # <— NEW: dict for contour line styling
    clabel=True                   # <— NEW: label the contour value on maps
):

    if contour_kwargs is None:
        contour_kwargs = dict(colors='k', linewidths=1.5)

    # Defaults for filled plot
    if vmin is None:
        vmin = np.nanmin(monthly_data.data)
    if vmax is None:
        vmax = np.nanmax(monthly_data.data)

    norm = colors.PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(
        nrows=3, ncols=4, figsize=(15, 10),
        subplot_kw={'projection': ccrs.NorthPolarStereo()}
    )
    axes = axes.flatten()

    # Circular boundary (polar cap look)
    theta = np.linspace(0, 2*np.pi, 100)
    x = 0.5 + 0.5*np.cos(theta)
    y = 0.5 + 0.5*np.sin(theta)
    circle_path = mpath.Path(np.column_stack([x, y]))

    # Plot each requested month
    for m in months_to_plot:
        ax = axes[m]
        ax.set_extent([-180, 180, 60, 90], crs=ccrs.PlateCarree())
        ax.set_boundary(circle_path, transform=ax.transAxes)

        plot_data = monthly_data[m]

        # Single contour line at the requested level
        # qplt.contour returns a QuadContourSet; use levels=[contour]
        cs = qplt.contour(plot_data, axes=ax, levels=[contour], **contour_kwargs)
        if clabel:
            ax.clabel(cs, fmt=f"{contour:g}")

        ax.coastlines()
        ax.gridlines(draw_labels=False)
        ax.add_feature(cfeature.LAND, color='#a9a9a9', zorder=4)
        ax.set_title(labels[m])

    # Remove any leftover blank axes (if months_to_plot is a subset)
    for a in fig.axes.copy():
        if not hasattr(a, 'projection'):
            fig.delaxes(a)

    plt.tight_layout()

    if save_name is not None:
        plt.savefig(f'/home/users/matt/plots/{save_name}.png', bbox_inches='tight')
    plt.show()


def plot_hovmoller(zonal_cube, save_name=None, cmap='viridis', colorbar_label = None, title = None, vmin=None, vmax=None):
    #monthly_cube = cube.aggregated_by('month', iris.analysis.MEAN)
    #zonal_cube = monthly_cube.collapsed('longitude', iris.analysis.MEAN)
    # Extract data
    lat = zonal_cube.coord('latitude').points
    time = zonal_cube.coord('time')
    data = zonal_cube.data  # shape: (time, lat)
    
    plt.figure(figsize=(10, 5))
    contour = plt.contourf(range(len(time.points)), lat, data.T, levels=20, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(label=(colorbar_label or f"{zonal_cube.units}"))
    plt.xlabel("Month")
    plt.ylabel("Latitude")
    plt.title(title or f"Hovmöller Diagram (Zonal Mean of {zonal_cube.name()})")
    plt.xticks(ticks=np.arange(12), labels=months)
    plt.tight_layout()
    if not save_name == None:
        plt.savefig('/home/users/matt/plots/'+save_name+'.png', bbox_inches='tight')
    plt.show()

def plot_multi_model_mean(nested_dict, variable, title=None, ylabel=None, legend=True, save_name=None):
    """
    Plot a specified variable for all models, plus the multi-model mean.

    Parameters:
    - nested_dict: dict of dicts (model -> variable -> array)
    - variable: str, the variable name to plot
    - title: optional title for the plot
    - ylabel: optional y-axis label
    - legend: whether to show the legend (default: True)
    """
    plt.figure(figsize=(10, 6))

    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    n_models = len(nested_dict)
    cmap = plt.cm.get_cmap('tab10' if n_models <= 10 else 'tab20', n_models) 
    
    model_data = []

    for i, (model, var_dict) in enumerate(nested_dict.items()):
        if variable not in var_dict:
            print(f"Warning: {variable} not found in {model}")
            continue
        data = var_dict[variable]
        color = cmap(i)
        model_data.append(data)
        plt.plot(data, label=model, color=color, linestyle='--')

    
    # Plot multi-model mean
    if model_data:
        model_array = np.array(model_data)
        mean = np.mean(model_array, axis=0)
        plt.plot(mean, label='Multi-model mean', color='black', linewidth=2)

    # Set x-axis ticks to month names
    plt.xticks(ticks=np.arange(12), labels=months)

    # Labels and legend
    plt.title(title or f"{variable} across models")
    plt.xlabel('Month')
    plt.ylabel(ylabel or variable)
    if legend:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid(True)
    if not save_name == None:
        plt.savefig('/home/users/matt/plots/'+save_name+'.png', bbox_inches='tight')
    plt.show()
    return mean










    