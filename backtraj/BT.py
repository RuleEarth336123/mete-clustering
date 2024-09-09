#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %%
"""
Created on 08/03/2023
@author: jdmantillaq
"""

import numpy as np
import os
import glob
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import pandas as pd
import gc
import seaborn as sns
sns.set(style="whitegrid")
sns.set_context('notebook', font_scale=1.2)


def create_download_path(path):
    # function to create a new directory path
    import os
    os.makedirs(path, exist_ok=True)


def Continentes_lon_lat(ax):
    """
    Creates a plot of continents using longitude and latitude coordinates.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes object
        The axes object on which to create the plot.

    Returns:
    --------
    ax : matplotlib.axes.Axes object
        The axes object with the continent plot added.

    Notes:
    ------
    The function uses the Cartopy library to create the continent plot.
    It loads a map of country borders with a resolution of 1:10m, sets 
    the x and y ticks, adds gridlines and coastlines, adds a layer of color
    to the continent, and adds the borders of countries to the plot.

    Example:
    --------
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax = Continentes_lon_lat(ax)
    plt.show()
    """
    import cartopy
    import cartopy.crs as ccrs
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    import cartopy.feature as cfeature
    import numpy as np

    # load a map of country borders with higher resolution 1:10m
    Borders = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_0_boundary_lines_land',
        scale='110m',
        facecolor='none')

    # set the labels for the x and y axes
    ax.set_xticks(np.arange(-180, 180, 30), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(-90, 91, 15), crs=ccrs.PlateCarree())
    ax.tick_params(axis='both', which='major', labelsize=14, color="#434343")
    lon_formatter = LongitudeFormatter(zero_direction_label=True,
                                       number_format='.0f')
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.set_axisbelow(False)

    # add gridlines to the plot
    ax.grid(which='major', linestyle='--', linewidth='0.6', color='gray',
            alpha=0.8)
    # add coastlines to the plot
    ax.coastlines(resolution='110m', color='k', alpha=0.7, lw=0.5)

    # add a layer of color to the continent
    ax.add_feature(cartopy.feature.LAND, color='silver', alpha=0.2)
    # add the borders of countries to the plot
    ax.add_feature(Borders, edgecolor='gray', facecolor='None',
                   alpha=0.8, lw=0.6)
    # return the plot object
    return ax


def walk_path_target(path, target):
    """
    Finds all files matching the specified target in a given directory and
    its subdirectories,
    and returns the list of file paths sorted in ascending order.

    Parameters:
    -----------
    path : str
        The path to the directory where the files are located.
    target : str
        The file name pattern to match.

    Returns:
    --------
    file_paths : numpy.ndarray
        An array of file paths that match the target pattern, sorted in
        ascending order.

    Example:
    --------
    path = "/path/to/directory"
    target = "*.txt"
    file_paths = walk_path_target(path, target)
    """

    # Search for files matching the target pattern in the directory and
    # its subdirectories
    file_paths = np.sort([y for x in os.walk(path) for y
                          in glob.glob(os.path.join(x[0], target))])

    return file_paths


def find_nearest_idx(array, value):
    """
    Finds the index of the element in a 1D array that is closest
    to a specified value.

    Parameters:
    -----------
    array : numpy.ndarray
        The input 1D array.
    value : float
        The target value to find the nearest index for.

    Returns:
    --------
    idx : int
        The index of the element in the array that is closest
        to the target value.

    Example:
    --------
    array = np.array([1, 2, 3, 4, 5])
    value = 3.7
    idx = find_nearest_idx(array, value)
    """

    # Find the index of the element in the array that is closest
    # to the target value
    idx = np.argmin(np.abs(array - value))
    return idx


def find_nearest_idx_time(array, value):
    """
    Finds the index of the element in a 1D array of datetime objects that is
    closest to a specified datetime.

    Parameters:
    -----------
    array : numpy.ndarray
        The input 1D array of datetime objects.
    value : datetime.datetime
        The target datetime object to find the nearest index for.

    Returns:
    --------
    idx : int or numpy.nan
        The index of the element in the array that is closest to the
        target datetime, or numpy.nan if the
        target datetime is outside the range of the input array.

    Example:
    --------
    import datetime
    array = np.array([datetime.datetime(2022, 1, 1),
            datetime.datetime(2022, 1, 2), datetime.datetime(2022, 1, 3)])
    value = datetime.datetime(2022, 1, 2, 12)
    idx = find_nearest_idx_time(array, value)
    """

    # Check if the target datetime is within the range of the input array
    if value >= array[0] and value <= array[-1]:
        # Find the index of the element in the array that is closest
        # to the target datetime
        idx = np.argmin(np.abs(array - value))
        return idx
    else:
        # If the target datetime is outside the range of the input array,
        # return NaN
        return np.nan


def save_nc(dictionary, file_out):
    """
    Save the data in a dictionary as a netCDF4 file.

    Args:
        dictionary (dict): A dictionary containing the data to be saved. It
            should have the following keys: 'datetime_traj', 'lon_traj',
            'lat_traj', 'plev_traj', and 'sh_traj'.
        file_out (str): The name of the output netCDF4 file.

    Returns:
        None
    """

    # Create a vector of dates
    dates_vector = dictionary['datetime_traj'].reshape(-1)
    [dates_dim, back_step_dim] = dictionary['lon_traj'].shape

    # Define a time reference
    time_ref = '1900-01-01 00:00:00'
    date_ref = pd.to_datetime(time_ref)

    # Get the vector of dates in seconds
    dates = np.zeros_like(dates_vector)*np.nan
    for i, date_i in enumerate(dates_vector):
        try:
            dates[i] = int((date_i - date_ref).total_seconds()/3600)
        except:
            pass

    # Define time reference for netCDF variable
    ncvar_time_units = f'hours since {time_ref}'
    ncvar_time_long_name = f'hours since {time_ref}, ' \
        f'Reshape vector as {dates_dim}x{back_step_dim}'

    # Open a new netCDF file for writing
    nw = Dataset(file_out, 'w', format='NETCDF4')

    # Define dimensions
    dates_vector = dates_dim*back_step_dim
    nw.createDimension('time_bt',  dates_dim)
    nw.createDimension('back_step', back_step_dim)
    nw.createDimension('time_vector', dates_vector)

    # Create variables
    ncvar_time = nw.createVariable('time', 'f', ('time_vector'))
    ncvar_lat = nw.createVariable('lat',  'f', ('time_bt', 'back_step'))
    ncvar_lon = nw.createVariable('lon',  'f', ('time_bt', 'back_step'))
    ncvar_plev = nw.createVariable('level', 'f', ('time_bt', 'back_step'))
    ncvar_sh = nw.createVariable('q', 'f', ('time_bt', 'back_step'))

    # Add attributes
    ncvar_lat.units = 'degrees_north'
    ncvar_lon.units = 'degrees_east'
    ncvar_plev.units = 'hPa'
    ncvar_sh.units = 'kg kg**-1'
    ncvar_time.units = ncvar_time_units

    ncvar_plev.long_name = 'Level'
    ncvar_lon.long_name = 'Longitude'
    ncvar_lat.long_name = 'Latitude'
    ncvar_time.long_name = ncvar_time_long_name
    ncvar_sh.long_name = 'Specific humidity'

    # Write variables to the netCDF file
    ncvar_time[:] = dates
    ncvar_plev[:] = dictionary['plev_traj']
    ncvar_lon[:] = dictionary['lon_traj']
    ncvar_lat[:] = dictionary['lat_traj']
    ncvar_sh[:] = dictionary['sh_traj']

    # Close the netCDF file
    nw.close()
    del(nw)


def read_nc(file_i):
    """
    Function to read netCDF files.

    Args:
        file_i (str): File path of the netCDF file to be read.

    Returns:
        fechas (array): Array of date and time values.
        dates (array): Array of numerical time values.
        plev_values (array): Array of pressure level values.
        lat_values (array): Array of latitude values.
        lon_values (array): Array of longitude values.
        sh_values (array): Array of specific humidity values.
    """
    import numpy as np  # Mathematical operations and matrix handling
    import pandas as pd  # Date handling
    from netCDF4 import Dataset

    Variable = Dataset(file_i, 'r')

    dates = np.array(Variable.variables['time'][:])

    fechas = pd.to_datetime("1900-01-01 00:00:00") \
        + pd.to_timedelta(dates, unit='h')

    lon_values = np.array(Variable.variables['lon'][:])
    lat_values = np.array(Variable.variables['lat'][:])
    sh_values = np.array(Variable.variables['q'][:])
    plev_values = np.array(Variable.variables['level'][:])
    fechas = np.array(fechas).reshape(plev_values.shape)
    dates = dates.reshape(plev_values.shape)

    return fechas, dates, plev_values, lat_values, lon_values, sh_values

# =============================================================================
# Classe BackTrajectories
# =============================================================================


class Follow:
    """
    A class that represents and calculates back trajectories for
    atmospheric particles.
    """

    def __init__(self, path=None, lati=None, loni=None):
        """
        Initializes the Follow object with the given path, initial
        latitude and longitude.

        Args:
            path (str, optional): Path to the data files. Defaults to None.
            lati (float, optional): Initial latitude. Defaults to None.
            loni (float, optional): Initial longitude. Defaults to None.
        """
        if not path:
            print('Warning: File path not specified')
        self.path = path
        self.lati = lati
        self.loni = loni

    def Trajectories_level_i(self, fechai=None, fechaf=None, delta_t=6,
                             ndays=10, level_i=None, *args, **kwargs):
        """
        Calculates back trajectories for a specified pressure level 
        and date range.

        Args:
            fechai (str, optional): Start date for the calculation. 
                    Defaults to None.
            fechaf (str, optional): End date for the calculation.
                    Defaults to None.
            delta_t (int, optional): Time step in hours. Defaults to 6.
            ndays (int, optional): Number of days for each trajectory.
                    Defaults to 10.
            level_i (float, optional): Pressure level for the calculation (hPa)
                    Defaults to None.
        """
        self.level_i = level_i
        self.ndays = ndays
        self.delta_t = delta_t

        if fechai and fechaf:
            self.fechai = pd.to_datetime(fechai)
            self.fechaf = pd.to_datetime(fechaf)
        else:
            print('Note: You must specify the dates for the calculation.')

        self.datetimes_iter = pd.date_range(self.fechai, self.fechaf,
                                            freq=f'{self.delta_t}H')

        '''
        # Reserve space for the BT
        ----------------------------------------------------------------------
        dates_dim:      dimension are the dates from which the trajectories
                        are to be calculated.
        back_step_dim:  time length dimension of each backtrajectory
        '''
        [dates_dim, back_step_dim] = [len(self.datetimes_iter),
                                      int((24/self.delta_t)*self.ndays)]

        self.BT = {}
        self.BT['lat_traj'] = np.zeros([dates_dim, back_step_dim])
        self.BT['lon_traj'] = np.zeros([dates_dim, back_step_dim])
        self.BT['plev_traj'] = np.zeros([dates_dim, back_step_dim])
        self.BT['sh_traj'] = np.zeros([dates_dim, back_step_dim])
        self.BT['datetime_traj'] = \
            np.zeros([dates_dim, back_step_dim]).astype(object)
        self.BT['steps_traj'] = np.zeros([dates_dim, back_step_dim])

        print('Computing BT:')
        print('.'*85)

        for di, dti in enumerate(self.datetimes_iter):
            self.BT['lat_traj'][di, :], \
                self.BT['lon_traj'][di, :],\
                self.BT['plev_traj'][di, :], \
                self.BT['sh_traj'][di, :], \
                self.BT['datetime_traj'][di, :],\
                self.BT['steps_traj'][di, :] = \
                self.compute_BT(datetime0=dti, lat0=self.lati,
                                lon0=self.loni, plev0=self.level_i,
                                ndays=self.ndays, delta_t=self.delta_t)

    def compute_BT(self, *args, **kwargs):
        """
        Calculate the back trajectory of air parcels given initial conditions.

        Args:
            plev0 (float): Initial pressure level in hPa for the calculation.
            datetime0 (datetime): Initial datetime for the calculation.
            delta_t (float): Time step in hours, maximum 4 days.
            lat0 (float): Initial latitude in degrees for the calculation.
            lon0 (float): Initial longitude in degrees for the calculation.

        Returns:
            lat_traj (np.ndarray): Latitude trajectory of air parcel.
            lon_traj (np.ndarray): Longitude trajectory of air parcel.
            plev_traj (np.ndarray): Pressure level trajectory of air parcel.
            sh_traj (np.ndarray): Specific humidity trajectory of air parcel.
            datetime_traj (np.ndarray): Datetime trajectory of air parcel.
            steps_traj (np.ndarray): Trajectory steps.
        """

        # Get initial conditions from input arguments
        datetime0 = kwargs.get('datetime0', None)
        plev0 = kwargs.get('plev0', None)
        lat0 = kwargs.get('lat0', None)
        lon0 = kwargs.get('lon0', None)

        # Calculate approximate u, v, omega velocities for datetime0 and plev0
        u0, v0, w0, sh0 = self.data_from_nc(plev0, datetime0, lat0, lon0)

        # Allocate space for storing the information
        N_hours_back = int(self.ndays*24)
        Dim_iteration = int(self.ndays*24/self.delta_t)

        lat_traj = np.zeros(Dim_iteration)*np.nan
        lon_traj = np.zeros(Dim_iteration)*np.nan
        plev_traj = np.zeros(Dim_iteration)*np.nan
        sh_traj = np.zeros(Dim_iteration)*np.nan
        steps_traj = np.zeros(Dim_iteration)*np.nan
        datetime_traj = np.zeros(Dim_iteration).astype(object)*np.nan

        # Initialize trajectory with initial conditions
        lat_traj[0] = lat0
        lon_traj[0] = lon0
        plev_traj[0] = plev0
        sh_traj[0] = sh0
        steps_traj[0] = 0
        datetime_traj[0] = datetime0

        # Temporary variables to store information during iterations
        lon_temp = lon0
        lat_temp = lat0
        plev_temp = plev0
        datetime_temp = datetime0
        u_temp = u0
        v_temp = v0
        w_temp = w0

        # Iterate through time steps to calculate back trajectory
        for ii, steps_back in enumerate(range(self.delta_t, N_hours_back,
                                              self.delta_t)):
            # Calculate spatial position for the next time step
            #       X(t_1) = X(t_0)+(delta_t)*V(t_0)

            lat_temp, lon_temp, plev_temp = \
                self.compute_location(lat0=lat_temp, lon0=lon_temp,
                                      plev0=plev_temp, u=u_temp, v=v_temp,
                                      w=w_temp, delta_t=self.delta_t)

            # Update datetime
            datetime_temp = datetime_temp - pd.Timedelta(f"{self.delta_t}h")

            # Adjust longitude if it goes beyond limits
            if lon_temp < -180:
                lon_temp += 360
            if lon_temp > 180:
                lon_temp -= 360

            # Recalculate velocity components for next iteration
            u_temp, v_temp, w_temp, sh_temp = self.data_from_nc(plev_temp,
                                                                datetime_temp,
                                                                lat_temp,
                                                                lon_temp)

            # Check for missing data or out-of-bounds conditions, and
            # break loop if needed
            if np.isnan(u_temp):
                break

            if ((lon_temp > np.nanmax(self.lon_dataset)) |
                    (lon_temp < np.nanmin(self.lon_dataset))):
                break

            elif ((lat_temp > np.nanmax(self.lat_dataset)) |
                    (lat_temp < np.nanmin(self.lat_dataset))):
                break

            # Update trajectory information
            else:
                lat_traj[ii+1] = lat_temp
                lon_traj[ii+1] = lon_temp
                plev_traj[ii+1] = plev_temp
                sh_traj[ii+1] = sh_temp
                steps_traj[ii+1] = steps_back
                datetime_traj[ii+1] = datetime_temp

        return lat_traj, lon_traj, plev_traj, sh_traj,\
            datetime_traj, steps_traj

    def compute_location(self, lat0, lon0, plev0, u, v, w, delta_t,
                         BT=True, *args, **kwargs):
        """
        Compute the new location of the air parcel after a time step using
        the zero acceleration solution to the differential
        trajectory equation  Stohl (1998):
        
            X(t_1) = X(t_0)+(delta_t)*V(t_0)
            
        Andreas Stohl, Computation, accuracy and applications of 
        trajectories—A review and bibliography, Atmospheric Environment, 
        Volume 32, Issue 6, 1998, Pages 947-966, ISSN 1352-2310,
        https://doi.org/10.1016/S1352-2310(97)00457-3.

        Args:
            lat0 (float): Initial latitude in degrees.
            lon0 (float): Initial longitude in degrees.
            plev0 (float): Initial pressure level in hPa.
            u (float): Zonal wind velocity in m/s.
            v (float): Meridional wind velocity in m/s.
            w (float): Vertical wind velocity in Pa/s.
            delta_t (float): Time step in hours.
            BT (bool, optional): If True, compute back trajectory. Otherwise,
            compute forward trajectory.
                Defaults to True.

        Returns:
            new_latitude (float): New latitude of air parcel
            new_longitude (float): The new longitude of the air parcel
                        after the given time step.
            plev (float): The new pressure level of the air parcel after 
                        the given time step.
        """
        BT = kwargs.get('BT', True)

        # Forward Trajectories
        dx = u*delta_t*60*60       # meters
        dy = v*delta_t*60*60       # meters
        dz = (w/100)*delta_t*60*60  # hPa

        plev = plev0+(-dz if BT else dz)
        r_earth = 6378000
        if BT:
            # Converting meters to degrees
            new_latitude = lat0 - (dy / r_earth) * (180 / np.pi)
            new_longitude = lon0 - (dx / r_earth) * (180 / np.pi) \
                / np.cos(lat0 * np.pi/180)
        else:
            new_latitude = lat0 + (dy / r_earth) * (180 / np.pi)
            new_longitude = lon0 + (dx / r_earth) * (180 / np.pi) \
                / np.cos(lat0 * np.pi/180)
        if plev > 1000:
            plev = 1000
        return new_latitude, new_longitude, plev

    def Plot_Trajectories(self, *args, **kwargs):
        """
        This function plots the trajectories of a dataset on a map using the
        cartopy library. It can plot the trajectories as a scatter plot with
        color mapping or as simple lines. The function also allows
        customization of the map extent, color map, and other plot features.

        Parameters:
        *args: Additional arguments for the function.
        **kwargs: Keyword arguments for the function, including:
            figsize (tuple): Size of the figure (default: (10, 6)).
            img_extent (tuple): Spatial extent of the map
                        (default: (-95, -30, -20, 30)).
            plot_scatter (bool): Whether to plot scatter plot with color
                        mapping (default: True).
            level_scatter (tuple): Range of values for color mapping
                        (default: (0, 0.012)).
            cmap (str): Color map for the scatter plot (default: 'viridis').

        Returns:
        tuple: A tuple containing the figure (fig) and axes (ax) of the plot.
        """

        import cartopy.crs as ccrs

        figsize = kwargs.get('figsize', (10, 6))
        img_extent = kwargs.get('img_extent', (-95, -30, -20, 30))
        plot_scatter = kwargs.get('plot_scatter', True)
        level_scatter = kwargs.get('level_scatter', (0, 0.012))
        cmap = kwargs.get('cmap', 'viridis')
        (vmin, vmax) = level_scatter

        fig = plt.figure(figsize=(figsize))
        proj = ccrs.PlateCarree(central_longitude=0)
        ax = fig.add_axes([0, 0, 1, 1], projection=proj)
        ax = Continentes_lon_lat(ax)
        ax.set_extent(img_extent, ccrs.PlateCarree())

        (dates_dim, back_step_dim) = self.BT['lon_traj'].shape
        if plot_scatter:
            for di in range(dates_dim):
                c = ax.scatter(self.BT['lon_traj'][di, :],
                               self.BT['lat_traj'][di, :],
                               c=self.BT['sh_traj'][di, :],
                               cmap=cmap,
                               vmin=vmin, vmax=vmax)

            plt.colorbar(c, label='kg kg**-1')
        else:
            for di in range(dates_dim):
                ax.plot(self.BT['lon_traj'][di, :],
                        self.BT['lat_traj'][di, :])
        ax.scatter(self.loni, self.lati, marker='*', c='firebrick', s=80)

        return fig, ax

# A subclass is created to read the particular data from ERA5 and how it is
# stored.


class Follow_ERA5(Follow):
    """
    A class to follow the path of a trajectory in the ERA5 reanalysis data
    """

    vars_ERA5 = {'uwnd': 'u',
                 'vwnd': 'v',
                 'sh': 'q',
                 'omega': 'w'}
    time_ref = '1900-01-01 00:00:00.0'
    time_units = 'h'

    def __init__(self, path=None, lati=None, loni=None):
        """
        Initialize the `Follow_ERA5` class

        Parameters
        ----------
        path : str, optional
            Path to the ERA5 data, by default None
        lati : float, optional
            Latitude of the point to follow, by default None
        loni : float, optional
            Longitude of the point to follow, by default None
        """
        super().__init__(path, lati, loni)

        # Identification of the file paths
        lista = walk_path_target(path, '*.nc')
        columns = ['var', 'level', 'date']
        df = pd.DataFrame([i.split('/')[-1][:-3].split('_') for i in lista],
                          columns=columns)
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        df['level'] = df['level'].apply(lambda x: int(x[:3]))
        df['path'] = lista

        # Dataset properties
        print('\nLoading properties from:'
              f'\n\t{df.loc[0, "path"]}')
        Variable = Dataset(df.loc[0, 'path'], 'r')

        self.lat_dataset = np.array(Variable.variables['latitude'][:])
        self.lon_dataset = np.array(Variable.variables['longitude'][:])
        dates_ref = np.array(Variable.variables['time'])

        dates = pd.to_datetime(self.time_ref) \
            + pd.to_timedelta(dates_ref,
                              unit=self.time_units)
        self.hour_ref = np.array(dates.hour)
        del(Variable)
        gc.collect()

        self.data_base = df
        self.levels_dataset = np.unique(df['level'])

    def data_from_nc(self, level, dti, lati, loni):
        """
        Returns the u, v, omega and specific humidity data for a given
        date, latitude, longitude, and level.

        Parameters
        ----------
        level : int
            Vertical level
        dti : datetime
            Date and time
        lati : float
            Latitude
        loni : float
            Longitude

        Returns
        -------
        tuple
            Tuple containing the u, v, omega, and specific humidity data
        """

        mask = (pd.to_datetime(dti.date()) >= self.data_base.date) & \
            (pd.to_datetime(dti.date()) <= self.data_base.date)
        pos_level = find_nearest_idx(self.levels_dataset, level)
        level = self.levels_dataset[pos_level]

        data = {}
        houri = int(dti.hour)

        for var_i in self.vars_ERA5.keys():
            row = self.data_base[mask & (self.data_base['var'] == var_i) &
                                 (self.data_base['level'] == level)]

            if len(row) > 0:
                pos_lat = find_nearest_idx(self.lat_dataset, lati)
                pos_lon = find_nearest_idx(self.lon_dataset, loni)
                pos_hour = find_nearest_idx(self.hour_ref, houri)
                Variable = Dataset(row['path'].iloc[0], 'r')

                data[var_i] = np.array(
                    Variable.variables[self.vars_ERA5[var_i]][pos_hour,
                                                              pos_lat,
                                                              pos_lon])
            elif var_i == 'omega':
                data[var_i] = 0
            else:
                data[var_i] = np.nan

        return data['uwnd'], data['vwnd'], data['omega'], data['sh']


if __name__ == "__main__":

    # Path where ERA5 data is stored
    path_files = 'ERA5_data/'

    # Route where the backtrajectories will be stored per day.
    path_out = 'processed_bt/'

    # Calculation start coordinates
    lati = 6.25184
    loni = -75.56359

    # Start level of the calculation
    level = [825]  # hPa

    # Calculation properties
    ndays = 10  # number of days on which the calculation will be made
    delta_t = 6   # hours

    # Dates on which the calculation is to be performed
    timerange = pd.date_range('2022-01-01', '2022-12-31', freq='d')

    # The db (database) object is loaded with the data needed to perform the
    # calculations. The properties are extracted from the read files.
    db = Follow_ERA5(path=path_files, lati=lati, loni=loni)

    # The .nc files can be viewed in this DataFrame
    db.data_base

    # the calculation of the BT will be done for each day.
    # For example, January 1, 2022, is taken, for this day 4 backtrajectories
    # will be calculated, starting independently at 00:00, 06:00, 12:00 and
    # 18:00, with a time resolution of 6 hours and a travel time of 10 days.
    for i, date_i in enumerate(timerange):

        for level_i in level:
            # File name, corresponding to the day being evaluated
            date_name = f'{date_i.strftime("%Y%m%d")}'

            # limited to one day of calculation
            fechai = date_i.strftime('%Y-%m-%d 00:00')
            fechaf = date_i.strftime('%Y-%m-%d 23:59')
            print(f'{fechai} -----> {fechaf}')
            name_file = f'BT.{delta_t}h.{level_i}hPa.{date_name}.nc'

            # Backtrajectories are calculated for a particular day.
            db.Trajectories_level_i(fechai=fechai, fechaf=fechaf,
                                    delta_t=delta_t, ndays=ndays,
                                    level_i=level_i)

            # For a particular day, the BT dictionary contains the spatial
            # information of the 4 backtrajectories.
            db.BT

            # The back trajectory of a particular day can be plotted
            db.Plot_Trajectories(plot_scatter=True)

            file_out = f'{path_out}{name_file}'

            #  A BT file is saved with the trajectory information.
            if not os.path.exists(file_out):
                save_nc(dictionary=db.BT, file_out=file_out)
                print(f'\t{name_file}: OK')
            print('-'*85)

# %%
