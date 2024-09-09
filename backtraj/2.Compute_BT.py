# %%
import numpy as np  # Mathematical operations and matrix handling
import matplotlib.pyplot as plt  # Graphics
import pandas as pd  # Date handling
import os
import glob
from netCDF4 import Dataset
from BT import *

'''
This code uses the BT library. Here the necessary functions and classes
are imported to calculate the backtrajectories.

It is necessary to change the file paths, calculation dates, and
starting coordinates for the backtrajectory calculations for the wanted
location.
'''

# Path where ERA5 data is stored
path_files = 'ERA5_data/'

# Path where the daily backtrajectories will be saved.
path_out = 'processed_bt/'
# Create the directory if it doesn't exist
create_download_path(path_out)

# Starting coordinates for the calculation
lati = 6.25184
loni = -75.56359

# Starting level for the calculation
level = [825]  # hPa

# Calculation properties
ndays = 10  # number of days for the calculation
delta_t = 6   # hours

# Dates for the calculations
# These are the dates that should be changed for the desired study dates
timerange = pd.date_range('2022-01-01', '2023-02-', freq='d')

# The db (database) object is loaded with the necessary data to perform
# the calculations. The properties are extracted from the read files
db = Follow_ERA5(path=path_files, lati=lati, loni=loni)

# The .nc files can be queried in this DataFrame
db.data_base

plot_trajectories = False

# The BT calculation will be done for each day
# For example, take January 1, 2022, for this day there will be
# 4 backtrajectories, started independently at
# 00:00, 06:00, 12:00, and 18:00, with a time resolution of 6 hours and a
# travel time of 10 days.
for i, date_i in enumerate(timerange):

    for level_i in level:
        # Filename, corresponding to the day evaluated
        date_name = f'{date_i.strftime("%Y%m%d")}'

        name_file = f'BT.{delta_t}h.{level_i}hPa.{date_name}.nc'
        file_out = f'{path_out}{name_file}'
        if os.path.exists(file_out):
            continue

        # limit to one day of calculation
        fechai = date_i.strftime('%Y-%m-%d 00:00')
        fechaf = date_i.strftime('%Y-%m-%d 23:59')
        print(f'{fechai} -----> {fechaf}')

        # Calculate backtrajectories for a particular day
        db.Trajectories_level_i(fechai=fechai, fechaf=fechaf, delta_t=delta_t,
                                ndays=ndays, level_i=level_i)

        # For a particular day, the BT dictionary contains the spatial
        # information of the 4 backtrajectories
        db.BT

        if plot_trajectories:
            # You can plot the backtrajectory for a particular day
            db.Plot_Trajectories(plot_scatter=True)

        # Save a BT file with the backtrajectory information
        save_nc(dictionary=db.BT, file_out=file_out)
        print(f'\t{name_file}: OK')
        print('-'*85)
