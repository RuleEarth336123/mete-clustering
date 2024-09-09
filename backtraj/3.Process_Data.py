# %%

import numpy as np  # Mathematical operations and matrix handling
import matplotlib.pyplot as plt  # Graphics
import pandas as pd  # Date handling
import os
import glob
import cartopy.crs as ccrs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from BT import *


'''
Once all the desired backtrajectories have been calculated,
they can be processed.

This code will help you read all the generated information.
'''

# Path where the calculated BT files are stored for the specific model
path_files = 'processed_bt/'

path_fig = 'figures/'

archivos = np.sort(glob.glob(os.path.join(path_files, 'BT*.nc')))

# %%

files_date = pd.to_datetime([i.split('\\')[-1].split('.')[-2]
                             for i in archivos], format='%Y%m%d')

# %%

# Clustering will be done only with 10 days (10*4)
num_days = 10
num_bt_dia = 4   # Number of BTs per day
dim_bt = num_days * num_bt_dia  # Time dimension of each BT

# Initialize arrays for latitude, longitude, specific humidity, and dates
lat = np.zeros((len(archivos), num_bt_dia, dim_bt)) * np.nan
lon = np.copy(lat)
sh = np.copy(lat)
dates = np.copy(lat)


for i, file_i in enumerate(archivos):
    try:
        fechas, dates_values, plev_values, lat_values, lon_values, sh_values =\
            read_nc(file_i)
    except:
        continue

    lat[i, :, :] = lat_values[:, :dim_bt]
    lon[i, :, :] = lon_values[:, :dim_bt]
    sh[i, :, :] = sh_values[:, :dim_bt]
    dates[i, :, :] = dates_values[:, :dim_bt]
    del(lat_values, lon_values, fechas, plev_values, dates_values)


lat = lat.reshape((len(archivos)*num_bt_dia, dim_bt))
lon = lon.reshape((len(archivos)*num_bt_dia, dim_bt))
sh = sh.reshape((len(archivos)*num_bt_dia, dim_bt))
dates = dates.reshape((len(archivos)*num_bt_dia, dim_bt))

# averiguar las posiciones con NaNs
idx_nan_array = np.where(np.isnan(lat[:, -1]))[0]

# Se eliminan las posiciones con los NaNs
lat = np.delete(lat, idx_nan_array, axis=0)
lon = np.delete(lon, idx_nan_array, axis=0)
sh = np.delete(sh, idx_nan_array, axis=0)
dates = np.delete(dates, idx_nan_array, axis=0)


fechas = pd.to_datetime("1900-01-01 00:00:00") \
    + pd.to_timedelta(dates.reshape(-1), unit='h')
dayofyear = np.array(fechas.dayofyear).reshape(lat.shape)

mean_lat, std_lat = np.mean(lat), np.std(lat)
mean_lon, std_lon = np.mean(lon), np.std(lon)
mean_sh, std_sh = np.mean(sh), np.std(sh)
mean_doy, std_doy = np.mean(dayofyear), np.std(dayofyear)


lat_std = (lat-mean_lat)/(std_lat)
lon_std = (lon-mean_lon)/(std_lon)
sh_std = (sh-mean_sh)/(std_sh)
doy_st = (dayofyear-mean_doy)/(std_doy)

# %%

# Starting coordinates for the calculation
lati = 6.25184
loni = -75.56359

figsize = (10, 6)
img_extent = (-95, -30, -20, 30)
cmap = sns.color_palette("viridis", as_cmap=True)
(vmin, vmax) = (0, 0.012)

fig = plt.figure(figsize=(figsize))
proj = ccrs.PlateCarree(central_longitude=0)
ax = fig.add_axes([0, 0, 1, 1], projection=proj)
ax = Continentes_lon_lat(ax)
ax.set_extent(img_extent, ccrs.PlateCarree())

(dates_dim, back_step_dim) = lon.shape

np.random.seed(28)
randint = np.random.randint(0, dates_dim, 75)

for i, di in enumerate(randint):  
    ax.plot(lon[di, :], lat[di, :], color='black', alpha=0.3, zorder=0)

for i, di in enumerate(randint):
    c = ax.scatter(lon[di, :], lat[di, :], c=sh[di, :], cmap=cmap,
                   vmin=vmin, vmax=vmax)

plt.colorbar(c, label='kg kg**-1')

ax.scatter(loni, lati, marker='*', c='firebrick', s=80)

plt.savefig(f'{path_fig}backtrajectories.png',
            dpi=150, bbox_inches='tight', pad_inches=0,
            transparent=False,
            facecolor='white')

# %%

# %%


matrix = np.hstack([lat_std, lon_std, sh_std, doy_st])
for jj, n_cluster in enumerate(np.arange(4, 9)):

    km1 = KMeans(n_clusters=n_cluster, random_state=25).fit(matrix)
    labels = km1.labels_

    labels_2 = np.array([labels]).T*np.ones((len(labels), dim_bt))

    colors = sns.color_palette("husl", n_cluster)
    Num_Fil = int(np.ceil(n_cluster/2))

    fig = plt.figure(figsize=(11.5, int(14.8/4*Num_Fil)))
    proj = ccrs.PlateCarree(central_longitude=0)

    img_extent = (-120, 20, -35, 35)

    Num_Col = 2

    # Variables en X
    x_bor_i = 0.01
    x_bor_r = 0.03
    x_inter = 0.00
    x_fig = (1 - (x_bor_i + x_bor_r + x_inter))/Num_Col
    x_fig2 = 0.38*x_fig
    x_space_f2 = 0.01*x_fig

    def x_corner(x): return x_bor_i + (x-1)*(x_fig + x_inter)

    def x_corner2(x): return x_corner(x) + (x_fig-x_fig2)

    # Coordenada X esquina inferior izquierda del axes_i
    x_col1 = x_corner(1)
    x_col2 = x_corner(2)

    # Variables en Y
    y_sup = 0.03
    y_inf = 0.03
    y_text = 0.008
    y_inter = 0.01
    y_fig = (1 - (y_sup + y_inf + (Num_Fil - 1)*y_inter))/Num_Fil

    y_fig2 = 0.32*y_fig
    y_space_f2 = 0.085*y_fig

    # Coordenada Y esquina inferior izquierda del axes_i
    y_rows = np.flip([y_inf + i*(y_fig + y_inter) for i in range(Num_Fil)])

    for i in range(Num_Fil):
        ii = i*Num_Col

        if ii+1 <= n_cluster:
            id_cluster = np.where(labels == ii)[0]

            ax = fig.add_axes([x_col1, y_rows[i], x_fig, y_fig],
                              projection=proj)

            plt.text(0.02, 0.97, f'Cluster {ii+1}',
                     va='top', ha='left', backgroundcolor='w',
                     transform=ax.transAxes, fontsize=18, alpha=0.75)

            ax = Continentes_lon_lat(ax)
            ax.set_extent(img_extent, ccrs.PlateCarree())
            ax.set_aspect('auto')
            ax.set_xticklabels([])
            ax.set_yticklabels([])

            for k, j in enumerate(id_cluster):
                plt.plot(lon[j], lat[j], c=colors[ii], lw=0.4)
            plt.plot(lon[id_cluster].mean(0),
                     lat[id_cluster].mean(0), c='k', lw=0.8)

    for i in range(Num_Fil):
        ii = i*Num_Col+1
        if ii+1 <= n_cluster:
            id_cluster = np.where(labels == ii)[0]
            ax = fig.add_axes([x_col2, y_rows[i], x_fig, y_fig],
                              projection=proj)
            plt.text(0.02, 0.97, f'Cluster {ii+1}',
                     va='top', ha='left', backgroundcolor='w',
                     transform=ax.transAxes, fontsize=18, alpha=0.75)
            ax = Continentes_lon_lat(ax)
            ax.set_extent(img_extent, ccrs.PlateCarree())
            ax.set_aspect('auto')
            ax.set_xticklabels([])
            ax.set_yticklabels([])

            for k, j in enumerate(id_cluster):
                plt.plot(lon[j], lat[j], c=colors[ii], lw=0.4)
            plt.plot(lon[id_cluster].mean(0),
                     lat[id_cluster].mean(0), c='k', lw=0.8)

    plt.savefig(f'{path_fig}{n_cluster:02}_cluster.png',
                dpi=150, bbox_inches='tight', pad_inches=0,
                transparent=False,
                facecolor='white')
