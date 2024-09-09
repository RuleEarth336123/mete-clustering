# Lagrangian Backtrajectories with ERA5 Data

This repository contains Python code to calculate backtrajectories using a Lagrangian method based on ERA5 reanalysis data.
The script relies on the custom BT library to perform the necessary calculations and save the output as NetCDF files.

<img src="figures/backtrajectories.png" alt="bactrajectories" width="600"/>
<img src="figures/05_cluster.png" alt="cluster_5" width="600"/>


## Lagrangian trajectories Algorithm 

The backtrajectory calculation algorithm used in this repository is based on the Lagrangian method proposed by Stohl (1998). This method computes the new location of an air parcel after a time step using the zero acceleration solution to the differential trajectory equation. The equation is as follows:

$$
X(t_1) = X(t_0) + (\Delta t) \cdot V(t_0)
$$

A smaller `delta_t` (time step) generally leads to a lower error in the backtrajectory calculation. However, smaller time steps may also increase the computational time. It is recommended to balance the choice of `delta_t` based on the desired accuracy and available computational resources.

**Reference:**
Andreas Stohl, Computation, accuracy and applications of trajectories—A review and bibliography, Atmospheric Environment, Volume 32, Issue 6, 1998, Pages 947-966, ISSN 1352-2310, https://doi.org/10.1016/S1352-2310(97)00457-3.

## Data Source
The data used in this code is from the ERA5 dataset, which is a global atmospheric reanalysis dataset provided by the European Centre for Medium-Range Weather Forecasts (ECMWF). ERA5 provides high-resolution, consistent and quality-controlled data for a wide range of atmospheric variables, such as wind components, specific humidity, and vertical velocity. The data is available from 1979 to present and has a spatial resolution of approximately 31 km on 137 pressure levels.


## Usage
1. **Download ERA5 Data**

    The first script `1.Download_ERA5`.py downloads the required ERA5 data for a region over South America and the Caribbean using the Climate Data Store API and stores it in a folder (e.g., `ERA5_data/`). The script also contains utility functions for walking through directories, creating new directory paths, and downloading the ERA5 data.

    **Variables**:
    -   `variables`: A dictionary with variable names and their corresponding parameter values.
    -   `level`: The pressure level to be downloaded.
    -   `directory`: The directory path to save the downloaded data.
    -   `time_range`: The time range for which data needs to be downloaded.
    -   `lat_min`, `lat_max`, `lon_min`, `lon_max`: The geographic limits for downloading the information.
    -   `delta_t`: The time step for downloading the information.

    **Usage**

    1. Set the desired variable names and their corresponding parameter values in the variables dictionary.
    2. Set the pressure level(s) to be downloaded in the level list.
    3. Specify the directory path to save the downloaded data in directory.
    4. Set the time range for which data needs to be downloaded in time_range.
    5. Define the geographic limits for downloading the information by setting lat_min, lat_max, lon_min, and lon_max.
    6. Set the time step (`delta_t`) for downloading the information.
    7. Run the `1.Download_ERA5.py` script to download the ERA5 data for the specified region, pressure levels, and date range. The data will be saved in the specified directory with one folder per year.

    After running the 1.Download_ERA5.py script, you will have the necessary ERA5 data saved in the specified folder (e.g., `ERA5_data/`). You can then proceed to run the `2.Compute_BT.py` script to calculate the backtrajectories using the downloaded ERA5 data.

2. **Compute Backtrajectories**

    The second script `2.Compute_BT.py` calculates backtrajectories for each day in the specified date range. The backtrajectories are initiated every `delta_t` hours, resulting in multiple backtrajectories per day. The backtrajectory data is saved as NetCDF files in the output folder specified by path_out. Optionally, the script can also plot the calculated backtrajectories.

    **Usage**

    Download ERA5 data for the desired dates and store it in a folder (e.g., `ERA5_data/`). Set the output folder for the processed backtrajectory data (e.g., `processed_bt/`). Update the starting coordinates (`lati`, `loni`), level (`level`), and date range (`timerange`) for the backtrajectory calculations. Run the script to perform backtrajectory calculations for the specified dates and starting location.

    **Configuration**
    -   `path_files`: The folder containing the input ERA5 data.
    path_out: The folder where the daily backtrajectory data will be saved.
    -   `lati`, `loni`: Starting latitude and longitude coordinates for the calculation.
    -   `level`: Starting pressure level for the calculation (in hPa).
    -   `ndays`: Number of days for the backtrajectory calculation.
    -   `delta_t`: Time resolution for the backtrajectory calculation (in hours).
    -   `timerange`: Date range for the backtrajectory calculations.

    **Example**:

    For a sample date range from January 15th to January 20th, 1980, with starting coordinates (6.25184, -75.56359), a pressure level of 825 hPa, a time resolution of 6 hours, and a backtrajectory duration of 10 days, the script calculates backtrajectories for each day and saves the output as NetCDF files in the specified output folder.

## Clustering Backtrajectories

The third script, `3.Process_Data.py`, applies clustering algorithms to the computed backtrajectories to identify common patterns or clusters in the data. This script can be used to examine air mass pathways over time, and can help provide insights into potential sources of air pollution, among other things.



<img src="figures/06_cluster.png" alt="cluster_6" width="500"/>



## Dependencies
To run the script, the following Python libraries are required:

- numpy
- pandas
- matplotlib
- netCDF4
- BT (custom library)
- cdsapi
- seaborn
- cartopy
- os
- glob
- gc


Make sure to install these libraries before running the script. You can install them using pip:
```pip install numpy pandas cdsapi netCDF4 matplotlib seaborn cartopy```

## CDS API Configuration
Before running `1.Download_ERA5.py`, configure the Climate Data Store API (`cdsapi`) to download ERA5 data by following the instructions at https://cds.climate.copernicus.eu/api-how-to.

## License
This project is licensed under the MIT License - see the `LICENSE` file for details.



