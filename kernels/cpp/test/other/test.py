from joblib import Parallel, delayed
from tqdm import tqdm
import ctypes
import numpy as np
import time

# Load the shared library
distance_lib = ctypes.CDLL('cpp/test/compu.so')

# Define the Point struct
class Point(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), 
                ("y", ctypes.c_double)]

# Define the function signature
distance_lib.distance_sse.argtypes = [Point, Point]
distance_lib.distance_sse.restype = ctypes.c_double

# Define the distance function
def distance(p1, p2):
    return distance_lib.distance_sse(p1, p2)

# Define the function to calculate the distance between two trajectories
def distance_between_trajectories(distances):
    return sum(distances)

# Generate some random trajectories
num_trajectories = 10000
num_points = 48
trajectories = [np.random.rand(num_points, 2) for _ in range(num_trajectories)]

# Prepare the tasks
tasks = [(distance(p1, p2) for p1, p2 in zip(trajectories[i], trajectories[j])) for i in range(num_trajectories) for j in range(i+1, num_trajectories)]

# Start timing
start_time = time.time()

# Create a progress bar
pbar = tqdm(total=len(tasks))

# Define a callback function to update the progress bar
def update(*a):
    pbar.update()

# Calculate the distances in parallel and update the progress bar
distances = Parallel(n_jobs=-1, callback=update)(delayed(distance_between_trajectories)(task) for task in tasks)

# Close the progress bar
pbar.close()

# End timing
end_time = time.time()

# Print the elapsed time
print("Elapsed time: ", end_time - start_time, "seconds")
