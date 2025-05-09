import numpy as np

import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from chem_topo.topo_features import PathHomology


max_path = 2
cloudpoints = np.array([
    [-1.1, 0, 0],
    [1.2, 0, 0],
    [0, -1.3, 0],
    [0, 1.4, 0],
    [1.5, 0, 0],
    [-1.6, 0, 0]
])
points_weight = [1, 2, 9, 4, 5, 6]
betti_nums = PathHomology(max_distance=5.0).persistent_path_homology(
        cloudpoints, points_weight, max_path, filtration=None)
betti_nums = PathHomology().persistent_angle_path_homology(
        cloudpoints, points_weight, max_path)
betti_nums = PathHomology().persistent_homology(cloudpoints,max_path)