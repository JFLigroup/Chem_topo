import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from chem_topo.topo_features import SimplicialComplexLaplacian
import numpy as np
def main():
    aa = SimplicialComplexLaplacian()  # Create an instance of the class.
    # Define an adjacency matrix for the graph (several examples are provided, here the second example is used).
    adjacency_matrix = np.array([
        [0, 1, 1, 0],
        [1, 0, 1, 0],
        [1, 1, 0, 1],
        [0, 0, 1, 0],
    ])
    # Convert the adjacency matrix to a simplicial complex (clique complex).
    ww = aa.adjacency_map_to_simplex(adjacency_matrix, max_dim=2)
    print(ww)
    # Compute the Laplacian eigenvalues of the simplicial complex.
    feat = aa.simplicialComplex_laplacian_from_connected_mat(adjacency_matrix, max_dim=2)
    print(feat)
    # Print the 0-dimensional Laplacian matrix (which relates to vertex connectivity).
    print(aa.laplacian_matrix_dict[0])

    # Compare with the classical graph Laplacian.
    print(np.diag(np.sum(adjacency_matrix, axis=0)) - adjacency_matrix)
    return None

if __name__ == "__main__":
    main()