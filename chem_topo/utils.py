import numpy as np
import scipy
from scipy.optimize import least_squares,minimize
from ase.data import atomic_numbers, covalent_radii,vdw_radii

def objective(x, points, weights, radius):
    """
    This function calculates the difference between the distance from a point `x` 
    to a set of given points `points` and the target radius, weighted by `weights`,
    for use in least squares optimization.
    """
    # Compute the Euclidean distance between the point `x` and each point in the `points` array
    distances = np.linalg.norm(points - x, axis=1)

    # Calculate the weighted squared difference between the distances and the target radius
    return weights * ((distances - radius) ** 2)
              
                                    
def calculate_centroid(points, cov_radii, radius):
    """
    Finding the outer center (centroid) of atomic combinations.

    Parameters
    ----------
    - points : ndarray
        The Cartesian coordinates of atomic combinations.
    - cov_radii : ndarray
        The covalent radii of the atoms, used as weights for optimization.
    - radius : float
        The target distance from the outer center to each point.

    Returns
    -------
    - coordinates of the centroid : ndarray
        The coordinates of the calculated centroid.
    """

    # Calculate weights based on the covalent radii normalized by their sum
    weights = cov_radii / np.sum(cov_radii)

    # Compute an initial guess for the centroid as the mean of the input points
    initial_guess = np.mean(points, axis=0)

    # Perform least-squares optimization to find the centroid
    result = least_squares(
        lambda x: np.sqrt(objective(x, points, weights, radius)),  # Minimize the square root of the objective function
        initial_guess,                                             # Initial guess for the optimization
        method="lm",                                               # Use the Levenberg-Marquardt algorithm
        max_nfev=50,                                               # Limit the number of function evaluations to 50
        ftol=1e-4                                                  # Set the tolerance for the optimization
    )

    # Return the optimized coordinates of the centroid
    return result.x


def get_cutoffs(atoms, metal_elements, mult=1.1):
    """
    Calculate cutoff distances for atoms based on their types and a scaling multiplier.

    Parameters:
    ------------
    - atoms : ASE Atoms object
        List of atoms in the structure.
    - metal_elements : list 
        List of elements considered as metals.
    - mult : float 
        Scaling multiplier for the cutoff distances (default is 1.1).

    Returns:
    ---------
    - cutoffs :  list of float
        List of calculated cutoff distances.
    """

    cutoffs = []  # Initialize a list to store cutoff distances
    for atom in atoms:
        symbol = atom.symbol  # Get the atomic symbol of the current atom
        if symbol in metal_elements:
            # Use the covalent radius for metal elements
            cutoff = covalent_radii[atomic_numbers[symbol]]
        else:
            # Use the van der Waals radius for non-metal elements
            cutoff = vdw_radii[atomic_numbers[symbol]]
        
        # Handle missing or invalid data by setting a default value
        if cutoff is None or cutoff <= 0:
            cutoff = 1.5  # Default value, can be adjusted as needed

        # Scale the cutoff distance by the multiplier
        cutoff *= mult
        cutoffs.append(cutoff)  # Append the scaled cutoff to the list

    return cutoffs  # Return the list of cutoff distances

#Set the metal atoms.
metal_elements = {
        "Li", "Be", "Na", "Mg", "Al", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", 
        "Ni", "Cu", "Zn", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", 
        "Cd", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", 
        "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", 
        "Hg", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", 
        "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", 
        "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
                }

import random
import numpy as np 
def calculate_distance(cell,point1, point2,pbc):
    """
    The function to calculated distance 
        for dealing with periodicity

    Parameters:
    -----------
    - cell : ndarray
        The cell in input atoms.
    - points1 , points2 : ndarray
        The positions to calculate distance with periodicity 
        
    Returns:
    ---------
    - dis : float
        The periodic distance between two points.
    """
    # Check if periodic boundary conditions are applied in any direction
    if pbc.any() == True:
        a_vec = cell[0]  # First lattice vector (a)
        b_vec = cell[1]  # Second lattice vector (b)
        dis = 10.  # Initialize a large distance (arbitrary value)
        neighbors = []  # List to store neighboring points
        
        # Generate neighbors by shifting point1's position in the range of -1, 0, 1 in both directions
        for da in range(-1, 2):
            for db in range(-1, 2):
                neighbor_position = point1['position'] + da * a_vec + db * b_vec  # Calculate neighbor position
                neighbors.append(neighbor_position)  # Append to neighbors list
        
        # Find the closest neighbor to point2 using the distance metric
        for n in neighbors:
            if dis > np.linalg.norm(n - point2['position']):
                dis = np.linalg.norm(n - point2['position'])  # Update distance if a closer neighbor is found
        
        return dis  # Return the minimum distance considering periodic boundary conditions
    else:
        # If no periodicity is applied, simply return the Euclidean distance
        return np.linalg.norm(point1['position'] - point2['position'])
    
def calculate_atom_distance(cell, point1, point2, pbc):
    """
    The function to calculated distance for dealing with periodicity

    Parameters:
    -----------
    - cell : ndarray
        The cell in input atoms.
    - points1 , points2 : ndarray
        The positions to calculate distance with periodicity 
        
    Returns:
    ---------
    - dis : float
        The periodic distance between two points.
    """
    # Check if periodic boundary conditions are applied in any direction
    if pbc.any() == True:
        a_vec = cell[0]  # First lattice vector (a)
        b_vec = cell[1]  # Second lattice vector (b)
        dis = 10.  # Initialize a large distance (arbitrary value)
        neighbors = []  # List to store neighboring points
        
        # Generate neighbors by shifting point1's position in the range of -1, 0, 1 in both directions
        for da in range(-1, 2):
            for db in range(-1, 2):
                neighbor_position = point1 + da * a_vec + db * b_vec  # Calculate neighbor position
                neighbors.append(neighbor_position)  # Append to neighbors list
        
        # Find the closest neighbor to point2 using the distance metric
        for n in neighbors:
            if dis > np.linalg.norm(n - point2):  # Compare distance with the current minimum distance
                dis = np.linalg.norm(n - point2)  # Update distance if a closer neighbor is found
        
        return dis  # Return the minimum distance considering periodic boundary conditions
    else:
        # If no periodicity is applied, simply return the Euclidean distance between the two points
        return np.linalg.norm(point1['position'] - point2['position'])

    
def select_points(points, num_points, min_distance, cell, pbc):
    """
    This function is used to search for and select adsorption sites from a list of potential points.

    Parameters:
    -----------
    - points : list of ndarray
        All possible potential sites from which adsorption points will be selected.

    - num_points : int
        The number of adsorbents (points) to select.

    - min_distance : float
        The minimum distance required between two neighboring adsorbents.

    - cell : ndarray
        The unit cell matrix of the input atomic structure.

    - pbc : ndarray
        The periodic boundary conditions (True or False for each direction).

    Returns:
    --------
    - selected_points : list of ndarray
        A list of selected adsorption points that satisfy the distance criteria.
    """
    
    selected_points = []  # Initialize an empty list to store selected adsorption points
    
    # Continue selecting points until the desired number of adsorbents is reached
    while len(selected_points) < num_points:
        point = random.choice(points)  # Randomly select a point from the list of potential points
        
        # Check if the selected point is at a valid distance from all previously selected points
        if all(calculate_distance(cell, point, selected_point, pbc) >= min_distance for selected_point in selected_points):
            selected_points.append(point)  # If valid, append the point to the list of selected points
    
    return selected_points  # Return the list of selected points


from networkx.algorithms import isomorphism
from ase.neighborlist import natural_cutoffs, NeighborList
import networkx as nx
def is_unique(graph, unique_graphs):
    """
    Determine if the current input graph is unique

    Parameters:
    -----------
    - graph : networkx.Graph object
        The graph for determining uniqueness

    - unique_graphs : list of networkx.Graph objects
        List of saved graphs to compare with the current graph
    """
    
    # If no unique graphs have been stored, the current graph is considered unique
    if unique_graphs == []:
        return True
    
    # Check if the current graph is isomorphic to any of the previously stored unique graphs
    for i, unique_graph in enumerate(unique_graphs):
        GM = isomorphism.GraphMatcher(graph, unique_graph, node_match=isomorphism.categorical_node_match('symbol', ''))
        
        # If a match is found (graphs are isomorphic), the current graph is not unique
        if GM.is_isomorphic():
            return False
    
    # If no match is found, the current graph is unique
    return True


def get_graph(atoms):
    """
    Generate a graph representation of the atomic structure.

    Parameters:
    -----------
    - atoms : ASE Atoms object
        The atomic structure to be converted into a graph.

    Returns:
    --------
    - graph : networkx.Graph object
        The generated graph of the atomic structure, representing atomic connectivity.
    """
    
    # Check if periodic boundary conditions (PBC) are applied
    if atoms.pbc.any() == True:
        # If PBC is applied, calculate the periodic graph
        cutoffs = natural_cutoffs(atoms, mult=1.2)  # Get cutoff distances for atomic interactions
        positions = atoms.get_positions()  # Get the positions of atoms
        cell = atoms.get_cell()  # Get the unit cell
        G = nx.Graph()  # Initialize a new graph object
        symbols = atoms.symbols  # Get the atomic symbols
        
        # Add nodes to the graph, each node represents an atom with its symbol
        G.add_nodes_from([(i, {'symbol': symbols[i]}) for i in range(len(symbols))])
        
        # Iterate over all pairs of atoms to check their distances and add edges if they are within the cutoff range
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                distance = calculate_atom_distance(cell=cell, point1=positions[i], point2=positions[j], pbc=atoms.pbc)
                if distance < cutoffs[i] + cutoffs[j]:  # If distance is within the cutoff, add an edge
                    G.add_edge(i, j)
        
        # Apply the Weisfeiler-Lehman (WL) graph kernel (not defined in the provided code)
        graph = wl(G)
        return G  # Return the graph

    else:
        # If PBC is not applied, calculate the non-periodic graph
        cutoffs = natural_cutoffs(atoms, mult=1.2)  # Get cutoff distances for atomic interactions
        nl = NeighborList(cutoffs, self_interaction=False, bothways=True)  # Create a NeighborList
        nl.update(atoms)  # Update the neighbor list based on the current atomic structure
        matrix = nl.get_connectivity_matrix(sparse=False)  # Get the connectivity matrix (adjacency matrix)
        
        G = nx.Graph()  # Initialize a new graph object
        symbols = atoms.symbols  # Get the atomic symbols
        
        # Add nodes to the graph, each node represents an atom with its symbol
        G.add_nodes_from([(i, {'symbol': symbols[i]}) for i in range(len(symbols))])
        
        # Add edges to the graph based on the connectivity matrix (atoms that are neighbors)
        rows, cols = np.where(matrix == 1)  # Get the indices of the connected atoms
        edges = zip(rows.tolist(), cols.tolist())  # Create edges between the connected atoms
        G.add_edges_from(edges)  # Add the edges to the graph
        
        # Apply the Weisfeiler-Lehman (WL) graph kernel (not defined in the provided code)
        graph = wl(G)
        return graph  # Return the graph

def wl(graph):
    """
    Convert an input graph into a Weisfeiler-Lehman (WL) graph by performing a WL conversion.

    Parameters:
    -----------
    - graph : networkx.Graph object
        The input graph to be converted.

    Returns:
    --------
    - new_graph : networkx.Graph object
        The transformed graph after applying the Weisfeiler-Lehman algorithm.
    """
    
    # Get the 'symbol' attribute for each node in the graph
    node_symbols = nx.get_node_attributes(graph, 'symbol')
    
    num_iterations = 3  # Define the number of iterations for the WL algorithm
    
    # Perform the Weisfeiler-Lehman algorithm for a specified number of iterations
    for _ in range(num_iterations):
        new_symbols = {}  # A dictionary to store new symbols for the nodes
        
        # Iterate through all nodes in the graph
        for node in graph.nodes():
            symbol = node_symbols[node]  # Get the current symbol for the node
            # Get the symbols of the neighboring nodes
            neighbor_symbols = [node_symbols[neighbor] for neighbor in graph.neighbors(node)]
            # Combine the node's symbol with its neighbors' symbols and sort them
            combined_symbol = symbol + ''.join(sorted(neighbor_symbols))
            new_symbols[node] = combined_symbol  # Store the new symbol for the node
        
        # Update node_symbols with the new symbols after this iteration
        node_symbols = new_symbols
    
    # Create a new graph with the updated node symbols
    new_graph = nx.Graph()
    
    # Add nodes to the new graph with their new symbols
    for node, symbol in node_symbols.items():
        new_graph.add_node(node, symbol=symbol)
    
    return new_graph  # Return the newly created Weisfeiler-Lehman graph

def plane_normal(xyz):
    """
    Return the surface normal vector to a plane of best fit.

    Parameters:
    -----------
    - xyz : ndarray (n, 3)
        3D points to fit the plane to.

    Returns:
    --------
    - vec : ndarray (1, 3)
        Unit vector normal to the plane of best fit.
    """
    
    # Set up the design matrix A with the x, y coordinates and a column of ones (for the intercept)
    A = np.c_[xyz[:, 0], xyz[:, 1], np.ones(xyz.shape[0])]
    
    # Solve for the coefficients of the plane using least squares to fit the z-values
    vec, _, _, _ = scipy.linalg.lstsq(A, xyz[:, 2])
    
    # Set the z-component of the normal vector to -1 (align with the z-axis)
    vec[2] = -1.0
    
    # Normalize the vector to make it a unit vector
    vec /= -np.linalg.norm(vec)
    
    return vec  # Return the normalized normal vector
