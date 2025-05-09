"""
Different persistent homology methods to obtain features.

The PathHomology class is used to extract features for continuous homology, 
distance-based path homology, and angle-based path homology.
The SimplicialComplexLaplacian class is used to implement feature extraction 
of the persistent topological hyperdigraph Laplacian.

References
----------
.. [1] Chen, Dong, Jian Liu, and Guo-Wei Wei. "Multiscale topology-enabled structure-to-sequence 
       transformer for protein–ligand interaction predictions." *Nature Machine Intelligence* (2024): 1–12.
.. [2] Persistent path topology in molecular and materials sciences.
.. [3] GUDHI Library. https://gudhi.inria.fr/
"""


CORDERO = {'Ac': 2.15, 'Al': 1.21, 'Am': 1.80, 'Sb': 1.39, 'Ar': 1.06,
           'As': 1.19, 'At': 1.50, 'Ba': 2.15, 'Be': 0.96, 'Bi': 1.48,
           'B' : 0.84, 'Br': 1.20, 'Cd': 1.44, 'Ca': 1.76, 'C' : 0.76,
           'Ce': 2.04, 'Cs': 2.44, 'Cl': 1.02, 'Cr': 1.39, 'Co': 1.50,
           'Cu': 1.32, 'Cm': 1.69, 'Dy': 1.92, 'Er': 1.89, 'Eu': 1.98,
           'F' : 0.57, 'Fr': 2.60, 'Gd': 1.96, 'Ga': 1.22, 'Ge': 1.20,
           'Au': 1.36, 'Hf': 1.75, 'He': 0.28, 'Ho': 1.92, 'H' : 0.31,
           'In': 1.42, 'I' : 1.39, 'Ir': 1.41, 'Fe': 1.52, 'Kr': 1.16,
           'La': 2.07, 'Pb': 1.46, 'Li': 1.28, 'Lu': 1.87, 'Mg': 1.41,
           'Mn': 1.61, 'Hg': 1.32, 'Mo': 1.54, 'Ne': 0.58, 'Np': 1.90,
           'Ni': 1.24, 'Nb': 1.64, 'N' : 0.71, 'Os': 1.44, 'O' : 0.66,
           'Pd': 1.39, 'P' : 1.07, 'Pt': 1.36, 'Pu': 1.87, 'Po': 1.40,
           'K' : 2.03, 'Pr': 2.03, 'Pm': 1.99, 'Pa': 2.00, 'Ra': 2.21,
           'Rn': 1.50, 'Re': 1.51, 'Rh': 1.42, 'Rb': 2.20, 'Ru': 1.46,
           'Sm': 1.98, 'Sc': 1.70, 'Se': 1.20, 'Si': 1.11, 'Ag': 1.45,
           'Na': 1.66, 'Sr': 1.95, 'S' : 1.05, 'Ta': 1.70, 'Tc': 1.47,
           'Te': 1.38, 'Tb': 1.94, 'Tl': 1.45, 'Th': 2.06, 'Tm': 1.90,
           'Sn': 1.39, 'Ti': 1.60, 'Wf': 1.62, 'U' : 1.96, 'V' : 1.53,
           'Xe': 1.40, 'Yb': 1.87, 'Y' : 1.90, 'Zn': 1.22, 'Zr': 1.75}  # Atomic radii from Cordero 
ELEM_EN = {'C': 2.55, 'H': 2.2, 'O': 3.44, 'N': 3.04, 
           'S': 2.58, 'Ag': 1.93, 'Au': 2.4, 'Cd': 1.69, 
           'Co': 1.88, 'Cu': 1.9, 'Fe': 1.83, 'Ir': 2.2, 
           'Ni': 1.91, 'Os': 2.2, 'Pd': 2.2, 'Pt': 2.2, 
           'Rh': 2.28, 'Ru': 2.2, 'Zn': 1.65} # electronegativities from mendeleev

import numpy as np
import copy

from ase import Atoms

class PathHomology(object):

    def __init__(self, initial_axes=None, cell = None,pbc = [False,False,False],max_distance=5.0,angle_step=30,max_path=2):
        self.initial_axes = initial_axes
        self.cell = cell
        self.pbc = pbc
        self.max_distance = max_distance
        self.angle_step = angle_step
        self.max_path = max_path
        self.initial_vector_x = np.array([1, 0, 0])
        self.initial_vector_y = np.array([0, 1, 0])
        self.initial_vector_z = np.array([0, 0, 1])
        self.save_temp_result = False
        return None
    

    @staticmethod
    def periodic_distance(coord1, coord2, cell_matrix, pbc):
        """
        Calculate the minimum image distance between two coordinates under Periodic Boundary Conditions (PBC).
        This method is particularly useful for crystal structure simulations where atoms can interact 
        with their periodic images in adjacent unit cells.
        
        Parameters:
        -----------
        - coord1,coord2 : np.ndarray 
                Cartesian coordinates of first point (3D vector).

        - cell_matrix : np.ndarray
                3x3 matrix representing unit cell vectors as rows
                                    [[a_x, a_y, a_z],
                                    [b_x, b_y, b_z],
                                    [c_x, c_y, c_z]]
        - pbc : array
                Boolean array of length 3 indicating which dimensions have PBC enabled
                            [x_pbc, y_pbc, z_pbc]

        Returns:
        --------
        - float    
                Minimum distance between the points considering periodic images
        """
        delta = coord1 - coord2
        inv_cell = np.linalg.inv(cell_matrix.T)
        delta_frac = np.dot(delta, inv_cell)
        for i in range(3):
            if pbc[i]:
                delta_frac[i] -= np.round(delta_frac[i])
        delta_abs = np.dot(delta_frac, cell_matrix.T)
        return np.linalg.norm(delta_abs)
    
    @staticmethod
    def vector_angle(v0, v1):
        """
        Calculate the angle between vector v0 and vector v1 in degree.

        Parameters:
        -----------
        - v0  : array 
                n dimension vector, n >= 2
        - v1  : array
                n dimension vector, n >= 2

        Returns:
        --------
        - angle : int
                angle in degree.
        """
        v0_u = v0 / np.linalg.norm(v0)
        v1_u = v1 / np.linalg.norm(v1)
        angle = np.degrees(np.arccos(np.clip(np.dot(v0_u, v1_u), -1.0, 1.0)))
        return angle

    @staticmethod
    def remove_loops(edges):
        """
        Remove the loops of the digraph.

        Parameters:
        -----------
        - edges : array
                shape = [n, 2]

        Returns:
        --------
        - edges : array
                shape = [n-m, 2], m is the number of the loops
        """
        loop_idx = []
        loop_nodes = []
        for i, e in enumerate(edges):
            if e[0] == e[1]:
                loop_idx.append(i)
                loop_nodes.append(e[0])
        if len(loop_nodes) > 0:
            print(f'Warning, loops on node {loop_nodes} were removed.')
        edges = np.delete(edges, loop_idx, axis=0)
        return edges

    @staticmethod
    def split_independent_compondent(edges, nodes):
        """
        If the digraph is not fully connected, then splitting it into independent components.
        Using the depth first search (DFS) algorithms to split the undirected graph.

        Parameters:
        -----------
        - edges : array
                shape = [n, 2]
        
        - nodes : array
                shape = [k, ], k is the number of the whole graph

        Returns:
        --------
        - all_components : list
                the nodes' set of independent components
        """
        # convert into str
        node_map_idx = {node: idx for idx, node in enumerate(nodes)}

        # adjacency list of the graph
        graph = [[] for i in range(len(nodes))]
        for i, one_edge in enumerate(edges):
            u, v = one_edge
            # Assuming graph to be undirected.
            graph[node_map_idx[u]].append(v)
            graph[node_map_idx[v]].append(u)

        # components list
        all_components = []
        visited = [False for n in nodes]

        def depth_first_search(node, component):
            # marking node as visited.
            visited[node_map_idx[node]] = True

            # appending node in the component list
            component.append(node)
            # visiting neighbours of the current node
            for neighbour in graph[node_map_idx[node]]:
                # if the node is not visited then we call dfs on that node.
                if visited[node_map_idx[neighbour]] is False:
                    depth_first_search(neighbour, component)
            return None

        for i, one_node in enumerate(nodes):
            if visited[i] is False:
                component = []
                depth_first_search(one_node, component)
                all_components.append(component)

        return all_components

    @staticmethod
    def split_independent_digraph(all_components, edges):
        """
        If the digraph is not fully connected, then splitting it into independent components.
        Using the depth first search (DFS) algorithms to split the undirected graph.

        Parameters:
        -----------
        - all_components : list
                the nodes' set of independent components
                edges (array): shape = [n, 2]

        Returns:
        --------
        - all_digraphs : list
                a list of digraphs, each digraph contains a list of edges.
        """
        all_digraphs = [[] for i in all_components]
        edges_visited = [False for i in edges]
        for i_c, component in enumerate(all_components):
            for i_e, edge in enumerate(edges):
                if (edges_visited[i_e] is False) and (edge[0] in component or edge[1] in component):
                    all_digraphs[i_c].append(edge)
                    edges_visited[i_e] = True
            if len(component) == 1 and np.shape(all_digraphs[i_c])[0] < 1:
                all_digraphs[i_c].append(component)

        return all_digraphs

    def utils_generate_allowed_paths(self, edges, max_path):
        """
        Generate the allowed paths of the digraph.

        Parameters:
        -----------
        - edges : array
                shape = [n, 2], int or str

        Returns:
        --------
        - allowed_path_str : dict
                all paths from dimension 0 to dimension max_path
        """
        # digraph info
        nodes = np.unique(edges)
        nodes_num = len(nodes)
        nodes_idx_map = {node: idx for idx, node in enumerate(nodes)}

        # edges matrix, start->end = row->column
        edge_matrix = np.zeros([nodes_num, nodes_num])
        for i, edge in enumerate(edges):
            edge_matrix[nodes_idx_map[edge[0]], nodes_idx_map[edge[1]]] = 1

        # path_0 = vertex set
        allowed_path = {0: [np.array([n]) for n in nodes]}
        allowed_path_str = {0: [str(n) for n in nodes]}

        # path_(1 to max_path)
        for i in range(0, max_path+1):
            allowed_path[i+1] = []
            allowed_path_str[i+1] = []
            for path_previous in allowed_path[i]:
                for node in nodes:
                    if edge_matrix[nodes_idx_map[path_previous[-1]], nodes_idx_map[node]]:
                        new_path = np.append(path_previous, node)
                        allowed_path[i+1].append(new_path)
                        allowed_path_str[i+1].append('->'.join([str(one_node) for one_node in new_path]))
                    else:
                        continue
        return allowed_path_str

    def utils_unlimited_boundary_operator(self, allowed_path, max_path):
        """
        Generate the n-th boundary matrix for mapping (n+1)-path to (n)-path.

        Parameters:
        -----------
        - allowed_path : list
                list of (n+1)-paths, each path stores in array
        - max_path : int
                n should >= 1, n is the dimension of the boundary matrix

        Returns:
        --------
        - unlimited_boundary_mat : array
                the matrix representation of the n-th boundary matrix
        """
        # For D_0, matrix is [0]*len(nodes)
        boundary_map_matrix = {0: np.zeros([len(allowed_path[0]), ])}
        boundary_mat_matrix_rank = {0: 0}
        allowed_path_idx_argument = {0: [1]*len(allowed_path[0])}

        for n in range(1, max_path+2):
            boundary_map_dict = {}
            boundary_operated_path_name_collect = []

            allowed_path_n_types = len(allowed_path[n])
            if allowed_path_n_types == 0:
                boundary_map_matrix[n] = np.zeros([1, len(allowed_path[n-1])])
                boundary_mat_matrix_rank[n] = 0
                allowed_path_idx_argument[n] = [1] * len(allowed_path[n-1])
                break

            for i_path, path in enumerate(allowed_path[n]):

                # split the path into nodes with idx
                path_node_idx = path.split('->')

                # record the result path after boundary operation
                boundary_operated_path_info = {}
                for i_kill in range(n+1):
                    # kill the  i_kill-th vertex
                    temp_path = np.delete(path_node_idx, i_kill)
                    temp_path_str = '->'.join([str(pp) for pp in temp_path])
                    boundary_operated_path_info[temp_path_str] = (-1)**(i_kill)

                    # record all possible n_path
                    boundary_operated_path_name_collect.append(temp_path_str)
                boundary_map_dict[path] = copy.deepcopy(boundary_operated_path_info)

            # generate the boundary matrix, D; row_p * column_p = n_1_path * n_path
            considered_operated_path_name = np.unique(boundary_operated_path_name_collect + allowed_path[n-1])
            unlimited_boundary_mat = np.zeros([allowed_path_n_types, len(considered_operated_path_name)])
            for i_path, (n_1_path_str, operated_n_path_dict) in enumerate(boundary_map_dict.items()):
                for j, n_path in enumerate(considered_operated_path_name):
                    if n_path in operated_n_path_dict:
                        unlimited_boundary_mat[i_path, j] = operated_n_path_dict[n_path]

            # collect informations
            boundary_map_matrix[n] = unlimited_boundary_mat
            boundary_mat_matrix_rank[n] = np.linalg.matrix_rank(unlimited_boundary_mat)
            allowed_path_idx_argument[n] = [1 if tpn in allowed_path[n-1] else 0 for tpn in considered_operated_path_name]

        return boundary_map_matrix, boundary_mat_matrix_rank, allowed_path_idx_argument

    def path_homology_for_connected_digraph(self, allowed_path, max_path):
        """
        Calculate the dimension of the path homology group for required dimensions.

        Parameters:
        -----------
        - allowed_path : dict
                the dict of all (0-n)-path, the format of the path is string.
        
        - max_path : int 
                the maximum length of path, (maximum dimension for homology)

        Returns:
        --------
        - betti_numbers : list
                the Betti number for dimension 0 to dimension max_path
        """
        betti_numbers = np.array([0] * (max_path + 1))

        boundary_map_matrix, boundary_mat_matrix_rank, allowed_path_idx_argument =\
            self.utils_unlimited_boundary_operator(allowed_path, max_path)

        # betti_0 = dim(H(omega_0)) = dim(omega_0 or allowed_path_0) - rank(D0) - rank(D1)
        betti_0 = len(allowed_path[0]) - 0 - boundary_mat_matrix_rank[1]
        betti_numbers[0] = betti_0

        # When dim > 0, H_n = (A_n U ker(\partial)) / (A_n U \partial(A_n+1))
        for n in range(1, max_path+1):
            if len(allowed_path[n]) == 0:
                break

            # dim of (A_n U ker(\partial)) = dim A_n - rank(D_n)
            dim_0 = len(allowed_path[n]) - boundary_mat_matrix_rank[n]

            # dim of (A_n intersect \partial(A_n+1)) = dim A_n + rank(D_n+1) - dim(A_n U (D_n+1 B_n))
            # >>> Let W = (D_n+1 B_n); note: B_n here means the image space
            # >>> dim(A_n U W) = dim A_n + dim W - dim (A_n + W)

            # >>> dim (A_n + W) = rank([I_(A_n_argument); D_n+1])
            dim_An_Bn = np.linalg.matrix_rank(
                np.vstack(
                    [
                        np.eye(len(allowed_path_idx_argument[n+1])) * allowed_path_idx_argument[n+1],
                        boundary_map_matrix[n+1]
                    ]
                )
            )
            dim_1 = len(allowed_path[n]) + boundary_mat_matrix_rank[n+1] - dim_An_Bn

            betti_numbers[n] = dim_0 - dim_1

        return betti_numbers

    def path_homology(self, edges, nodes, max_path):
        # check the data type
        if edges.dtype != nodes.dtype:
            edges = edges.astype(str)
            nodes = nodes.astype(str)

        # split into independent components
        all_components = PathHomology.split_independent_compondent(edges, nodes)
        all_digraphs = PathHomology.split_independent_digraph(all_components, edges)

        betti_numbers = []
        for i_d, edges in enumerate(all_digraphs):
            if np.shape(edges)[1] <= 1:
                betti_numbers.append(np.array([1] + [0] * (max_path)))
            else:
                edges = PathHomology.remove_loops(edges)
                if np.shape(edges)[0] == 0:
                    betti_numbers.append(np.array([1] + [0] * (max_path)))
                    continue
                allowed_path = self.utils_generate_allowed_paths(edges, max_path)
                betti_numbers.append(self.path_homology_for_connected_digraph(allowed_path, max_path))
        return np.sum(betti_numbers, axis=0)

    def persistent_path_homology(self, cloudpoints, points_weight, max_path, filtration=None):
        """
        Distance based filtration for cloudpoints.

        Parameters:
        -----------
        - cloudpoints : array
                the coordinates of the points
                
        - points_weight : list or array
                the weights of point in the cloudpoints, used to define the digraph
                
        - max_path : int
                maximum path length, or maximum dimension of path homology
        
        - filtration : array
                distance-based filtration, default: array(0, max_distance, 0.1)

        Returns:
        --------
        - all_betti_num : list
                a list of betti numbers of the path homology groups in each
                dimension (0-max_path) obtained during the filtation.
        """
        points_num = np.shape(cloudpoints)[0]
        points_idx = np.arange(points_num)

        # initial
        distance_matrix = np.zeros([points_num, points_num], dtype=float)
        # fully connected map, [0: no edge, 1: out]
        fully_connected_map = np.zeros([points_num, points_num], dtype=int)
        for i in range(points_num):
            for j in range(points_num):
                if i == j:
                    continue
                if self.cell is None:
                    distance = np.sqrt(np.sum((cloudpoints[i] - cloudpoints[j])**2))
                else:
                    distance = self.periodic_distance(
                        cloudpoints[i], cloudpoints[j], 
                        self.cell, self.pbc
                    )
                distance_matrix[i, j] = distance
                if points_weight[i] < points_weight[j]:
                    fully_connected_map[i, j] = 1
                if points_weight[i] == points_weight[j]:
                    if i < j:
                        fully_connected_map[i, j] = 1
        max_distance = np.max(distance_matrix)
        self.total_edges_num = np.sum(np.abs(fully_connected_map))
        self.max_distance = max_distance

        # filtration process
        if filtration is None:
            filtration = np.arange(0, np.round(max_distance, 2)+0.1, 0.1)

        all_betti_num = []
        save_time_flag = 0
        snapshot_map_temp = np.ones([points_num]*2, dtype=int)
        for n, snapshot_dis in enumerate(filtration):
            snapshot_map = np.ones([points_num]*2, dtype=int) * (distance_matrix <= snapshot_dis) * fully_connected_map
            if (snapshot_map == snapshot_map_temp).all():
                betti_numbers = all_betti_num[-1]
                all_betti_num.append(betti_numbers)
                continue
            else:
                snapshot_map_temp = copy.deepcopy(snapshot_map)

            start_ids = []
            end_ids = []
            for i in range(points_num):
                for j in range(points_num):
                    if i == j:
                        continue
                    if snapshot_map[i, j] == 1:
                        start_ids.append(i)
                        end_ids.append(j)
            edges = np.vstack([start_ids, end_ids]).T
            if save_time_flag == 1:
                betti_numbers = all_betti_num[-1]
                all_betti_num.append(betti_numbers)
                continue
            if np.shape(edges)[0] == self.total_edges_num:
                save_time_flag = 1
            betti_numbers = self.path_homology(edges, points_idx, max_path)
            all_betti_num.append(betti_numbers)

        return all_betti_num

    def persistent_path_homology_from_digraph(
        self, cloudpoints, all_edges, max_path, filtration=None
    ):
        """
        Distance-based filtration for digraph is performed.

        Parameters:
        -----------
        - cloudpoints : array
                the coordinates of the points
        
        - all_edges : array
                the maximum edges of the final digraph, shape: [n, 2]
        
        - filtration_angle_step : int, default 30
                the angle step during the angle-based filtration

        Returns:
        --------
        - all_betti_num : list
                a list of betti numbers of the path homology groups in each
                dimension (0-max_path) obtained during the angle-based filtation.
        """
        points_num = np.shape(cloudpoints)[0]
        points_idx = np.arange(points_num)

        # initial
        distance_matrix = np.zeros([points_num, points_num], dtype=float)
        for i in range(points_num-1):
            for j in range(i+1, points_num):
                if self.cell is None:
                    distance = np.sqrt(np.sum((cloudpoints[i] - cloudpoints[j])**2))
                else:
                    distance = self.periodic_distance(
                        cloudpoints[i], cloudpoints[j], 
                        self.cell, self.pbc
                    )
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
        max_distance = np.max(distance_matrix)
        self.max_distance = max_distance

        # fully connected map, [0: no edge, 1: out]
        fully_connected_map = np.zeros([points_num, points_num], dtype=int)
        for i, one_edge in enumerate(all_edges):
            fully_connected_map[one_edge[0], one_edge[1]] = 1
        self.total_edges_num = np.sum(np.abs(fully_connected_map))

        # filtration process
        if filtration is None:
            filtration = np.arange(0, 10+0.1, 0.1)

        all_betti_num = []
        save_time_flag = 0
        snapshot_map_temp = np.ones([points_num]*2, dtype=int)
        for n, snapshot_dis in enumerate(filtration):
            snapshot_map = np.ones([points_num]*2, dtype=int) * (distance_matrix <= snapshot_dis) * fully_connected_map
            if (snapshot_map == snapshot_map_temp).all():
                betti_numbers = all_betti_num[-1]
                all_betti_num.append(betti_numbers)
                continue
            else:
                snapshot_map_temp = copy.deepcopy(snapshot_map)

            start_ids = []
            end_ids = []
            for i in range(points_num):
                for j in range(points_num):
                    if i == j:
                        continue
                    if snapshot_map[i, j] == 1:
                        start_ids.append(i)
                        end_ids.append(j)
            edges = np.vstack([start_ids, end_ids]).T
            if save_time_flag == 1:
                betti_numbers = all_betti_num[-1]
                all_betti_num.append(betti_numbers)
                continue
            if np.shape(edges)[0] == self.total_edges_num:
                save_time_flag = 1
            betti_numbers = self.path_homology(edges, points_idx, max_path)
            all_betti_num.append(betti_numbers)

        return all_betti_num

    def persistent_angle_path_homology(self, cloudpoints, points_weight, max_path, filtration_angle_step=30):
        """
        Angle-based filtration of cloudpoints is performed.
        Specifically, we divide the 3-dimensional space in two steps. First, along the z-axis, for
        any plane perpendicular to the xy-plane and passing through the z-axis will be divided into
        12 sectors by rays passing through the origin at 30-degree intervals. Second, any of the
        partitioned planes in the first step (usually the plane passing through the x-axis) will be
        rotated along the z-axis at 30 degree intervals to de-partition the space and obtain 12 radial
        regions perpendicular to the z-axis direction. Combining these two steps, the 3D space can be
        divided into 72 (72 = 360/30 * 360/30 /2) regions. The angle filtration process is introduced
        by considering all edges contained in the region in order.

        Parameters:
        -----------
        - cloudpoints : array
                the coordinates of the points
                
        - points_weight : list or array
                the weights of point in the cloudpoints, used to define the digraph
        
        - max_path : int
                maximum path length, or maximum dimension of path homology
                
        - filtration_angle_step : int , default 30
                the angle step during the angle-based filtration

        Returns:
        --------
        - all_betti_num : list
                a list of betti numbers of the path homology groups in each
                dimension (0-max_path) obtained during the angle-based filtation.
        """

        points_num = np.shape(cloudpoints)[0]
        points_idx = np.arange(points_num)

        # based on the initial xyz, to generate the filtration, degree
        filtration = []
        # angle between edge vector and z axis
        for vz in range(filtration_angle_step, 180+1, filtration_angle_step):
            # angle between edge vector and y
            for vy in [0*180, 1*180]:
                # angle between cross(edge_vector, z) and x axis
                for cross_vz_y in range(filtration_angle_step, 180+1, filtration_angle_step):
                    filtration.append([vz, cross_vz_y + vy])

        # initial vector
        initial_vector_x = np.array([0., 0., 0.])
        max_distance = 0
        initial_vector_y_idx = [0, 0]
        all_edge_vectors = []
        all_edge_idx = []  # from [0] -> [1]

        for i in range(points_num):
            for j in range(points_num):
                if i == j:
                    continue
                if self.cell is None:
                    distance = np.sqrt(np.sum((cloudpoints[i] - cloudpoints[j])**2))
                else:
                    distance = self.periodic_distance(
                        cloudpoints[i], cloudpoints[j], 
                        self.cell, self.pbc
                    )
                edge_vector = cloudpoints[j, :] - cloudpoints[i, :]

                if distance > max_distance:
                    max_distance = distance
                    # record the vector idx corresponding to the max distance
                    initial_vector_y_idx = [i, j]

                if points_weight[i] <= points_weight[j]:
                    initial_vector_x += edge_vector
                    all_edge_vectors.append(edge_vector)
                    all_edge_idx.append([i, j])
        all_edge_idx = np.array(all_edge_idx)
        self.total_edges_num = len(all_edge_vectors)

        initial_vector_y = cloudpoints[initial_vector_y_idx[0], :] - cloudpoints[initial_vector_y_idx[1], :]
        # make sure the angle between x and y are acute angle
        if PathHomology().vector_angle(initial_vector_x, initial_vector_y) >= 90:
            initial_vector_y = -initial_vector_y
        initial_vector_z = np.cross(initial_vector_x, initial_vector_y)
        initial_vector_y = np.cross(initial_vector_x, initial_vector_z)  # make sure for orthogonal
        if self.initial_axes is None:
            self.initial_vector_x = initial_vector_x
            self.initial_vector_y = initial_vector_y
            self.initial_vector_z = initial_vector_z

        # calculate the angle between edge vector and x, y, z and realted rules
        two_related_angles = []
        for e_i, edge_v in enumerate(all_edge_vectors):
            edge_z = PathHomology().vector_angle(edge_v, self.initial_vector_z)
            edge_cross_vz_x = PathHomology().vector_angle(
                np.cross(edge_v, self.initial_vector_z), self.initial_vector_x)
            edge_y_flag = PathHomology().vector_angle(edge_v, self.initial_vector_y) // 90

            two_related_angles.append([edge_z, edge_cross_vz_x + 180*edge_y_flag])
        two_related_angles = np.array(two_related_angles)

        # persistent betti num
        all_betti_num = []
        edges = np.zeros([0, 2])
        for n, snapshot_angle in enumerate(filtration):
            snapshot_map_idx = np.append(
                np.where(two_related_angles[:, 0] < snapshot_angle[0])[0],
                np.where(
                    (
                        (two_related_angles[:, 0] <= snapshot_angle[0]) * (
                            two_related_angles[:, 0] > snapshot_angle[0]-filtration_angle_step
                        ) * (two_related_angles[:, 1] <= snapshot_angle[1])
                    )
                )
            )
            edges_temp = all_edge_idx[snapshot_map_idx, :]

            # delete original set
            two_related_angles = np.delete(two_related_angles, snapshot_map_idx, axis=0)
            all_edge_idx = np.delete(all_edge_idx, snapshot_map_idx, axis=0)
            edges = np.vstack([edges, edges_temp])

            if len(edges) > 0 and len(edges_temp) == 0:
                betti_numbers = all_betti_num[-1]
                all_betti_num.append(betti_numbers)
                continue
            betti_numbers = self.path_homology(edges.astype(int), points_idx, max_path)
            all_betti_num.append(betti_numbers)

        return all_betti_num

    def persistent_angle_path_homology_from_digraph(
        self, cloudpoints, all_edges, max_path, filtration_angle_step=30
    ):
        """
        Angle-based filtration for digraph is performed.

        Parameters:
        -----------
        - cloudpoints : array
                the coordinates of the points
                
        - all_edges : array
                the maximum edges of the final digraph, shape: [n, 2]
                
        - filtration_angle_step  : int , default 30
                he angle step during the angle-based filtration

        Returns:
        --------
        - all_betti_num : list
                a list of betti numbers of the path homology groups in each
                dimension (0-max_path) obtained during the angle-based filtation.
        """
        points_num = np.shape(cloudpoints)[0]
        points_idx = np.arange(points_num)

        # based on the initial xyz, to generate the filtration, degree
        filtration = []
        # angle between edge vector and z axis
        for vz in range(filtration_angle_step, 180+1, filtration_angle_step):
            # angle between edge vector and y
            for vy in [0*180, 1*180]:
                # angle between cross(edge_vector, z) and x axis
                for cross_vz_y in range(filtration_angle_step, 180+1, filtration_angle_step):
                    filtration.append([vz, cross_vz_y + vy])

        # initial vector
        initial_vector_x = np.array([0., 0., 0.])
        all_edge_vectors = []
        all_edge_idx = all_edges  # from [0] -> [1]
        for i_v, (s_v, e_v) in enumerate(all_edges):
            edge_vector = cloudpoints[e_v, :] - cloudpoints[s_v, :]
            initial_vector_x += edge_vector
            all_edge_vectors.append(edge_vector)
        self.total_edges_num = len(all_edge_vectors)

        # get new xyz axis
        max_distance = 0
        initial_vector_y_idx = [0, 0]
        # upper triangular matrix
        for i in range(points_num-1):
            for j in range(i+1, points_num):
                if self.cell is None:
                    distance = np.sqrt(np.sum((cloudpoints[i] - cloudpoints[j])**2))
                else:
                    distance = self.periodic_distance(
                        cloudpoints[i], cloudpoints[j], 
                        self.cell, self.pbc
                    )
                if distance > max_distance:
                    max_distance = distance
                    # record the vector idx corresponding to the max distance
                    initial_vector_y_idx = [i, j]

        initial_vector_y = cloudpoints[initial_vector_y_idx[0], :] - cloudpoints[initial_vector_y_idx[1], :]
        # make sure the angle between x and y are acute angle
        if PathHomology().vector_angle(initial_vector_x, initial_vector_y) >= 90:
            initial_vector_y = -initial_vector_y
        initial_vector_z = np.cross(initial_vector_x, initial_vector_y)
        initial_vector_y = np.cross(initial_vector_x, initial_vector_z)  # make sure for orthogonal
        if self.initial_axes is None:
            self.initial_vector_x = initial_vector_x
            self.initial_vector_y = initial_vector_y
            self.initial_vector_z = initial_vector_z

        # calculate the angle between edge vector and x, y, z and realted rules
        two_related_angles = []
        for e_i, edge_v in enumerate(all_edge_vectors):
            edge_z = PathHomology().vector_angle(edge_v, self.initial_vector_z)
            edge_cross_vz_x = PathHomology().vector_angle(
                np.cross(edge_v, self.initial_vector_z), self.initial_vector_x)
            edge_y_flag = PathHomology().vector_angle(edge_v, self.initial_vector_y) // 90
            two_related_angles.append([edge_z, edge_cross_vz_x + 180*edge_y_flag])
        two_related_angles = np.array(two_related_angles)

        # persistent betti num
        all_betti_num = []
        edges = np.zeros([0, 2])
        for n, snapshot_angle in enumerate(filtration):
            snapshot_map_idx = np.append(
                np.where(two_related_angles[:, 0] < snapshot_angle[0])[0],
                np.where(
                    (
                        (two_related_angles[:, 0] <= snapshot_angle[0]) * (
                            two_related_angles[:, 0] > snapshot_angle[0]-filtration_angle_step
                        ) * (two_related_angles[:, 1] <= snapshot_angle[1])
                    )
                )
            )

            edges_temp = all_edge_idx[snapshot_map_idx, :]

            # delete original set
            two_related_angles = np.delete(two_related_angles, snapshot_map_idx, axis=0)
            all_edge_idx = np.delete(all_edge_idx, snapshot_map_idx, axis=0)
            edges = np.vstack([edges, edges_temp])

            if len(edges) > 0 and len(edges_temp) == 0:
                betti_numbers = all_betti_num[-1]
                all_betti_num.append(betti_numbers)
                continue
            
            betti_numbers = self.path_homology(edges.astype(int), points_idx, max_path)
            all_betti_num.append(betti_numbers)

        return all_betti_num
    
    def custom_pbc_distance(self,points, cell_size):
        """
        Calculate pairwise minimum-image distances under orthogonal Periodic Boundary Conditions (PBC).

        Parameters:
        -----------
        - points : np.ndarray
                (N,3) array of atomic coordinates
            
        - cell_size : np.ndarray
                (3,) array of orthogonal cell dimensions [a,b,c]
        
        Returns:
        --------
        - dist_matrix : np.ndarray
                (N,N) symmetric distance matrix, where element [i,j] gives the 
                minimum-image distance between points i and j under PBC.
        """
        n = len(points)
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                delta = points[j] - points[i]
                delta -= np.round(delta / cell_size) * cell_size  
                dist_matrix[i][j] = np.linalg.norm(delta)
                dist_matrix[j][i] = dist_matrix[i][j]
        return dist_matrix
    
    def persistent_homology(self,cloudpoints,max_path,filtration_values=None):
        """
        Compute persistent homology using the Gudhi library with optional PBC handling.
        
        Parameters:
        -----------
        - cloudpoints : np.ndarray
                (N,3) array of input coordinates.
        
        - max_path : int
                Maximum homology dimension to compute (H_0 to H_max_path).
            
        - filtration_values : list or array, optional
                Filter thresholds. Auto-generated if None.
        
        Returns:
        --------
        - list
                Betti numbers at each filtration step, where each element is a 
                [β0, β1,..., β_max_path] array.
        
        """
        import gudhi as gd
        if self.pbc == True:
            distance_matrix = self.custom_pbc_distance(cloudpoints, self.cell)
            rips_complex = gd.RipsComplex(distance_matrix=distance_matrix)
        else:
            diff = cloudpoints[:, np.newaxis, :] - cloudpoints[np.newaxis, :, :]
            distance_matrix = np.linalg.norm(diff, axis=-1)
            rips_complex = gd.RipsComplex(points=distance_matrix)

        simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_path+1)
        persistence = simplex_tree.persistence()
        max_distance = np.max(distance_matrix)
        if filtration_values is None:
            filtration_values = np.arange(0, np.round(max_distance, 2)+0.1, 0.1)
        betti_num_all = []
        for d in filtration_values:
            betti = np.zeros(max_path + 1, dtype=int)
            for dim, (birth, death) in persistence:
                if dim > max_path:
                    continue
                if birth <= d and (death > d or death == -1):  
                    betti[dim] += 1
            betti_num_all.append(betti)
        
        return betti_num_all

import numpy as np
import itertools
from functools import wraps
import copy
import time
from scipy.spatial import distance

# Define a decorator to measure the execution time of functions.
def timeit(func):
    """ 
    Timer decorator to print the execution time of the wrapped function. """
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()   
        result = func(*args, **kwargs)       
        end_time = time.perf_counter()      
        total_time = end_time - start_time  
        print(f"{'='*5} Function - {func.__name__} - took {total_time:.3f} seconds {'='*5}")
        return result
    return timeit_wrapper


class statistic_eigvalues(object):
    
    def __init__(self, eigvalues: np.array) -> None:
        digital = 5
        
        values = np.round(eigvalues, 5)
        self.all_values = sorted(values)         
        self.nonzero_values = values[np.nonzero(values)]  
        self.count_zero = len(values) - np.count_nonzero(values)  
        self.max = np.max(values)                
        self.sum = np.round(np.sum(values), digital) 

        # If there are nonzero eigenvalues, compute their statistics.
        if len(self.nonzero_values) > 0:
            self.nonzero_mean = np.round(np.mean(self.nonzero_values), digital)  
            self.nonzero_std = np.round(np.std(self.nonzero_values), digital)    
            self.nonzero_min = np.round(np.min(self.nonzero_values), digital)    
            self.nonzero_var = np.round(np.var(self.nonzero_values), digital)    
        else:
            # If there are no nonzero values, set statistics to 0.
            self.nonzero_mean = 0
            self.nonzero_std = 0
            self.nonzero_min = 0
            self.nonzero_var = 0


class SimplicialComplexLaplacian(object):
    def __init__(self, eigenvalue_method='numpy_eigvalsh'):
        self.distance_matrix = None
        # Set the eigenvalue calculator to numpy's eigvalsh function,
        # which is efficient for symmetric matrices.
        if eigenvalue_method == 'numpy_eigvalsh':
            self.eigvalue_calculator = np.linalg.eigvalsh

    def utils_powersets(self, nodes: list, max_dim: int = 2) -> dict:
        """
        Generate all subsets (up to a specified dimension) of a set of nodes.
        Returns a dictionary where keys represent the dimension and values are lists of node combinations.
        """
        complete_edge_dict = {i: [] for i in range(max_dim)}
        # The maximum length for combinations is the minimum of the number of nodes and max_dim.
        max_len = min([len(nodes), max_dim])
        for i in range(max_len+1):
            complete_edge_dict[i] = list(itertools.combinations(nodes, i+1))
        return complete_edge_dict

    def adjacency_map_to_simplex(self, adjacency_matrix: np.array, max_dim: int = 1) -> dict:
        """
        Given an adjacency matrix for an undirected graph, construct the clique complex (simplicial complex)
        of the graph.
        
        Parameters:
        -----------
        - adjacency_matrix : np.array
                Adjacency matrix of the graph.
        - max_dim : int 
                Maximum simplex dimension to construct (0: vertices, 1: edges, 2: triangles, etc.)
        
        Returns:
        ---------
        - simplicial_complex
                Dictionary where keys are dimensions and values are lists of simplices.
        """
        n = adjacency_matrix.shape[0]  # Number of nodes in the graph
        # Initialize the simplicial complex dictionary for dimensions 0 to max_dim.
        simplicial_complex = {dim: [] for dim in range(max_dim+1)}

        # Add 0-simplices (vertices).
        simplicial_complex[0] = [(i, ) for i in range(n)]

        # For dimensions higher than 0, add simplices corresponding to cliques.
        target_dim = min(max_dim, n)
        for k in range(1, target_dim+1):
            
            for S in itertools.combinations(range(n), k+1):
                if all(adjacency_matrix[i, j] for i in S for j in S if i < j):
                    simplicial_complex[k].append(tuple(S))
        return simplicial_complex
    
    def complex_to_boundary_matrix(self, complex: dict) -> dict:
        """
        Convert a simplicial complex into its boundary matrices.
        Each boundary matrix encodes the relationship between n-simplices and (n-1)-simplices.
        
        Parameters:
        -----------
        - complex : dict 
                Dictionary representing the simplicial complex.
        
        Returns:
        --------
        - boundary_matrix_dict : dict
                Dictionary where each key corresponds to a dimension and its value is the boundary matrix.
        """
        boundary_matrix_dict = {dim_n: None for dim_n in complex.keys()}
        for dim_n in sorted(complex.keys()):
            if dim_n == 0:
                # For 0-simplices, there is no boundary; use a zero vector.
                boundary_matrix_dict[dim_n] = np.zeros([len(complex[0]), ])
                continue
            # Get sorted lists of n-simplices and (n-1)-simplices.
            simplex_dim_n = sorted(complex[dim_n])
            simplex_dim_n_minus_1 = sorted(complex[dim_n-1])
            if len(simplex_dim_n) == 0:
                break
            # Initialize the boundary matrix for dimension n with rows = (n-1)-simplices and columns = n-simplices.
            boundary_matrix_dict[dim_n] = np.zeros([len(simplex_dim_n_minus_1), len(simplex_dim_n)])
            # For each n-simplex, compute its boundary.
            for idx_n, simplex_n in enumerate(simplex_dim_n):
                # Iterate over each vertex in the n-simplex.
                for omitted_n in range(len(simplex_n)):
                    # Create the (n-1)-simplex by removing one vertex.
                    omitted_simplex = tuple(np.delete(simplex_n, omitted_n))
                    # Find the index of the (n-1)-simplex in the sorted list.
                    omitted_simplex_idx = simplex_dim_n_minus_1.index(omitted_simplex)
                    # Assign a sign based on the position of the omitted vertex.
                    boundary_matrix_dict[dim_n][omitted_simplex_idx, idx_n] = (-1)**omitted_n

        # Store the highest boundary dimension for later use.
        self.has_boundary_max_dim = dim_n
        return boundary_matrix_dict

    def boundary_to_laplacian_matrix(self, boundary_matrix_dict: dict) -> dict:
        """
        Construct the Laplacian matrices of the simplicial complex using its boundary matrices.
        The Laplacian is given by:
            L = B_{n+1} * B_{n+1}^T + B_n^T * B_n
        
        Parameters:
        -----------
        - boundary_matrix_dict : dict 
                Dictionary of boundary matrices.
        
        Returns:
        --------
        - laplacian_matrix_dict : dict
                Dictionary where each key corresponds to a dimension and its value is the Laplacian matrix.
        """
        laplacian_matrix_dict = {}
        for dim_n in sorted(boundary_matrix_dict.keys()):
            boundary_matrix = boundary_matrix_dict[dim_n]
            
            if dim_n >= self.has_boundary_max_dim:
                break
            
            elif dim_n == 0 and boundary_matrix_dict[dim_n+1] is not None:
                laplacian_matrix_dict[dim_n] = np.dot(boundary_matrix_dict[dim_n+1], boundary_matrix_dict[dim_n+1].T)
            
            elif dim_n == 0 and boundary_matrix_dict[dim_n+1] is None:
                laplacian_matrix_dict[dim_n] = np.zeros([len(boundary_matrix_dict[0])]*2)
            
            elif dim_n > 0 and boundary_matrix_dict[dim_n+1] is None:
                laplacian_matrix_dict[dim_n] = np.dot(boundary_matrix.T, boundary_matrix)
                break
            else:
                
                laplacian_matrix_dict[dim_n] = np.dot(boundary_matrix_dict[dim_n+1], boundary_matrix_dict[dim_n+1].T) + np.dot(boundary_matrix.T, boundary_matrix)
        return laplacian_matrix_dict

    def simplicialComplex_laplacian_from_connected_mat(self, adjacency_matrix: np.array, max_dim: int = 1) -> np.array:
        """
        Given the adjacency matrix of a connected graph, compute the eigenvalues of the Laplacian matrices
        for its corresponding simplicial complex.
        
        Parameters:
        -----------
        - adjacency_matrix : np.array
                Adjacency matrix of the graph.
        - max_dim : int 
                Maximum dimension of the Laplacian to compute.
        
        Returns:
        --------
        - laplacian_eigenv : dict
                Dictionary of eigenvalues keyed by dimension.
        """
        self.max_dim = max_dim
        self.max_boundary_dim = max_dim + 1
        
        complex = self.adjacency_map_to_simplex(adjacency_matrix, self.max_boundary_dim)
        
        boundary_matrix_dict = self.complex_to_boundary_matrix(complex)
       
        laplacian_matrix_dict = self.boundary_to_laplacian_matrix(boundary_matrix_dict)

        laplacian_eigenv = {}
        
        for dim_n in range(self.max_dim+1):
            if dim_n in laplacian_matrix_dict:
                laplacian_matrix = laplacian_matrix_dict[dim_n]
                eig_value = self.eigvalue_calculator(laplacian_matrix)
                eig_value = eig_value.real  # Use the real part of the eigenvalues
                laplacian_eigenv[dim_n] = sorted(np.round(eig_value, 5))
            else:
                laplacian_eigenv[dim_n] = None
        
        self.laplacian_matrix_dict = laplacian_matrix_dict
        return laplacian_eigenv

    def persistent_simplicialComplex_laplacian(
        self, input_data: np.array = None,
        max_adjacency_matrix: np.array = None, min_adjacency_matrix: np.array = None,
        is_distance_matrix: bool = False, max_dim: int = 1, filtration: np.array = None,
        cutoff_distance: float = None, step_dis: float = None, print_by_step: bool = True,
        lattice_vectors: np.array = None,
        pbc: list = None
    ) -> np.array:
        """
        Compute persistent Laplacian eigenvalues of a simplicial complex over a filtration.

        Parameters:
        -----------
        - input_data : np.ndarray
                Either point cloud data (N, D) or a precomputed (N, N) distance matrix depending on `is_distance_matrix`.

        - max_adjacency_matrix : np.ndarray, optional
                A binary (N, N) matrix indicating maximum allowed edges (1 = allow, 0 = disallow).

        - min_adjacency_matrix : np.ndarray, optional
                A binary (N, N) matrix indicating mandatory edges (1 = always include, 0 = optional).

        - is_distance_matrix : bool
                If True, `input_data` is interpreted as a distance matrix; otherwise, it is treated as point cloud coordinates.

        - max_dim : int
                The maximum dimension of simplices for which Laplacians are computed (e.g., 1 for edges, 2 for triangles).

        - filtration : np.ndarray, optional
                A 1D array of distance thresholds used in the filtration process. If None, will be generated from cutoff/step.

        - cutoff_distance : float, optional
                The upper bound for the filtration. Required if `filtration` is not provided.

        - step_dis : float, optional
                The step size for generating the filtration thresholds (used with `cutoff_distance`).

        - print_by_step : bool
                Whether to print Laplacian eigenvalues at each filtration step.

        - lattice_vectors : np.ndarray, optional
                (D, D) array of lattice vectors for PBC calculation. If provided, periodic distances will be computed.

        - pbc : list of bool, optional
             Periodic boundary conditions along each axis, e.g. [True, True, False]. Only used if `lattice_vectors` is set.

        Returns:
        --------
        - all_laplacian_features : list of dict
                A list of dictionaries; each dict maps dimension (int) to Laplacian eigenvalues (sorted list of floats),
                collected at each filtration step.
        """
        # If the input is a distance matrix, use it directly; otherwise compute from point cloud data.
        # If periodicity is considered , inputdata needs to enter point cloud data.
        if pbc is None:
            pbc = [False] * 3  
            
        if lattice_vectors is not None:
            dim = lattice_vectors.shape[0]
            valid_pbc = pbc[:dim] + [False]*(dim-len(pbc))  
            distance_matrix = self.periodic_distance_general(
                input_data, lattice_vectors, valid_pbc
            )
            points_num = distance_matrix.shape[0]
        else:
            if is_distance_matrix:
                distance_matrix = input_data
                points_num = distance_matrix.shape[0]
            else:
                cloudpoints = input_data
                points_num = cloudpoints.shape[0]
                distance_matrix = distance.cdist(cloudpoints, cloudpoints)
                
        if max_adjacency_matrix is None:
            max_adjacency_matrix = np.ones([points_num, points_num], dtype=int)
            np.fill_diagonal(max_adjacency_matrix, 0)
        
        if min_adjacency_matrix is None:
            min_adjacency_matrix = np.zeros([points_num, points_num], dtype=int)

        if filtration is None:
            filtration = np.arange(0, cutoff_distance, step_dis)
        
        all_laplacian_features = []
        
        adjacency_matrix_temp = np.ones([points_num]*2, dtype=int)
        
        for threshold_dis in filtration:
            
            adjacency_matrix = (((distance_matrix <= threshold_dis) * max_adjacency_matrix + min_adjacency_matrix) > 0)

            
            if not (adjacency_matrix == adjacency_matrix_temp).all():
                adjacency_matrix_temp = copy.deepcopy(adjacency_matrix)
                laplacian_eigenv = self.simplicialComplex_laplacian_from_connected_mat(adjacency_matrix, max_dim)
                all_laplacian_features.append(laplacian_eigenv)
            else:
               
                all_laplacian_features.append(all_laplacian_features[-1])
                
           
            if print_by_step:
                for dim_ii in range(max_dim):
                    print(f"filtration param: {threshold_dis} dim_n: {dim_ii} eigenvalues:{laplacian_eigenv[dim_ii]}")

        return all_laplacian_features

    def periodic_distance_general(self, 
                                points: np.array, 
                                lattice_vectors: np.array,
                                pbc: list = [True, True, True]) -> np.array:
        """
        General periodic distance calculation
        
        Parameters:
        -----------
        - points : np.array (N, dim)
                Point cloud coordinates
        - lattice_vectors : np.array (dim, dim)
                Lattice basis vector matrix
        - pbc : list of bool
                Whether to apply periodic boundaries to each dimension, such as [True, True, False] means the first two dimensions are periodic
            
        Returns:
        --------
        - distance_matrix : np.array (N, N)
        """
        dim = lattice_vectors.shape[0]
        assert len(pbc) >= dim
        
        inv_lattice = np.linalg.inv(lattice_vectors.T)
        n_points = points.shape[0]
        distance_matrix = np.zeros((n_points, n_points))
        
 
        for i in range(n_points):
            for j in range(i+1, n_points):
                delta = points[j] - points[i]
                delta_frac = np.dot(delta, inv_lattice)
                
                for d in range(dim):
                    if pbc[d]:
                        delta_frac[d] -= np.round(delta_frac[d])
                
                delta_cart = np.dot(delta_frac, lattice_vectors.T)
                
                distance_matrix[i, j] = np.linalg.norm(delta_cart)
                distance_matrix[j, i] = distance_matrix[i, j]  
                
        return distance_matrix
    
    def persistent_simplicialComplex_laplacian_dim0(
        self, input_data: np.array = None,
        max_adjacency_matrix: np.array = None, min_adjacency_matrix: np.array = None,
        is_distance_matrix: bool = False, max_dim: int = 1, filtration: np.array = None,
        cutoff_distance: float = None, step_dis: float = None, print_by_step: bool = True,
        lattice_vectors: np.array = None,
        pbc: list = None
    ) -> np.array:
        """
        Compute 0-dimensional (classical graph) Laplacian eigenvalues across a filtration.
        This is a simplified version of the persistent Laplacian pipeline that only uses:
            L = D - A
        where L is the graph Laplacian, D is the degree matrix, and A is the adjacency matrix.

        Parameters:
        -----------
        - input_data : np.ndarray
                Either (N, D) point cloud coordinates or a (N, N) distance matrix, depending on `is_distance_matrix`.

        - max_adjacency_matrix : np.ndarray, optional
                Binary mask matrix specifying maximum allowed connections (1 = allow edge).

        - min_adjacency_matrix : np.ndarray, optional
                Binary mask matrix specifying forced connections (1 = must connect).

        - is_distance_matrix : bool
                If True, `input_data` is treated as a distance matrix; otherwise, it's treated as point cloud data.

        - max_dim : int
                Maximum dimension for Laplacian eigenvalue reporting (here used for formatting/logging only).

        - filtration : np.ndarray, optional
                Array of filtration thresholds. If not provided, it will be generated from `cutoff_distance` and `step_dis`.

        - cutoff_distance : float, optional
                Maximum filtration threshold used to generate `filtration` if it is None.

        - step_dis : float, optional
                Step size for distance threshold increments (used with `cutoff_distance`).

        - print_by_step : bool
                If True, prints eigenvalues at each filtration step.

        - lattice_vectors : np.ndarray, optional
                Lattice basis vectors (D×D) for periodic boundary condition distance calculation.

        - pbc : list of bool, optional
                Periodic boundary condition flags for each axis (e.g. [True, True, False]).

        Returns:
        --------
        - all_laplacian_features : list of dict
                List of dictionaries containing 0-dimensional Laplacian eigenvalues at each filtration step.
                Format: [{0: [eigvals]}, {0: [eigvals]}, ...]
        """
        if pbc is None:
            pbc = [False] * 3  
            
        if lattice_vectors is not None:
            dim = lattice_vectors.shape[0]
            valid_pbc = pbc[:dim] + [False]*(dim-len(pbc))  
            distance_matrix = self.periodic_distance_general(
                input_data, lattice_vectors, valid_pbc
            )
        else:
            if is_distance_matrix:
                distance_matrix = input_data
                points_num = distance_matrix.shape[0]
            else:
                cloudpoints = input_data
                points_num = cloudpoints.shape[0]
                distance_matrix = distance.cdist(cloudpoints, cloudpoints)

        if max_adjacency_matrix is None:
            max_adjacency_matrix = np.ones([points_num, points_num], dtype=int)
            np.fill_diagonal(max_adjacency_matrix, 0)
        
        if min_adjacency_matrix is None:
            min_adjacency_matrix = np.zeros([points_num, points_num], dtype=int)

        if filtration is None:
            filtration = np.arange(0, cutoff_distance, step_dis)
        
        all_laplacian_features = []
        adjacency_matrix_temp = np.ones([points_num]*2, dtype=int)
        for threshold_dis in filtration:
            # Construct the adjacency matrix based on the distance threshold.
            adjacency_matrix = (((distance_matrix <= threshold_dis) * max_adjacency_matrix + min_adjacency_matrix) > 0)

            if not (adjacency_matrix == adjacency_matrix_temp).all():
                adjacency_matrix_temp = copy.deepcopy(adjacency_matrix)
                # Compute the classical graph Laplacian: L = D - A.
                laplacian_matrix_dim0 = np.diag(np.sum(adjacency_matrix, axis=0)) - adjacency_matrix
                eig_value = self.eigvalue_calculator(laplacian_matrix_dim0)
                eig_value = eig_value.real
                laplacian_eigenv = {0: sorted(np.round(eig_value, 5))}
                all_laplacian_features.append(laplacian_eigenv)
            else:
                all_laplacian_features.append(all_laplacian_features[-1])
                
            if print_by_step:
                for dim_ii in range(max_dim):
                    print(f"filtration param: {threshold_dis} dim_n: {dim_ii} eigenvalues:{laplacian_eigenv[dim_ii]}")

        return all_laplacian_features




