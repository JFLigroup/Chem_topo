import numpy as np 
from ase import Atoms
from ase.neighborlist import natural_cutoffs, NeighborList
import gudhi
import math
from scipy.spatial import KDTree
from ase.data import atomic_numbers, covalent_radii
from .utils import calculate_centroid , get_cutoffs ,metal_elements,plane_normal
class ClusterAdsorptionSitesFinder():
        """
        Parameters:
        -----------        
        - atoms : ase.Atoms object
                The nanoparticle to use as a template to generate surface sites. 
                Accept any ase.Atoms object. 

        - adsorbate_elements : list of strs
                If the sites are generated without considering adsorbates, 
                the symbols of the adsorbates are entered as a list, resulting in a clean slab

        - bond_len : float , default None
                The sum of the atomic radii between a reasonable adsorption 
                substrate and the adsorbate.If not setting, the threshold for 
                the atomic hard sphere model is automatically calculated using 
                the covalent radius of ase

        - mul : float , default 0.8
                A scaling factor for the atomic bond lengths used to modulate
                the number of sites generated, resulting in a larger number of potential sites 

        - tol : float , default 0.6
                The minimum distance in Angstrom between two site for removal
                of partially overlapping site

        - radius : float , default 5.0
                The maximum distance between two points 
                when performing persistent homology calculations  

        - k : float , default 1.1
                Expand the key length, so as to calculate the length of the expansion
                in the direction of the normal vector, k value is too small will lead 
                to the calculation of the length of the expansion of the error, 
                you can according to the needs of appropriate increase

        """
        def __init__(self,atoms,
                        adsorbate_elements=[],
                        bond_len = None,
                        mul=0.8,
                        tol=0.7,
                        k=1.1,
                        radius=5.):
                assert True not in atoms.pbc, 'the cell must be non-periodic' # Ensure the cell is non-periodic (PBC must be False)
                atoms = atoms.copy() # Make a copy of the atoms object to avoid modifying the original
                # Loop through each dimension to ensure the cell dimensions are valid
                for dim in range(3):
                        # If the dimension length is zero, set it to the range of atomic positions plus a buffer
                        if np.linalg.norm(atoms.cell[dim]) == 0:
                                # Set dimension length to the range plus 10
                                atoms.cell[dim][dim] = np.ptp(atoms.positions[:, dim]) + 10.
                # Identify the indices of metal atoms (those whose symbols are not in adsorbate_elements)
                self.metal_ids = [a.index for a in atoms if a.symbol not in adsorbate_elements]

                # Create a new Atoms object with only the metal atoms
                atoms = Atoms(atoms.symbols[self.metal_ids], 
                        atoms.positions[self.metal_ids], 
                        cell=atoms.cell, pbc=atoms.pbc) 
                # Initialize various attributes of the class with relevant data from the atoms object
                self.atoms = atoms
                self.bond_len = bond_len
                self.positions = atoms.positions
                self.symbols = atoms.symbols
                self.numbers = atoms.numbers
                self.tol = tol
                self.k = k
                self.mul = mul
                self.radii = radius
                self.cell = atoms.cell
                self.pbc = atoms.pbc
                self.metals = sorted(list(set(atoms.symbols)))
                #get surface atoms based on coordination
                self.surf_ids, self.surf_atoms = self.get_surface_atoms()
                # Initialize empty lists for surface sites and inside sites
                self.surf_site_list = []
                self.inside_site_list = []
                # Set radii for surface and inside adsorption sites(drfault H)
                self.surface_add_radii = covalent_radii[1]
                self.inside_add_radii = covalent_radii[1]
                # Initialize lists for surface indices and sites
                self.sur_index = []
                self.sites = []
        def get_surface_atoms(self, cutoff=3, tolerance = 1e-3):
                """
                Identify surface atoms in a structure based on the sum of neighbor direction vectors.

                Parameters:
                -----------
                - cutoff : float
                        Cutoff radius for neighbor search (in Ångströms)
                
                - tolerance : float 
                        If the magnitude of the summed normal vectors is less than this value, 
                        the atom is considered a non-surface atom

                Returns:
                --------
                - surf_ids : list
                        List of indices of surface atoms.

                - surface_atoms : ase.Atoms object
                        Atoms object containing only the detected surface atoms.
                """
                positions = self.positions
                num_atoms = len(self.atoms)
                
                neighbor_list = NeighborList([cutoff / 2.0] * num_atoms, self_interaction=False, bothways=True)
                neighbor_list.update(self.atoms)
                surface_indices = []
                max_neighbors = np.max([len(neighbor_list.get_neighbors(i)[0]) for i in range(num_atoms)])
                threshold_neighbors = max_neighbors - 2
                
                for i in range(num_atoms):
                        indices, offsets = neighbor_list.get_neighbors(i)
                        
                        if len(indices) == 0:
                                surface_indices.append(i)
                                continue

                        pos_i = positions[i]
                        summed_normal = np.zeros(3)

                        for j, offset in zip(indices, offsets):
                                pos_j = positions[j] + np.dot(offset, self.cell)
                        normal = pos_j - pos_i
                        norm = np.linalg.norm(normal)
                        if norm > 1e-6:
                                summed_normal += normal / norm  

                        if len(indices) < threshold_neighbors and  np.linalg.norm(summed_normal) > tolerance:
                                surface_indices.append(i)

                surf_atoms = self.atoms[surface_indices]

                return surface_indices , surf_atoms


        def extend_point_away_from_center(self, center, point, distance):
                """
                Offset a surface site along the normal vector.

                This function generates a normal vector using the center of mass of the 
                initial atom cluster and offsets the point outward by a specified distance.

                Parameters:
                -----------
                - center : ndarray
                        The center point of the cluster.

                - point : ndarray
                        The site to be offset outward.

                - distance : float
                        The distance to offset in the direction of the normal vector.

                Returns:
                --------
                - new_point : ndarray
                        The new position of the point after being offset.

                - unit_vector : ndarray
                        The unit vector representing the direction of the offset.
                """
                # Calculate the vector from the center to the point
                vector = np.array(point) - np.array(center)

                # Calculate the length of the vector
                length = np.linalg.norm(vector)

                # Ensure the point and center are not the same
                if length == 0:
                        raise ValueError("The point and center cannot be the same.")

                # Normalize the vector to get the unit vector
                unit_vector = vector / length

                # Offset the point along the unit vector by the specified distance
                new_point = np.array(center) - distance * unit_vector

                return new_point, unit_vector

        def get_sites(self, absorbent=[]):
                """
                Retrieve all adsorption sites (surface and inside).

                This method combines surface and inside sites into a single list. If these 
                sites have not been previously calculated, it computes them first.

                Parameters:
                -----------
                - absorbent : list, optional
                        A list of atoms or elements to be considered during site calculations.
                        Defaults to an empty list.

                Returns:
                --------
                - sites : list
                        A list of all adsorption sites, including both surface and inside sites.
                """
                # Return precomputed sites if available
                if self.sites:
                        return self.sites

                # Calculate surface sites if not already computed
                if not self.surf_site_list:
                        self.get_surface_sites(absorbent=absorbent)

                # Calculate inside sites if not already computed
                if not self.inside_site_list:
                        self.get_inside_sites(absorbent=absorbent)

                # Combine surface and inside sites into a single list
                self.sites = self.surf_site_list + self.inside_site_list

                return self.sites


        def surf_topo(self):
                """
                The function to topologize surface atoms and obtain sites.
                It uses Alpha Complex to calculate surface topology and generates adsorption sites.

                Returns:
                --------
                Updates self.surf_site_list with the surface sites generated.
                """
                # Get the center of mass of the atoms and positions of surface atoms
                center = self.atoms.get_center_of_mass()
                surf_pos = self.surf_atoms.get_positions()
                surface_atoms = self.surf_atoms

                # Initialize an Alpha Complex with surface atom positions
                rc = gudhi.AlphaComplex(points=surf_pos)
                st = rc.create_simplex_tree((self.radii / 2) ** 2)  # Alpha Complex filtration radius

                # Get the simplicial complex (up to 4-simplex)
                combinations = st.get_skeleton(4)

                # Initialize site types and groups for tracking
                sites = []
                site_type = ['top', 'bridge', 'hollow', "4fold"]
                fold4_group = []
                del_bri_couple = []

                # Sort combinations by simplex size (from higher dimensions to lower)
                combinations = sorted(list(combinations), key=lambda x: len(x[0]), reverse=True)

                # Iterate through the simplices in the Alpha Complex
                for com in combinations:
                        temp = surf_pos[com[0]]  # Get positions of the atoms in the simplex
                        cov_radii = [covalent_radii[self.surf_atoms[c].number] for c in com[0]]  # Covalent radii of atoms

                        if len(com[0]) == 1:  # Single atom (top site)
                                temp_com = com[0]
                                site = temp[0]
                        elif len(com[0]) == 2:  # Two atoms (bridge site)
                                temp_com = com[0]
                                # Remove the bridge of the two hypotenuses of the 4-fold site.
                                if tuple(sorted(com[0])) in del_bri_couple:
                                        continue
                                t = cov_radii[1] / sum(cov_radii)
                                site = t * temp[0] + (1 - t) * temp[1]  # Weighted position between the two atoms
                        else:  # Higher-order sites (hollow, 4-fold)
                                if self.bond_len is None:
                                        bond_len = max(cov_radii) + self.surface_add_radii
                                else:
                                        bond_len = self.bond_len

                                site = calculate_centroid(temp, cov_radii, math.sqrt(com[1]))  # Calculate the centroid
                                temp_com = []
                                cov_radii = []

                                for i, coord in enumerate(surf_pos):  # Adjust site based on nearby atoms
                                        if np.linalg.norm(site - coord) <= bond_len:
                                                temp_com.append(i)
                                        cov_radii.append(covalent_radii[surface_atoms[i].number])

                                # Handle 4-fold and hollow cases
                                if len(temp_com) == 4:
                                        index_tuple = tuple(temp_com)
                                        if index_tuple in fold4_group:
                                                continue
                                        else:
                                                site = calculate_centroid(surf_pos[temp_com], cov_radii, math.sqrt(com[1]))
                                                fold4_group.append(index_tuple)

                                        # Identify and exclude redundant bridge combinations
                                        max_d = -1
                                        for ind in temp_com[1:]:
                                                if np.linalg.norm(surf_pos[temp_com[0]] - surf_pos[ind]) > max_d:
                                                        max_d = np.linalg.norm(surf_pos[temp_com[0]] - surf_pos[ind])
                                                        temp_i = ind
                                        del_bri_couple.append((temp_com[0], temp_i))

                                        remain = []
                                        for i in temp_com:
                                                if i != temp_com[0] and i != temp_i:
                                                        remain.append(i)
                                        del_bri_couple.append(tuple(remain))
                                elif len(temp_com) == 3:
                                        temp_com = com[0]

                        # Offset the site away from the center of mass
                        if self.bond_len is None:
                                bond_len = min(cov_radii) + self.surface_add_radii
                        else:
                                bond_len = self.bond_len

                        try:
                                height = math.sqrt((bond_len * self.k) ** 2 - (com[1]))
                        except Exception:
                                height = 0.1

                        site, normal = self.extend_point_away_from_center(site, center, height)

                        # Check if the site is too close to existing atoms
                        flag = True
                        for ap in self.positions:
                                if np.linalg.norm(ap - site) + 0.01 < bond_len * self.mul:
                                        flag = False
                                        break

                        # If valid, add the site to the list
                        if flag:
                                sites.append({
                                        'site': site_type[len(temp_com) - 1],
                                        'type': 'surface',
                                        'normal': normal,
                                        'position': site,
                                        'indices': [c for c in temp_com]
                                })

                # Handle site proximity for final site list
                if self.tol == False:
                        self.surf_site_list = sites
                else:
                        for site in sites:
                                flag = True
                                for s in self.surf_site_list:
                                        if np.linalg.norm(np.array(s['position']) - np.array(site['position'])) < self.tol:
                                                flag = False
                                if flag:
                                        self.surf_site_list.append(site)


        def get_surface_sites(self, absorbent=[]):
                """
                Generate and return surface adsorption sites based on surface topology.

                Parameters:
                -----------
                - absorbent : list of str, optional
                        A list of chemical symbols for adsorbent elements (e.g., ['H', 'O']).
                        Used to adjust radii for adsorbent-related calculations.

                Returns:
                --------
                - surf_site_list : list of dict
                        A list of dictionaries containing information about surface adsorption sites.
                """
                # Adjust the additional radii based on the provided absorbent elements
                if absorbent:
                        # Calculate the minimum covalent radius for the given absorbent elements
                        self.surface_add_radii = min([covalent_radii[atomic_numbers.get(ele, None)] for ele in absorbent])

                # Generate surface sites based on surface topology
                self.surf_topo()

                # Return the list of surface sites
                return self.surf_site_list


        def inside_topo(self):
                """
                Identify embedding sites within non-periodic structures (e.g., nanoclusters).

                Uses Alpha Complex and Rips Complex to find topological embedding sites.

                - Updates `self.inside_site_list` with identified embedding sites.
                """
                pos = self.positions  # Atomic positions of the structure
                
                # Computing sites in 4 dimensions using Alpha Complex
                ac = gudhi.AlphaComplex(points=pos)
                st = ac.create_simplex_tree((self.radii / 2) ** 2)    
                combinations = st.get_skeleton(4)
                sites = []
                
                # Process lower-dimensional combinations (up to 4)
                for com in combinations:
                        if len(com[0]) >= 2:  # Sites formed by at least 2 atoms
                                temp = pos[com[0]]  # Extract positions
                                cov_radii = [covalent_radii[self.atoms[c].number] for c in com[0]]
                        
                                # Compute site position
                                if len(com[0]) == 2:
                                        t = cov_radii[1] / sum(cov_radii)
                                        site = t * temp[0] + (1 - t) * temp[1]
                                else:
                                        site = calculate_centroid(temp, cov_radii, math.sqrt(com[1]))

                                # Define bond length
                                bond_len = min(cov_radii) + self.surface_add_radii if self.bond_len is None else self.bond_len
                                
                                # Check site validity (not too close to existing atoms)
                                flag = True
                                for ap in pos:
                                        if np.linalg.norm(ap - site) + 0.001 < bond_len * self.mul:
                                                flag = False
                                                break
                                
                                # Save valid site
                                if flag:
                                        sites.append({
                                        'site': 'inside',
                                        'type': 'inside',
                                        'normal': None,
                                        'position': site,
                                        'indices':[c for c in com[0]]
                                        })

                # Computing higher-dimensional sites using Rips Complex
                rc = gudhi.RipsComplex(points=pos, max_edge_length=self.radii)
                st = rc.create_simplex_tree(9)
                combinations = st.get_skeleton(9)
                
                # Process higher-dimensional combinations (> 4)
                for com in combinations:
                        if len(com[0]) > 4:
                                temp = pos[com[0]]
                                cov_radii = [covalent_radii[self.atoms[c].number] for c in com[0]]
                                site = calculate_centroid(temp, cov_radii, com[1] / 2)
                        
                        # Define bond length
                        bond_len = min(cov_radii) + self.surface_add_radii if self.bond_len is None else self.bond_len
                        
                        # Check site validity
                        flag = True
                        for ap in pos:
                                if np.linalg.norm(ap - site) + 0.001 < bond_len * self.mul:
                                        flag = False
                                        break
                        
                        # Save valid site
                        if flag:
                                sites.append({
                                'site': 'inside',
                                'type': 'inside',
                                'normal': None,
                                'position': site,
                                'indices':[c for c in com[0]]
                                })

                # Validate and add sites to `self.inside_site_list`
                if self.tol == False:
                        self.inside_site_list = sites
                else:
                        for site in sites:
                                flag = True
                                for s in self.inside_site_list:
                                        if np.linalg.norm(np.array(s['position']) - np.array(site['position'])) < self.tol:
                                                flag = False
                                if flag:
                                        self.inside_site_list.append(site)


        
        def get_inside_sites(self, absorbent=[]):
                """
                Identify and return the list of inside sites for the structure.

                Parameters:
                -----------
                - absorbent : list, optional
                        A list of element symbols representing adsorbent elements. If provided, 
                        the `inside_add_radii` parameter is adjusted based on their covalent radii.

                Returns:
                --------
                - inside_site_list : list
                        A list of dictionaries, each describing an inside site.
                """
                # Adjust the additional radius for inside sites if adsorbent elements are provided
                if absorbent:
                        self.inside_add_radii = min([
                        covalent_radii[atomic_numbers.get(ele, None)] 
                        for ele in absorbent
                        ])
                
                # Compute the inside sites
                self.inside_topo()
                
                # Return the list of inside sites
                return self.inside_site_list




class SlabAdsorptionsSitesFinder():
        """
        Parameters:
        -----------
        - atoms : ase.Atoms object
                The nanoparticle to use as a template to generate surface sites. 
                Accept any ase.Atoms object. 

        - adsorbate_elements : list of str
                If the sites are generated without considering adsorbates, 
                the symbols of the adsorbates are entered as a list, resulting in a clean slab.

        - bond_len : float or None , default None
                The sum of the atomic radii between a reasonable adsorption 
                substrate and the adsorbate.If not setting, the threshold for 
                the atomic hard sphere model is automatically calculated using 
                the covalent radius of ase.

        - mul : float , default 0.8
                A scaling factor for the atomic bond lengths used to modulate
                the number of sites generated, resulting in a larger number of potential sites. 

        - tol : float , default 0.6
                The minimum distance in Angstrom between two site for removal
                of partially overlapping site.

        - radius : float , default 5.0
                The maximum distance between two points 
                when performing persistent homology calculations. 

        - k : float , default 1.1
                Expand the key length, so as to calculate the length of the expansion
                in the direction of the normal vector, k value is too small will lead 
                to the calculation of the length of the expansion of the error, 
                you can according to the needs of appropriate increase

        - both_surface : bool , default False
                Input whether sites generation is performed on both the upper and 
                lower surfaces; when the input is False, only the upper surface is output.
        """
        def __init__(self,atoms,
                        adsorbate_elements=[],
                        mul=0.8,
                        tol=0.6,
                        bond_len=None,
                        k=1.1,
                        radius=5.,
                        both_surface = False
                        ):
                # Create a copy of the atoms object to avoid modifying the original
                atoms = atoms.copy()
                # Get the indices of all metal atoms by excluding atoms from adsorbate_elements
                self.metal_ids = [a.index for a in atoms if a.symbol not in adsorbate_elements]
                # Create a new Atoms object with only the metal atoms
                atoms = Atoms(atoms.symbols[self.metal_ids], 
                        atoms.positions[self.metal_ids], 
                        cell=atoms.cell, pbc=atoms.pbc) 
                # Initialize various attributes of the class with relevant data from the atoms object
                self.atoms = atoms
                self.positions = atoms.positions
                self.symbols = atoms.symbols
                self.numbers = atoms.numbers
                self.tol = tol
                self.k = k
                self.bond_len = bond_len
                self.mul = mul
                self.radii = radius
                self.cell = atoms.cell
                self.pbc = atoms.pbc
                self.metals = sorted(list(set(atoms.symbols)))
                self.both_surface = both_surface
                self.metal_elements = metal_elements

                # Set the default covalent radius for surface site calculations(default H)
                self.surface_add_radii = covalent_radii[1] 
                self.inside_add_radii = covalent_radii[1] 

                # Initialize an empty list to store site information
                self.surf_site_list = []
                self.inside_site_list = []
                self.sites = []
                self.surf_index = []
                
        def get_surface_atoms(self,cutoff=3, tolerance = 1e-3,both_surface=False):
                """The function to access to surface atoms by coordination number

                Parameters:    
                -----------
                - cutoff : float
                        Cutoff radius for neighbor search (in Ångströms)
                
                - tolerance : float 
                        If the magnitude of the summed normal vectors is less than this value, 
                        the atom is considered a non-surface atom

                - both_surface : boolen , default False
                        Whether to return the atoms of the lower surface, 
                        if True, the atoms of the upper and lower surfaces will be returned.
                
                Returns:
                --------
                - surf_ids : list
                        List of indices of surface atoms.

                - surface_atoms : ase.Atoms object
                        Atoms object containing only the detected surface atoms.
                """
                # Get the z-coordinates of all atoms
                pos = self.positions
                z_coords = pos[:, 2]

                num_atoms = len(self.atoms)
                # If all z-coordinates are identical, return all atoms as surface atoms
                if np.all(z_coords == z_coords[0]) :
                        return range(len(self.atoms)),self.atoms
                
                neighbor_list = NeighborList([cutoff / 2.0] * num_atoms, self_interaction=False, bothways=True)
                neighbor_list.update(self.atoms)
                surface_indices = []
                # Initialize variables to store surface atoms and their indices
                for i in range(num_atoms):
                        indices, offsets = neighbor_list.get_neighbors(i)
                        if len(indices) == 0:
                                surface_indices.append(i)
                                continue

                        pos_i = pos[i]
                        summed_normal = np.zeros(3)

                        for j, offset in zip(indices, offsets):
                                pos_j = pos[j] + np.dot(offset, self.cell)
                                normal = pos_j - pos_i
                                norm = np.linalg.norm(normal)
                                if norm > 1e-6:
                                        summed_normal += normal / norm  

                        if np.linalg.norm(summed_normal) > tolerance:
                                surface_indices.append(i)  # Add the atom to the surface Atoms object
                
                surface_atoms = self.atoms[surface_indices]
                # Get the z-coordinates of surface atoms and calculate the median z-coordinate
                z_positions = surface_atoms.positions[:, 2]
                median_z = np.median(z_positions) 
                # If both_surface is True, return all surface atoms and their indices
                if both_surface == True:
                        return surface_indices , surface_atoms
                
                # Otherwise, return only the atoms on the upper surface
                upper_index = []
                upper_atoms = Atoms()
                for i , atom in zip(surface_indices,surface_atoms):
                        if atom.position[2] > median_z: # Check if the atom is on the upper surface
                                upper_index.append(i)  # Add the atom index to the upper surface list
                                upper_atoms += atom  # Add the atom to the upper surface Atoms object
                return upper_index , upper_atoms         
        
        def expand_cell(self, cutoff=None, padding=[1,1,0]):
                """
                Return Cartesian coordinates of atoms within a supercell,
                which contains repetitions of the unit cell and at least one neighboring atom.
                Borrowed from Catkit.

                Parameters:
                -----------
                - cutoff : float, optional
                        A cutoff value to determine the maximum distance for expansion. If None, it is computed.

                - padding : list of int, optional, default [1,1,0]
                        Padding values for the expansion in each direction. These define the number of unit cell repetitions
                        along each of the three axes (x, y, z).
                """
                cell = self.atoms.cell  # Get the cell matrix (unit cell) and positions of the atoms
                pbc = [1, 1, 0]  # Define periodic boundary conditions (no periodicity in z-direction)
                pos = self.atoms.positions

                # If no padding or cutoff is provided, compute the cutoff based on the system geometry
                if padding is None and cutoff is None:
                        # Compute diagonal lengths of the unit cell
                        diags = np.sqrt((([[1, 1, 1],
                                        [-1, 1, 1],
                                        [1, -1, 1],
                                        [-1, -1, 1]]
                                        @ cell)**2).sum(1))
                        # If there is only one atom, set cutoff to half of the maximum diagonal length
                        if pos.shape[0] == 1:
                                cutoff = max(diags) / 2.
                        else:
                                # Compute the distances between atoms and update the cutoff
                                dpos = (pos - pos[:, None]).reshape(-1, 3)
                                Dr = dpos @ np.linalg.inv(cell)
                                D = (Dr - np.round(Dr) * pbc) @ cell
                                D_len = np.sqrt((D**2).sum(1))

                                cutoff = min(max(D_len), max(diags) / 2.)
                # Calculate the lattice lengths and volume of the unit cell
                latt_len = np.sqrt((cell**2).sum(1))
                V = abs(np.linalg.det(cell))

                # padding = pbc * np.array(np.ceil(cutoff * np.prod(latt_len) /
                                                # (V * latt_len)), dtype=int)
                # Create offset grid based on the provided padding in x, y, and z directions
                offsets = np.mgrid[-padding[0]:padding[0] + 1,
                                -padding[1]:padding[1] + 1,
                                -padding[2]:padding[2] + 1].T
                # Generate translation vectors by multiplying offsets with the cell matrix
                tvecs = offsets @ cell
                # Calculate the coordinates of atoms in the expanded supercell
                coords = pos[None, None, None, :, :] + tvecs[:, :, :, None, :]
                # Flatten the offsets and coordinates to return the expanded supercell structure
                ncell = np.prod(offsets.shape[:-1]) # Total number of unit cell repetitions
                index = np.arange(len(self.atoms))[None, :].repeat(ncell, axis=0).flatten()
                coords = coords.reshape(np.prod(coords.shape[:-1]), 3)
                offsets = offsets.reshape(ncell, 3)

                return index, coords, offsets
        
        def point_in_range(self,pos):
                """
                Determine if a given site (position) is within the current lattice boundaries.

                Parameters:
                -----------
                - pos : array
                        The Cartesian coordinates of the point to be checked.

                Returns:
                --------
                - bool
                        True if the point is within the lattice boundaries, False otherwise.
                """
                # Get the minimum and maximum z-coordinates of the atoms
                z_min = min(self.atoms.positions[:,2])
                z_max = max(self.atoms.positions[:,2])

                cell = self.cell # Get the cell dimensions (lattice vectors)
                # Check if the point lies within the boundaries along x, y, and z directions
                for i in range(3): 
                        if not (-0.1 <= pos[i] < cell[i, i]-0.1):  # Check x, y, and z against lattice boundaries
                                return False
                # Check if the z-coordinate is within the range of atomic position
                if pos[2]<z_min-0.1 or pos[2]>z_max+0.1: 
                        return False
                return True

                        
        def inside_topo(self):
                """ This is a function used to find embedding sites for various
                periodic structures.
                """
                # If the structure is not periodic, use the current atomic positions
                if not self.pbc.all():
                        pos = self.positions
                else:
                        # For periodic structures, expand the cell to account for neighboring atoms
                        index, pos, offsets = self.expand_cell()
                # Computing sites in 4 dimensions using alpha complexes
                ac = gudhi.AlphaComplex(points=pos)
                st = ac.create_simplex_tree((self.radii / 2) ** 2)    # Alpha complex with squared radii 
                combinations = st.get_skeleton(4) # Alpha complex with squared radii
                sites = [] # List to store identified embedding sites
                n = len(self.atoms) # Number of atoms
                kdtree = None  # To store the KDTree for fast distance checking
                # Iterate through the simplices of the Alpha Complex
                for com in combinations:
                        # Ensure the simplex is at least 2D and not on the surface
                        if len(com[0]) >= 2 and sorted([c % n for c in com[0]]) not in self.surf_index:
                                temp = pos[com[0]] # Extract positions of the simplex vertices
                                cov_radii = [covalent_radii[self.atoms[c%n].number] for c in com[0]] # Covalent radii of vertices
                                # Calculate the site position based on simplex vertices and radii
                                if len(com[0])==2: # If the simplex is 2D (edge)
                                        t = cov_radii[1]/sum(cov_radii)
                                        site = t * temp[0] + (1 - t) * temp[1]
                                else: # For higher-dimensional simplices
                                        site = calculate_centroid(temp,cov_radii,math.sqrt(com[1]))
                                # Skip the site if it's outside the current lattice boundaries
                                if not self.point_in_range(site):
                                        continue
                                # Determine bond length for proximity checks
                                if self.bond_len is None:
                                        bond_len = min(cov_radii)+self.surface_add_radii
                                else:
                                        bond_len = self.bond_len

                                # Create a KDTree once, and use it to check proximity for all points
                                if kdtree is None:
                                        kdtree = KDTree(pos)

                                # Check if any point is within bond_len range
                                nearby_points = kdtree.query_ball_point(site, bond_len * self.mul + 0.02)
                                if len(nearby_points) == 0 : # If no nearby points, add the site
                                        sites.append({
                                        'site': 'inside',
                                        'type': 'inside',
                                        'normal': None,
                                        'position': site,
                                        'indices':[c%n for c in com[0]]
                                        })                       
                # Compute loci larger than 4 dimensions using VR complex shapes
                rc = gudhi.RipsComplex(points=pos, max_edge_length=self.radii) # Rips Complex with max edge length
                st = rc.create_simplex_tree(9) # Build a simplex tree up to 10 dimensions
                combinations = st.get_skeleton(9) # Get simplices up to 10 dimensions
                combinations = sorted(list(combinations), key=lambda x: len(x[0]), reverse=True) # Sort simplices by size
                # Process high-dimensional simplices
                for com in combinations:
                        if len(com[0]) > 4: # Only consider simplices with more than 4 vertices
                                temp = pos[com[0]] # Extract positions of simplex vertices
                                cov_radii = [covalent_radii[self.atoms[c%n].number] for c in com[0]] # Covalent radii of vertices
                                site = calculate_centroid(temp,cov_radii,com[1] / 2)  # Calculate site position

                                # Skip the site if it's outside the current lattice boundaries
                                if not self.point_in_range(site):
                                        continue

                                # Determine bond length for proximity checks
                                if self.bond_len is None:
                                        bond_len = min(cov_radii)+self.surface_add_radii
                                else:
                                        bond_len = self.bond_len
                                # Check proximity using KDTree
                                nearby_points = kdtree.query_ball_point(site, bond_len * self.mul + 0.02)
                                if len(nearby_points) == 0 : # If no nearby points, add the site
                                        sites.append({
                                        'site': 'inside',
                                        'type': 'inside',
                                        'normal': None,
                                        'position': site,
                                        'indices':[c%n for c in com[0]]
                                        })
                # Filter and store sites based on tolerance criteria
                if self.tol is False: # If no tolerance filtering is required
                        for site in sites:
                                flag  = True
                                for s in self.surf_site_list:
                                                if np.linalg.norm(np.array(s['position']) - np.array(site['position'])) < self.tol:
                                                        flag = False
                                                        break
                                if flag == True:
                                                self.inside_site_list.append(site) 
                else: # Apply tolerance-based filtering
                        for site in sites:
                                flag  = True
                                 # Check against surface site list
                                for s in self.surf_site_list:
                                        if np.linalg.norm(np.array(s['position']) - np.array(site['position'])) < self.tol:
                                                flag = False
                                                break
                                if not flag:
                                        continue
                                # Check against existing inside site list
                                for s in self.inside_site_list:
                                        if np.linalg.norm(np.array(s['position']) - np.array(site['position'])) < self.tol:
                                                flag = False
                                                break
                                if flag == True:
                                        self.inside_site_list.append(site) 
        def expand_surface_cells(self,original_atoms,cell):
                """
                Return Cartesian coordinates of surface atoms within a supercell.
                
                Parameters:
                -----------
                - original_atoms : ase.Atoms object
                        The surface atoms that need to be expanded.
                
                - cell : list of float
                        Lattice vectors of the atomic structure.
                """
                # Extract lattice vectors in the x and y directions
                la_x = cell[0] # Lattice vector along x-direction
                la_y = cell[1] # Lattice vector along y-direction

                # Initialize a list to store new atom positions
                new_atoms = []

                # Define the expansion factor for the supercell
                factor = 3 # Supercell will be (3x3) in x and y directions

                # Supercell will be (3x3) in x and y directions
                offset = (factor - 1) / 2  # Offset to center the supercell expansion
                offset = int(offset) # Convert to integer for iteration

                # Iterate through shifts in x and y directions
                for dx in range(-offset, offset + 1): # Loop over x-axis offsets
                        for dy in range(-offset, offset + 1): # Loop over y-axis offsets
                                # Iterate through all atoms in the original structure
                                for atom in original_atoms:
                                        # Calculate the new position by adding lattice translations
                                        new_position = atom + dx*la_x + dy*la_y
                                        # Append the new position to the list
                                        new_atoms.append(new_position)
                return np.array(new_atoms)

        def calculate_normal_vector(self,positions):
                """
                Calculate the normal vector of a surface given three points.

                Parameters:
                -----------
                - positions : array-like, shape (3, 3)
                        The Cartesian coordinates of three points on the surface.

                Returns:
                --------
                - normal : numpy array
                        A normalized normal vector pointing outward from the surface.
                """
                # Compute two vectors lying on the surface
                vec1 = positions[1] - positions[0] # Vector from point 0 to point 1
                vec2 = positions[2] - positions[0] # Vector from point 0 to point 2

                # Calculate the cross product of the two vectors to get the normal vector
                normal = np.cross(vec1, vec2)
                # Check if the normal vector has zero magnitude (degenerate triangle)
                if np.linalg.norm(normal) == 0:
                        # If degenerate, return the default normal vector [0, 0, 1]
                        return np.array([0.,0.,1.])
                # Ensure the normal vector points upwards (positive z-component)
                if normal[2] < 0:
                        normal = -normal # Flip the normal vector
                # Normalize the normal vector to have a magnitude of 1
                normal /= np.linalg.norm(normal)
                return normal

        def extend_point_away(self,site,pos,center,height):
                """
                Offset a surface site along its normal vector direction.

                Parameters:
                -----------
                - site : ndarray
                        The position of the site to be offset.

                - pos : list of ndarray
                        The positions of the initial atoms used to generate the site.

                - center : ndarray
                        The center point of the slab (used to determine the direction of offset).

                - height : float
                        The distance to offset the site along the normal vector.

                Returns:
                --------
                - new_site : ndarray
                        The new position of the site after applying the offset.

                - normal_vector : ndarray
                        The normal vector used for the offset.
                """
                # Determine the direction of offset based on the site's position relative to the slab center
                if site[2] >= center[2]:
                        sign = 1 # Offset upwards if the site is above the slab center
                else:
                        sign = -1 # Offset downwards if the site is below the slab center
                if len(pos) == 3:
                        # If there are exactly 3 points, calculate the normal vector directly
                        normal_vector = self.calculate_normal_vector(pos)
                else:
                        # Indices of all unique triangles formed by 4 points
                        normal_vectors = []
                        index = [[0,1,2],[0,1,3],[0,2,3],[1,2,3]]
                        for ind in index:
                               # Calculate the normal vector for each triangle
                               vector = self.calculate_normal_vector(pos[ind])
                               normal_vectors.append(vector) 
                        # Compute the mean normal vector (average direction)
                        normal_vector = np.mean(normal_vectors, axis=0)

                return  site+normal_vector*height*sign , normal_vector*sign

        def surf_topo(self):
                # Identify surface atoms and their indices based on coordination number.
                surface_index , surface_atoms = self.get_surface_atoms(both_surface=self.both_surface)
                # Get positions of surface atoms depending on periodic boundary conditions (PBC).
                if self.pbc.all() == False:
                        coords = surface_atoms.get_positions()
                else:
                        coords = self.expand_surface_cells(surface_atoms.get_positions(),self.cell)

                # Create an AlphaComplex from surface atom coordinates.
                rc = gudhi.AlphaComplex(points=coords)
                st = rc.create_simplex_tree((self.radii/2)**2)   # Build simplex tree with radius constraint.

                # Retrieve simplices (points, edges, triangles, etc.) from the AlphaComplex.   
                combinations = st.get_skeleton(4)
                # Compute the center of mass for the entire structure.
                center = self.atoms.get_center_of_mass()
                # Initialize variables to store surface sites and helper data structures.
                sites= []
                combinations = sorted(list(combinations), key=lambda x: len(x[0]), reverse=True) # Sort simplices by size.
                del_bri_couple = [] # Store bridge sites to be deleted.
                n = len(surface_atoms) # Number of surface atoms.
                fold4_group = [] # Store  4-fold groups.
                tri_groups = [] # Store triangle groups.
                 # Iterate through all simplices in the AlphaComplex.
                for com in combinations :
                        if len(com[0])>2: # Process simplices with more than two points.
                                temp = coords[com[0]] # Get coordinates of the simplex vertices.
                                cov_radii = [covalent_radii[surface_atoms[c%n].number] for c in com[0]] # Get covalent radii.
                                site = calculate_centroid(temp,cov_radii,math.sqrt(com[1])) # Calculate centroid of the simplex.

                                # Skip sites above a certain height threshold.
                                if site[2] > max(temp[:,2]) + 0.1:
                                        continue
                                # Determine bond length threshold for neighboring atoms.
                                if self.bond_len is None:
                                        bond_len = max(cov_radii)+self.surface_add_radii
                                else:
                                        bond_len = self.bond_len
                                # Identify indices of neighboring atoms within the bond length.
                                temp_com = []
                                cov_radii = []
                                
                                for i,coord in enumerate(coords):
                                        if np.linalg.norm(site - coord) < bond_len:
                                                temp_com.append(i)
                                                j = i%len(surface_atoms)
                                                cov_radii.append(covalent_radii[surface_atoms[j].number])
                                # Process 4-fold sites.
                                if len(temp_com) == 4:
                                        site_type = '4fold'
                                        index_tuple = tuple(temp_com)
                                        if index_tuple in fold4_group:
                                                continue # Skip duplicate 4-fold groups.
                                        else:
                                                site = calculate_centroid(coords[temp_com],cov_radii,math.sqrt(com[1]))  # Recalculate centroid.
                                                fold4_group.append(index_tuple) # Add group to the processed list.

                                                # Determine bridge sites to delete for 4-fold groups.
                                                max_d = -1
                                                for ind in temp_com[1:]:
                                                        if np.linalg.norm(coords[temp_com[0]]-coords[ind]) > max_d:
                                                                max_d = np.linalg.norm(coords[temp_com[0]]-coords[ind])
                                                                temp_i = ind
                                                del_bri_couple.append((temp_com[0],temp_i))
                                                remain = []
                                                for i in temp_com:
                                                        if i != temp_com[0] and i != temp_i:
                                                                remain.append(i)
                                                del_bri_couple.append(tuple(remain))  
                                # Process triangular hollow sites.                    
                                elif len(temp_com)==3:
                                        site_type = 'hollow'
                                        tri_groups.append(temp_com) # Add to triangle groups.
                                else:
                                        continue
                                # Skip sites outside the valid range.
                                if not self.point_in_range(site):
                                        continue
                                # Determine the height offset for the site based on bond length.
                                if self.bond_len is None:
                                        bond_len = min(cov_radii)+self.surface_add_radii
                                else:
                                        bond_len = self.bond_len
                                try:       
                                        height = math.sqrt((bond_len*self.k)** 2-(com[1]))
                                except Exception as r:
                                        height = 0.1 # Default height for numerical issues.
                                # Offset site along the surface normal vector.
                                site , normal = self.extend_point_away(site,coords[temp_com],center,height)

                                # Check if the site is too close to any existing atom.
                                flag = True
                                for ap in coords:
                                        if np.linalg.norm(ap - site)+0.01 < bond_len*self.mul:
                                                flag = False
                                                break 
                                # Add valid sites to the list.
                                if flag :
                                        sites.append({
                                                'site':site_type,
                                                'type':'surface',
                                                'normal':normal,
                                                'position':site,
                                                'indices':[c%n for c in temp_com]
                                        })
                                        self.surf_index.append(sorted([surface_index[c%n] for c in com[0]]))
                        if len(com[0])==2: # Process bridge sites.
                                
                                temp = coords[com[0]]
                                if tuple(sorted(com[0])) in del_bri_couple:
                                        continue # Skip bridge sites that overlap with 4-fold sites.
                                lam = self.k # Default scaling factor for bond length.
                                for couple in tri_groups:
                                        if com[0][0] in couple and com[0][1] in couple:
                                                lam = 1.0 # Adjust factor for triangles.
                                                break             
                                cov_radii = [covalent_radii[surface_atoms[c%n].number] for c in com[0]]   # Get covalent radii.
                                t = cov_radii[1]/sum(cov_radii)  # Weight factor for interpolation.
                                site = t * temp[0] + (1 - t) * temp[1] # Interpolated position for bridge site.

                                # Skip sites outside the valid range.
                                if not self.point_in_range(site):
                                        continue
                                # Find neighboring atoms for normal vector calculation.
                                neigh_coords = []
                                for coord in coords:
                                        if np.linalg.norm(site - coord) < sum(cov_radii)*lam:
                                                neigh_coords.append(coord)
                                xyz = np.array(neigh_coords)

                                # Calculate surface normal vector.
                                normal = plane_normal(xyz)
                                center = self.atoms.get_center_of_mass()
                                if site[2] < center[2]:
                                        up = -1
                                else:
                                        up = 1
                                normal *= up  
                                # Determine height offset.
                                if self.bond_len is None:
                                        bond_len = min(cov_radii)+self.surface_add_radii
                                else:
                                        bond_len = self.bond_len
                                try:      
                                        height = math.sqrt((bond_len*self.k)** 2-(com[1]))
                                except Exception as r:
                                        height = 0.1
                                # Offset site along the normal vector.
                                site = site + normal*height
                                flag = True
                                # Check if the site is too close to any existing atom.
                                for ap in coords:
                                        if np.linalg.norm(ap - site)+0.01 < bond_len*self.mul:
                                                flag = False
                                                break 
                                # Add valid sites to the list.
                                if flag :
                                        sites.append({
                                                'site':'bridge',
                                                'type':'surface',
                                                'normal':normal,
                                                'position':site,
                                                'indices':[c%n for c in com[0]]
                                        })
                                        self.surf_index.append(sorted([surface_index[c%n] for c in com[0]]))
                        if len(com[0])==1:# Process top sites.
                                temp = coords[com[0]]
                                site = temp[0]
                                # Skip sites outside the valid range.
                                if not self.point_in_range(site):
                                        continue
                                metal = surface_atoms[com[0][0]%n].symbol
                                # Find neighboring atoms for normal vector calculation.
                                neigh_coords = []
                                for i,coord in enumerate(coords):
                                        neigh_len =covalent_radii[surface_atoms[com[0][0]%n].number] +covalent_radii[surface_atoms[i%n].number]
                                        if np.linalg.norm(site - coord) < neigh_len*self.k:
                                                neigh_coords.append(coord)
                                xyz = np.array(neigh_coords)
                                # Calculate surface normal vector.
                                normal = plane_normal(xyz)
                                center = self.atoms.get_center_of_mass()
                                if site[2] < center[2]:
                                        up = -1
                                else:
                                        up = 1
                                normal *= up  
                                # Determine height offset.
                                if self.bond_len is None:
                                        bond_len = min([covalent_radii[atomic_numbers.get(metal,None)]])+self.surface_add_radii
                                else:
                                        bond_len = self.bond_len
                                height = bond_len*self.k
                                # Offset site along the normal vector.
                                site = site+normal*height
                                flag = True
                                 # Check if the site is too close to any existing atom.
                                for ap in coords:
                                        if np.linalg.norm(ap - site)+0.01 < bond_len*self.mul:
                                                flag = False
                                                break 
                                # Add valid sites to the list.
                                if flag :
                                        sites.append({
                                                'site':'top',
                                                'type':'surface',
                                                'normal':normal,
                                                'position':site,
                                                'indices':[c%n for c in com[0]]
                                        })
                if self.tol == False:
                        self.surf_site_list = sites
                # Determine if the generating sites are too close together
                else:
                        for site in sites:
                                flag  = True
                                for s in self.surf_site_list:
                                        if np.linalg.norm(np.array(s['position']) - np.array(site['position'])) < self.tol:
                                                flag = False
                                                break
                                if flag == True:
                                        self.surf_site_list.append(site)  

        def get_inside_sites(self,absorbent = []):
                """Retrieve sites inside the structure using topology analysis.
    
                Parameters:
                -----------
                - absorbent : list, optional
                        List of chemical elements to be considered when adjusting radii.
                
                Returns:
                --------
                - inside_site_list :  list
                        A list of identified inside sites.
                """
                if absorbent: # If an absorbent list is provided, adjust the radii based on the minimum covalent radius of the absorbent elements.
                        self.inside_add_radii = min([covalent_radii[atomic_numbers.get(ele,None)] for ele in absorbent])
                self.inside_topo() # Perform topological analysis to find inside sites.
                return self.inside_site_list
        
        def get_surface_sites(self,absorbent = []):
                """Retrieve sites inside the structure using topology analysis.
    
                Parameters:
                -----------
                - absorbent : list, optional
                        List of chemical elements to be considered when adjusting radii.
                
                Returns:
                --------
                - surf_site_list : list
                        A list of identified surface sites.
                """
                # If an absorbent list is provided, adjust the radii based on the minimum covalent radius of the absorbent elements.
                if absorbent:
                        self.surface_add_radii = min([covalent_radii[atomic_numbers.get(ele,None)] for ele in absorbent])
                self.surf_topo() # Perform topological analysis to find surface sites
                return self.surf_site_list
        
        def get_sites(self,absorbent = []):
                """Retrieve all sites (surface and inside) for the structure.

                Parameters:
                -----------
                - absorbent : list, optional
                        List of chemical elements to adjust radii for site generation.

                Returns:
                --------
                - sites : list
                        A list of all sites (surface and inside) identified.
                """
                # If the sites have already been generated, return the cached result
                if self.sites:
                        return self.sites
                # If surface sites have not been generated, generate them using the given absorbent elements.
                if not self.surf_site_list:
                        self.get_surface_sites(absorbent=absorbent)
                # If inside sites have not been generated, generate them using the given absorbent elements.
                if not self.inside_site_list:
                        self.get_inside_sites(absorbent=absorbent)
                # Combine surface sites and inside sites into the full sites list.
                self.sites = self.surf_site_list + self.inside_site_list
                return self.sites
