import time
import sys
import os
import numpy as np
import homcloud.interface as hc
from ase.io import read
from ase import Atoms

class AlphaComplexAnalyzer:
    """
    A class to perform Alpha Filtration and persistent homology analysis
    on atomic trajectory data using the HomCloud library.

    This class focuses on extracting β1 (1-dimensional) topological features
    and identifying corresponding atomic coordinates involved in persistent cycles.
    """
    def __init__(self, folder_path, file_name, batch_size=500):
        """
        Initialize the analyzer with dataset path and file information.

        Parameters:
        -----------
        - folder_path : str
                Path to the folder containing the XDATCAR trajectory file.

        - file_name :str 
                Name of the trajectory file.

        - batch_size : int
                Number of frames to process per batch (for parallel execution).
        """
        self.folder_path = folder_path
        self.file_name = file_name
        self.batch_size = batch_size
        self.frames = self._load_frames()
    
    def _load_frames(self):
        """
        Load the last 15,000 frames from the trajectory file using ASE.

        Returns:
        --------
        - Atoms : List 
                A list of ASE Atoms objects representing the frames.
        """
        all_frames = read(os.path.join(self.folder_path, self.file_name), index=':', format='vasp-xdatcar')[-15000:]
        return all_frames

    def _remove_atoms(self, atoms: Atoms, symbol: str):
        """
        Remove atoms of a specific element type from an ASE Atoms object.

        Parameters:
        -----------
        - atoms : ase.Atoms
                The atomic structure to modify.

        - symbol : str
                The chemical symbol of atoms to be removed.

        Returns:
        --------
        - Atoms : ase.Atoms
                The filtered atomic structure.
        """
        del atoms[[atom.index for atom in atoms if atom.symbol == symbol]]
        return atoms

    def _seek_point(self, pos, point_list):
        """
        Match coordinates of interest (from stable volume) to original input points.

        Parameters:
        -----------
        - pos : np.ndarray
                Original point cloud with weights as an appended dimension.

        - point_list : List[np.ndarray]
                Points returned from stable_volume analysis.

        Returns:
        --------
        - List[int]
                Indices of matched points in the original dataset.
        """
        result = []
        for index in range(len(pos)):
            points = pos[index][:-1]
            for target_point in point_list:
                if np.linalg.norm(points - target_point) < 1e-5:
                    result.append(index)
        return result

    def _process_frame(self, atoms,  frame_index, task_id,remove_atoms_symbols = ['Pt']):
        """
        Perform persistent homology analysis on a single frame.

        Parameters:
        -----------
        - atoms : ase.Atoms
                ASE object for a single frame.

        - remove_atoms_symbols : list
                list of symbols to remove.

        - frame_index : int
                Global index of the current frame.

        - task_id : int 
                Task index used to name intermediate files.

        Returns:
        --------
        - List : [frame_index, persistence data with associated point indices]
        """
        for symbol in remove_atoms_symbols:
                atoms = self._remove_atoms(atoms, symbol)
        pos = np.array(atoms.get_positions())[:-3, :]
        weight = np.empty(pos.shape[0])
        weight[:48] = 0.175**2
        weight[48:] = 0.775**2
        pos_weighted = np.hstack((pos, weight.reshape(-1, 1)))

        pd_data = []
        try:
            pdgm_path = f"pointcloud_{task_id}.pdgm"
            hc.PDList.from_alpha_filtration(pos_weighted, weight=True, save_to=pdgm_path, save_boundary_map=True)
            pdlist = hc.PDList(pdgm_path)
            pd1 = pdlist.dth_diagram(1)

            for birth, death in zip(pd1.births, pd1.deaths):
                pair = pd1.nearest_pair_to(birth, death)
                if pair.lifetime() < 1e-5:
                    continue
                stable_volume = pair.stable_volume(pair.lifetime() * 0.05)
                point_list = stable_volume.boundary_points()
                pd_data += [birth, death]
                pd_data += self._seek_point(pos_weighted, point_list)

        except Exception as e:
            print(f"Task_id {task_id}: Error at iteration {frame_index}, {e}")
        return [frame_index, pd_data]

    def run(self, task_index):
        """
        Execute persistent homology analysis for a batch of frames.

        Parameters:
        -----------
        - task_index : int
                The current task index used to determine frame range.
        """
        start = task_index * self.batch_size
        end = (task_index + 1) * self.batch_size
        frames_to_process = self.frames[start:end]

        results = []
        t0 = time.time()

        for i, atoms in enumerate(frames_to_process):
            frame_id = start + i
            result = self._process_frame(atoms, frame_id, task_index)
            results.append(result)
            t1 = time.time()
            print(f"Task_id {task_index}: Frame {frame_id} processed in {t1 - t0:.2f} seconds.")
            t0 = t1
            sys.stdout.flush()

        print(f"Task_id {task_index}: Frames {start} to {end} all finished.")
        np.save(os.path.join(self.folder_path, f'result_{task_index}.npy'), np.array(results, dtype=object))



