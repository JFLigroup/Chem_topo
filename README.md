# Chem_topo
`chem_topo` is a tool for topological analysis of nanomaterials and surfaces, combining persistent homology, path homology, and feature extraction methods based on geometry and graph theory. The tool is particularly suitable for analyzing adsorption sites in atomic structures, topological invariants, and their significance in surface catalysis.
***
## Project Introduction
This project aims to assist in identifying and quantifying key structural features on material surfaces or in nanoclusters through topological data analysis methods. Capabilities include:

+ Generate potential adsorption sites based on structure;

+ Analyze topological changes in kinetic trajectories;

+ Extract topological features for machine learning modeling.

***
## Code structure
```
chem_topo/
├── chem_topo/                  # Main Module
│   ├── adsorption_sites.py         # Adsorption site identification
│   ├── persistent_path_homology_cli.py  # CLI entry, calculate path homology features
│   ├── post_process.py             # Post-processing and visualization of coherence results
│   ├── topo_features.py            # Core topological feature extraction classes (including PathHomology, etc.)
│   ├── utils.py                    # General function tools
├── examples/                   
│   ├── 711.vasp, Pt55.vasp,PtKOH   # Example structure file
│   ├── result_0.npy        		# Run output example
├── docs/                      # Documentation building template
├── test/                      # Unit Testing
│   ├── pathhomology_test.py       

```
***
## Requirements
This project relies on the following third-party libraries:
```
pip install numpy scipy ase gudhi homcloud

```
Make sure you have installed:

+ Python 3.7+

+ ASE

+ GUDHI

+ HomCloud
***
## Instructions
### 1. Calculation of adsorption sites
```
from chem_topo.adsorption_sites import ClusterAdsorptionSitesFinder
finder = ClusterAdsorptionSitesFinder(atoms)
sites = finder.get_surface_sites()
```
### 2. Computing path coherence features
Topological feature extraction can be performed from the command line
```
python chem_topo/persistent_path_homology_cli.py --data your_points.csv --filtration_type distance --max_path 4
```
Or use the PathHomology class directly to extract features of continuous homology, path homology, or angle homology
```
from chem_topo.topo_features import PathHomology
betti_nums = PathHomology(max_distance=5.0).persistent_path_homology(
        cloudpoints, points_weight, max_path, filtration=None)
betti_nums = PathHomology().persistent_angle_path_homology(
        cloudpoints, points_weight, max_path)
betti_nums = PathHomology().persistent_homology(cloudpoints,max_path)
```
### 3. Post-processing dynamics trajectories

```
from chem_topo.post_process import AlphaComplexAnalyzer
analyzer = AlphaComplexAnalyzer(folder_path='.', file_name='XDATCAR')
analyzer.run()

```
***
## Key functions
+ **Adsorption site identification**: Automatically identify possible adsorption sites on the surface and subsurface based on atomic distance, bond length, and geometric rules;

+ **Topological feature extraction**: Supports multiple path homology features such as distance filtering, angle filtering, and path length filtering;

+ **Persistent homology analysis**: Extract topological invariants in trajectories through HomCloud to track structural evolution;

+ **Batch processing support**: Designed CLI and parallel interfaces to support processing of large numbers of structure or trajectory files;

+ **Feature output for machine learning**: Output Betti number spectrum, cyclic structure position, etc.

For comprehensive documentation, including installation, usage, and API references, please visit:  
[https://chem-topo.readthedocs.io/en/latest/](https://chem-topo.readthedocs.io/en/latest/)

