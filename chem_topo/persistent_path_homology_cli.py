from .topo_features import PathHomology
import numpy as np
import argparse
import sys
"""
Persistent Path Homology CLI Tool

This script provides a command-line interface to compute persistent path homology features 
from point cloud or directed graph data, supporting distance-based, angle-based, and 
default filtrations. Periodic boundary conditions (PBC) are also supported.

"""
def input_cloudpoints(data_file, save_path, args):
    import pandas as pd
    data = pd.read_csv(data_file, header=0, index_col=0).values
    cloudpoints = data[:, 0:-1]
    points_weight = data[:, -1]
    max_path = args.max_path
    if args.filtration_type == 'distance':
        if args.pbc == True:
            PH = PathHomology(cell=args.cell, pbc=[True, True, False])
        else:
            PH = PathHomology()
        define_filtration = np.arange(0, args.max_distance, 0.1)
        betti_num_all = PH.persistent_path_homology(
            cloudpoints, points_weight, max_path, filtration=define_filtration)
        result = {
            'max_distance': PH.max_distance,
            'betti_num': betti_num_all,
            'edges_num': PH.total_edges_num,
        }
        np.save(save_path, result, allow_pickle=True)
    elif args.filtration_type == 'angle':
        if args.pbc == True:
            PH = PathHomology(cell=args.cell, pbc=[True, True, False])
        else:
            PH = PathHomology()
        betti_num_all = PH.persistent_angle_path_homology(
            cloudpoints, points_weight, max_path, filtration_angle_step=args.angle_step)
        result = {
            'betti_num': betti_num_all,
            'initial_vector_x': PH.initial_vector_x,
            'initial_vector_y': PH.initial_vector_y,
            'initial_vector_z': PH.initial_vector_z,
            'edges_num': PH.total_edges_num,
        }
        np.save(save_path, result, allow_pickle=True)
    elif args.filtration_type == 'default':
        if args.pbc == True:
            PH = PathHomology(cell=args.cell, pbc=[True, True, False])
        else:
            PH = PathHomology()
        define_filtration = np.arange(0, args.max_distance, 0.1)
        betti_num_all = PH.compute_persistent_homology(
            cloudpoints, max_path,filtration=define_filtration)
        result = {
            'betti_num': betti_num_all,
        }
        np.save(save_path, result, allow_pickle=True)
    return None


def input_digraph(data_file, save_path, args):
    import pandas as pd
    data = pd.read_csv(data_file, header=0, index_col=0)
    col_name = list(data.columns)
    cloudpoints = data[col_name[0:-2]].dropna(axis=0).values
    start_n = data[col_name[-2]].dropna(axis=0).values
    end_n = data[col_name[-1]].dropna(axis=0).values
    all_edges = np.vstack([start_n, end_n]).T
    max_path = args.max_path
    if args.filtration_type == 'distance':
        if args.pbc == True:
            PH = PathHomology(cell=args.cell, pbc=[True, True, False],max_path=max_path)
        else:
            PH = PathHomology(max_path=max_path)
        betti_num_all = PH.persistent_path_homology_from_digraph(
            cloudpoints, all_edges, max_path)
        result = {
            'max_distance': PH.max_distance,
            'betti_num': betti_num_all,
            'edges_num': PH.total_edges_num,
        }
        np.save(save_path, result, allow_pickle=True)
    elif args.filtration_type == 'angle':
        if args.pbc == True:
            PH = PathHomology(cell=args.cell, pbc=[True, True, False])
        else:
            PH = PathHomology()
        betti_num_all = PH.persistent_angle_path_homology_from_digraph(
            cloudpoints, all_edges, max_path, filtration_angle_step=args.angle_step)
        result = {
            'betti_num': betti_num_all,
            'initial_vector_x': PH.initial_vector_x,
            'initial_vector_y': PH.initial_vector_y,
            'initial_vector_z': PH.initial_vector_z,
            'edges_num': PH.total_edges_num,
        }
        np.save(save_path, result, allow_pickle=True)
    elif args.filtration_type == 'default':
        if args.pbc == True:
            PH = PathHomology(cell=args.cell, pbc=[True, True, False])
        else:
            PH = PathHomology()
        define_filtration = np.arange(0, args.max_distance, 0.1)
        betti_num_all = PH.compute_persistent_homology(
            cloudpoints, max_path,filtration_values=define_filtration)
        result = {
            'betti_num': betti_num_all,
        }
        np.save(save_path, result, allow_pickle=True)
    return None


def main(args):
    if args.input_type == 'cloudpoints':
        input_cloudpoints(args.input_data, args.save_name, args)
    elif args.input_type == 'digraph':
        input_digraph(args.input_data, args.save_name, args)
    else:
        print('Interal program.')
    return None


def parse_args(args):
    parser = argparse.ArgumentParser(description='Angle, distance, -based persistent path homology')

    parser.add_argument('--input_type', default='No', type=str, choices=['cloudpoints', 'digraph', 'No'])
    parser.add_argument('--input_data', default='cloudpoints.csv', type=str,
                        help='If the input type is cloudpoints, the input data should be the csv file, which contians the '
                        'cloudpoints and weights with the shape n*m, the n means the number of the points, (m-1) is the '
                        'dimension of the points, the last column are treated as weights.'
                        'For the digraph, the format of the file is .csv. The contents of the file is cloudpoints and edges.'
                        'The last two columns are start point idx and end point idx of the edges. All indices are start from 0')
    parser.add_argument('--filtration_type', default='default', type=str, choices=['angle', 'distance','default'])
    parser.add_argument('--max_distance', default=5, type=float, help='if filtration_type is angle, it will be ignored')
    parser.add_argument('--angle_step', default=30, type=int, help='Int, divisible by 180. if filtration_type is distance, it will be ignored')
    parser.add_argument('--save_name', default='./', type=str)
    parser.add_argument('--max_path', default=2, type=int)
    parser.add_argument('--cell', nargs=3, type=float, 
                        help='Crystal cell size [a, b, c] in Angstroms for PBC. Required if --pbc is enabled.')
    parser.add_argument('--pbc', action='store_true',default=False,
                        help='Enable Periodic Boundary Conditions (PBC) for crystal structures')
    if args.pbc and not args.cell_size:
        parser.error("--cell is required when --pbc is enabled")
    args = parser.parse_args()
    return args


def cli_main():
    args = parse_args(sys.argv[1:])
    print(args)
    main(args)


if __name__ == "__main__":
    cli_main()
    print('End!')
