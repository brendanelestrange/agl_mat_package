#!/usr/bin/env python

"""
Introduction:
    AGL-TM: Algebraic Graph Learning for Transition Metal Complexes
    Batch Processor (C++ Accelerated)

Author:
    Adapted for Transition Metals (XYZ support) by Brendan LeStrange
    Original Author: Masud Rana
"""

import sys
import os
import pandas as pd
import numpy as np
import ntpath
import argparse
import time

import agl_tmc_cpp 

TRANSITION_METALS = {
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "La", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Ac", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn"
}

# Must match the order in agl_tmc.cpp LIGAND_ELEMS
LIGAND_ATOMS = ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']

MASTER_ATOM_PAIRS = [f"{m}-{l}" for m in sorted(list(TRANSITION_METALS)) for l in sorted(LIGAND_ATOMS)]

AGL_FEATURE_NAMES = ['COUNTS', 'SUM', 'MIN', 'MAX', 'MEAN', 'MEDIAN', 'STD', 'VAR', 'NUM_EWS', 'SUM_SQUARED']


def get_metal_symbol(xyz_path):
    """
    Quickly scans XYZ file to find the transition metal symbol.
    Necessary because C++ returns raw features without labels.
    """
    try:
        with open(xyz_path, 'r') as f:
            lines = f.readlines()
            # Skip atom count (line 0) and comment (line 1)
            for line in lines[2:]:
                atom = line.split()[0]
                # Clean atom (remove digits, fix case)
                atom = ''.join([i for i in atom if not i.isdigit()])
                atom = atom.capitalize()
                
                if atom in TRANSITION_METALS:
                    return atom
    except Exception:
        return None
    return None


class AlgebraicGraphLearningFeatures:

    df_kernels = pd.read_csv('../utils/kernels.csv')

    def __init__(self, args):
        self.kernel_index = args.kernel_index
        self.cutoff = args.cutoff
        self.path_to_csv = args.path_to_csv
        self.matrix_type = args.matrix_type 
        self.data_folder = args.data_folder
        self.feature_folder = args.feature_folder
        
        self.final_feature_columns = [
            f"{pair}_{feat}" for pair in MASTER_ATOM_PAIRS for feat in AGL_FEATURE_NAMES
        ]

    def get_agl_features(self, parameters):
        df_pdbids = pd.read_csv(self.path_to_csv)
        
        # Target column handling
        target_col = 'tzvp_homo_lumo_gap'
        if target_col not in df_pdbids.columns:
            if 'pK' in df_pdbids.columns:
                target_col = 'pK'
        
        all_molecular_features = []
        valid_ids = []
        valid_targets = []

        print(f"Processing molecules with C++ Backend...")
        
        for index, row in df_pdbids.iterrows():
            _id = row['id']
            target_val = row[target_col] * 1000 if target_col in df_pdbids.columns else 0

            # Path resolution
            xyz_file_flat = os.path.join(self.data_folder, f"{_id}.xyz")
            xyz_file_nested = os.path.join(self.data_folder, str(_id), f"{_id}.xyz")
            
            xyz_file = xyz_file_flat if os.path.exists(xyz_file_flat) else xyz_file_nested if os.path.exists(xyz_file_nested) else None
            
            if not xyz_file:
                print(f"Skipping {_id}: File not found.")
                continue

            try:
                # 1. Identify Metal
                metal_symbol = get_metal_symbol(xyz_file)
                if metal_symbol is None:
                    print(f"Skipping {_id}: No transition metal found.")
                    continue

                # 2. Compute Features (C++ side)
                raw_feats = agl_tmc_cpp.get_tmc_scores(
                    xyz_file, 
                    float(parameters['cutoff']),
                    float(parameters['power']),
                    float(parameters['tau']), 
                    str(parameters['type'])
                )

                # 3. Map C++ Output to Sparse Row Dict
                row_dict = {}
                chunk_size = len(AGL_FEATURE_NAMES)
                
                for i, ligand in enumerate(LIGAND_ATOMS):
                    stats = raw_feats[i * chunk_size : (i + 1) * chunk_size]
                    pair_name = f"{metal_symbol}-{ligand}"
                    for k, stat_name in enumerate(AGL_FEATURE_NAMES):
                        row_dict[f"{pair_name}_{stat_name}"] = stats[k]

                # 4. Align to Master Columns
                row_vector = [row_dict.get(col, 0.0) for col in self.final_feature_columns]
                
                # Only if we reach here, we add the data to our results
                all_molecular_features.append(row_vector)
                valid_ids.append(_id)
                valid_targets.append(target_val)

            except Exception as e:
                print(f"Error processing {_id}: {e}")
                continue

        # Create DataFrame only from valid entries
        df_features = pd.DataFrame(all_molecular_features, columns=self.final_feature_columns)
        df_features.insert(0, 'ID', valid_ids)
        df_features.insert(1, 'Target', valid_targets)

        print(f"Successfully processed {len(valid_ids)} / {len(df_pdbids)} molecules.")
        return df_features

    def main(self):
        parameters = {
            'type': self.df_kernels.loc[self.kernel_index, 'type'],
            'power': self.df_kernels.loc[self.kernel_index, 'power'], # This is 'kappa' in C++
            'tau': self.df_kernels.loc[self.kernel_index, 'tau'],
            'cutoff': self.cutoff
        }

        print(f"--- Running Kernel {self.kernel_index}: {parameters['type']} (C++) ---")
        df_features = self.get_agl_features(parameters)

        csv_name = ntpath.basename(self.path_to_csv).split('.')[0]
        # Include 'cpp' in filename to distinguish
        output_file_name = f'{csv_name}_AGL_TM_CPP_k{self.kernel_index}_c{self.cutoff}.csv'
        output_path = os.path.join(self.feature_folder, output_file_name)

        if not os.path.exists(self.feature_folder):
            os.makedirs(self.feature_folder)
            
        df_features.to_csv(output_path, index=False, float_format='%.5f')
        print(f"Saved features to: {output_path}")


def get_args(args):
    parser = argparse.ArgumentParser(description="Get AGL-TM Features (C++)")

    parser.add_argument('-k', '--kernel-index', help='Kernel Index (row in kernels.csv)', type=int, required=True)
    parser.add_argument('-c', '--cutoff', help='Distance cutoff (Angstroms)', type=float, default=12.0)
    parser.add_argument('-f', '--path_to_csv', help='Path to CSV with IDs and Targets', required=True)
    parser.add_argument('-m', '--matrix_type', type=str,
                        help="Graph matrix (Legacy arg, not used in this C++ ver)", default='Laplacian')
    parser.add_argument('-dd', '--data_folder', type=str, help='Directory containing XYZ files', required=True)
    parser.add_argument('-fd', '--feature_folder', type=str, help='Output directory', required=True)

    return parser.parse_args(args)


def cli_main():
    args = get_args(sys.argv[1:])
    AGL_Features = AlgebraicGraphLearningFeatures(args)
    AGL_Features.main()


if __name__ == "__main__":
    t0 = time.time()
    cli_main()
    print(f'Total Elapsed time: {time.time()-t0:.2f}s')