#!/usr/bin/env python

"""
Introduction:
    AGL-TM: Algebraic Graph Learning for Transition Metal Complexes
    Batch Processor (C++ Accelerated) - Multi-Kernel Support

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
        
        # Load Data once here to prevent re-reading CSV inside loops
        print(f"Loading input data from {self.path_to_csv}...")
        self.df_pdbids = pd.read_csv(self.path_to_csv)
        
        self.final_feature_columns = [
            f"{pair}_{feat}" for pair in MASTER_ATOM_PAIRS for feat in AGL_FEATURE_NAMES
        ]

        if not os.path.exists(self.feature_folder):
            os.makedirs(self.feature_folder)

    def get_agl_features(self, parameters):
        # Target column handling
        target_col = 'tzvp_homo_lumo_gap'
        if target_col not in self.df_pdbids.columns:
            if 'pK' in self.df_pdbids.columns:
                target_col = 'pK'
        
        all_molecular_features = []
        valid_ids = []
        valid_targets = []

        # print(f"Processing molecules with C++ Backend...")
        
        for index, row in self.df_pdbids.iterrows():
            _id = row['id']
            target_val = row[target_col] * 1000 if target_col in self.df_pdbids.columns else 0

            # Path resolution
            xyz_file_flat = os.path.join(self.data_folder, f"{_id}.xyz")
            xyz_file_nested = os.path.join(self.data_folder, str(_id), f"{_id}.xyz")
            
            xyz_file = xyz_file_flat if os.path.exists(xyz_file_flat) else xyz_file_nested if os.path.exists(xyz_file_nested) else None
            
            if not xyz_file:
                # Silent skip to reduce log noise in batch mode
                continue

            try:
                # 1. Identify Metal
                metal_symbol = get_metal_symbol(xyz_file)
                if metal_symbol is None:
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

        return df_features

    def process_kernel(self, k_idx):
        """
        Helper function to run feature generation and saving for a single kernel index.
        """
        try:
            parameters = {
                'type': self.df_kernels.loc[k_idx, 'type'],
                'power': self.df_kernels.loc[k_idx, 'power'], 
                'tau': self.df_kernels.loc[k_idx, 'tau'],
                'cutoff': self.cutoff
            }

            print(f"--- Processing Kernel {k_idx}: {parameters['type']} (Power: {parameters['power']}, Tau: {parameters['tau']}) ---")
            
            t_start = time.time()
            df_features = self.get_agl_features(parameters)
            t_end = time.time()

            csv_name = ntpath.basename(self.path_to_csv).split('.')[0]
            output_file_name = f'{csv_name}_AGL_TM_CPP_k{k_idx}_c{self.cutoff}.csv'
            output_path = os.path.join(self.feature_folder, output_file_name)

            df_features.to_csv(output_path, index=False, float_format='%.5f')
            print(f"    -> Saved {len(df_features)} molecules to: {output_file_name} ({t_end - t_start:.2f}s)")
            
        except Exception as e:
            print(f"!!! CRITICAL ERROR on Kernel {k_idx}: {e}")

    def main(self):
        # Check if user wants a specific kernel or ALL kernels
        if self.kernel_index == -1:
            target_indices = self.df_kernels.index.tolist()
            print(f"No kernel specified. Running ALL {len(target_indices)} kernels found in CSV.")
        else:
            if self.kernel_index not in self.df_kernels.index:
                print(f"Error: Kernel index {self.kernel_index} not found in kernels.csv")
                return
            target_indices = [self.kernel_index]

        # Loop through the target indices
        for k_idx in target_indices:
            self.process_kernel(k_idx)


def get_args(args):
    parser = argparse.ArgumentParser(description="Get AGL-TM Features (C++)")

    # Set default to -1 to indicate 'ALL', made required=False
    parser.add_argument('-k', '--kernel-index', 
                        help='Kernel Index (row in kernels.csv). Omit to run ALL kernels.', 
                        type=int, default=-1, required=False)
    
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
    print(f'Total Session time: {time.time()-t0:.2f}s')