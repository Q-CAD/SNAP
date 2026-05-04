import sys
import os
import subprocess
import re
import numpy as np
from pathlib import Path
import argparse

# Subprocess Wrapper

def subprocess_wrapper(cmd_args, working_directory):
    try:
        subprocess.run(
            cmd_args,
            cwd=working_directory,
            check=True,              # <-- THIS is the key
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,  # capture error output
            text=True
        )
    except subprocess.CalledProcessError as e:
        if e.returncode == 1:
            print(
                "Exited with code 1.\n"
            )
        else:
            print("Workflow failed with unexpected error:")
        print(e.stderr)

# Find the .extxyz files in each subdirectory

def get_files(directory, filename='md_run.xyz'):
    all_paths = []
    for root, _, _ in os.walk(directory):
        path = os.path.join(root, filename)
        if os.path.exists(path):
            all_paths.append(path)

    return all_paths 

# Extract per-atom energy predictions using the same MACE model (write to subdirectory)

def per_atom_extraction(mace_model_path, xyz_paths, npz_filename):
    for xyz_path in xyz_paths:
        directory = str(Path(xyz_path).parent)
        npz_write_path = os.path.join(directory, npz_filename)
        if not os.path.exists(npz_write_path):
            cmd_args = ["run_embeddings_mace", "--sample", xyz_path, 
                    "--savedir", directory, 
                    "--model-size", mace_model_path, 
                    "--index", ":"]
            subprocess_wrapper(cmd_args, os.getcwd())
            print(f'{npz_filename} written to {directory}')
        else:
            print(f'{npz_write_path} exists')

    return 

# Run the trained GBM model on the per-atom uncertainties

def predict_uncertainties(gbm_model_path, npz_paths, estimators=100):
    gbm_model_directory = str(Path(gbm_model_path).parent)
    pattern = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
    s_floats = re.findall(pattern, Path(gbm_model_path).name)
    s_floats = [str(np.abs(float(s_float))) for s_float in s_floats]
   
    if len(s_floats) != 2:
        raise ValueError(f"Could not parse upper/lower alpha from filename: {Path(gbm_model_path).name}")

    for npz_path in npz_paths:
        directory = str(Path(npz_path).parent)
        cmd_args = ["run_gbm", "--embeddings", npz_path,
                    "--savedir", directory,
                    "--loaddir", gbm_model_directory,
                    "--upper-alpha", s_floats[1],
                    "--lower-alpha", s_floats[0],
                    "--estimators", str(estimators)]
        subprocess_wrapper(cmd_args, os.getcwd())
        print(f'UQ file written to {directory}')

    return

def main():

    parser = argparse.ArgumentParser(description="Run GBM MACE prediction pipeline on MD .xyz files.")
    parser.add_argument("--mace-model", required=True, type=str, help="Path to the MACE model file")
    parser.add_argument("--gbm-model", required=True, type=str, help="Path to the trained GBM model file")
    parser.add_argument("--xyz-directory", required=True, type=str, help="Directory containing .extxyz files")
    parser.add_argument("--xyz-filename", required=True, type=str, help="Filename of the .xyz file to search for in subdirectories")
    parser.add_argument("--estimators", default=100, type=int, help="Number of GBM estimators (default: 100)")
    args = parser.parse_args()  # don't forget this line

    if not os.path.exists(args.mace_model):
        raise FileNotFoundError(f"{args.mace_model} not found")

    if not os.path.exists(args.gbm_model):
        raise FileNotFoundError(f"{args.gbm_model} not found")

    if not os.path.isdir(args.xyz_directory):
        raise NotADirectoryError(f"{args.xyz_directory} not a directory")

    xyz_label = args.xyz_filename.replace('.xyz', '')
    npz_write_file = f"embedding_info_{xyz_label}.npz"
    uq_write_file = f"UQ_embedding_info_{xyz_label}.csv.gz"

    xyz_paths = get_files(args.xyz_directory, args.xyz_filename)
    per_atom_extraction(args.mace_model, xyz_paths, npz_write_file)

    npz_paths = get_files(args.xyz_directory, npz_write_file)
    predict_uncertainties(args.gbm_model, npz_paths, estimators=args.estimators)

    return 

if __name__ == "__main__":
    main()


