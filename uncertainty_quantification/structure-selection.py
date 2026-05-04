import sys
import pandas as pd
import os
import numpy as np
from glob import glob
from collections import defaultdict
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor
from pathlib import Path
import argparse

def sort_and_aggregate(path):
    data = pd.read_csv(path)
    agg = (
        data.groupby("sample_idx")
            .agg(
                U_upper_sum=("U_upper", "sum"),
                U_lower_sum=("U_lower", "sum"),
                n_atoms=("atom_idx", "count")
            )
            .reset_index()
    )
    agg['Norm_Range'] = np.abs((agg['U_lower_sum'] - agg['U_upper_sum'])) / agg['n_atoms']
    agg = agg.sort_values("Norm_Range", ascending=False).reset_index(drop=True)
    agg['Path'] = os.path.abspath(Path(path).parent)

    return agg


def compile_results(check_path, check_extension='*.csv.gz'):
    df = pd.DataFrame()

    for root, _, _ in os.walk(check_path):
        ext_file = list(glob(os.path.join(root, check_extension)))
        if ext_file:
            print(ext_file[0])
            data = sort_and_aggregate(ext_file[0])  # Assumes one per directory
            df = pd.concat([df, data], ignore_index=True)

    df = df.sort_values("Norm_Range", ascending=False).reset_index(drop=True)
    return df

def select_structures(df, total=80, score_cap=10, max_per_label=6,
                      index_distance=1, filename='md_run.traj', top_ranked=False):

    aaa = AseAtomsAdaptor()

    # --- Select labels and images based on mode ---
    if top_ranked:
        filtered = df[df['Norm_Range'] <= score_cap].reset_index(drop=True)
        if len(filtered) < total:
            print(f"Warning: only {len(filtered)}/{total} structures found below score_cap={score_cap}")
        filtered = filtered.head(total)

        selected_labels = filtered['Path'].tolist()
        selected_images = [int(i) for i in filtered['sample_idx'].tolist()]
        selected_scores = filtered['Norm_Range'].tolist()

    else:
        label_images = defaultdict(list)
        label_counts = defaultdict(int)
        selected_labels, selected_images, selected_scores = [], [], []

        for i in range(len(df)):
            if len(selected_labels) >= total:
                break

            label = df['Path'].iloc[i]
            norm_range = df['Norm_Range'].iloc[i]
            sample_idx = int(df['sample_idx'].iloc[i])

            if norm_range > score_cap:
                continue
            if label_counts[label] >= max_per_label:
                continue
            if not all(abs(sample_idx - p) > index_distance for p in label_images[label]):
                continue

            selected_labels.append(label)
            selected_images.append(sample_idx)
            selected_scores.append(norm_range)

            label_images[label].append(sample_idx)
            label_counts[label] += 1

        if len(selected_labels) < total:
            print(f"Warning: only {len(selected_labels)}/{total} structures found below score_cap={score_cap}")

    # --- Read structures (shared by both branches) ---
    selected_structures = []
    for label, image in zip(selected_labels, selected_images):
        atoms = read(os.path.join(label, filename), index=image)
        selected_structures.append(aaa.get_structure(atoms).get_sorted_structure())

    return selected_labels, selected_images, selected_structures, selected_scores

def main():
    parser = argparse.ArgumentParser(description="Compile Quantile predictions and select the most uncertain structures.")

    # Original sys.argv arguments
    parser.add_argument("--check-path", required=True, type=str, help="Path to directory containing UQ results to compile")
    parser.add_argument("--to-path", required=True, type=str, help="Path to directory to write selected structures")

    # New arguments
    parser.add_argument("--image-multiplier", default=250, type=int, help="Multiplier applied to structure index for image numbering (default: 250)")
    parser.add_argument("--write-path-depth", default=8, type=int, help="Index into path parts for constructing output subdirectory structure (default: 8)")
    parser.add_argument("--n-structures", default=80, type=int, help="Total number of structures to select (default: 80)")
    parser.add_argument("--score-cap", default=1.0, type=float, help="Maximum UQ score threshold for structure selection (default: 1.0)")
    parser.add_argument("--max-per-label", default=6, type=int, help="Maximum number of structures to select per label/run type (default: 6)")
    parser.add_argument("--index-distance", default=1, type=int, help="Minimum index distance between selected structures (default: 1)")
    parser.add_argument("--structure-filename", default='md_run.traj', type=str, help="Filename of structure file to read with ASE (default: 'md_run.traj')")
    parser.add_argument("--top-ranked", action="store_false", default=True, help="If set, strictly select the top ranked images by UQ")
    args = parser.parse_args()

    # Path validation
    if not os.path.exists(args.check_path):
        raise FileNotFoundError(f"check-path not found: {args.check_path}")
    if not os.path.exists(args.to_path):
        raise FileNotFoundError(f"to-path not found: {args.to_path}")

    data = compile_results(args.check_path)

    s_paths, s_idxs, s_structs, s_ranges = select_structures(
        data,
        total=args.n_structures,
        score_cap=args.score_cap,
        max_per_label=args.max_per_label,
        index_distance=args.index_distance,
        filename=args.structure_filename,
        top_ranked=args.top_ranked
    )

    for i, s_path in enumerate(s_paths):
        scaled_image = s_idxs[i] * args.image_multiplier
        print(f'Path: {s_path}, Image: {scaled_image}, Range: {s_ranges[i]}')
        s_path_p = Path(s_path)
        s_path_parts = s_path_p.parts
        tail = os.path.join(args.to_path, *s_path_parts[args.write_path_depth:], str(scaled_image))
        os.makedirs(tail, exist_ok=True)
        s_structs[i].to(fmt='poscar', filename=os.path.join(tail, "POSCAR"))

    return 


if __name__ == '__main__':
    main() 
