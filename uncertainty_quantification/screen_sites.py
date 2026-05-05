import pandas as pd
import numpy as np
from collections import defaultdict
import argparse
import os
from ase.io import read
from extract_env import extract_env
from pymatgen.io.ase import AseAtomsAdaptor
from ase.cell import Cell


def get_sorted_df(csv_path, per_image=10, per_element=10, total=200):
    """
    Screen atomic sites by uncertainty range with caps on:
      - per_image:   max sites selected from any single sample_idx
      - per_element: max sites selected for any single element type
      - total:       max total sites returned

    Sites are selected in descending Range order, with caps enforced
    dynamically so no single image or element dominates.

    Args:
        csv_path:    path to CSV with columns [sample_idx, atom_idx, element, U_lower, U_upper]
        per_image:   max sites to select from any one sample_idx
        per_element: max sites to select for any one element
        total:       max total sites to return

    Returns:
        result: pd.DataFrame of selected sites, sorted by Range descending
    """

    df = pd.read_csv(csv_path)
    df['Range'] = np.absolute(df['U_lower'] - df['U_upper'])
    df = df.sort_values('Range', ascending=False).reset_index(drop=True)

    image_counts = defaultdict(int)
    element_counts = defaultdict(int)

    selected_rows = []

    for _, row in df.iterrows():

        if len(selected_rows) >= total:
            break

        sample_idx = int(row['sample_idx'])
        element = int(row['element'])

        if image_counts[sample_idx] >= per_image:
            continue

        if element_counts[element] >= per_element:
            continue

        selected_rows.append(row)
        image_counts[sample_idx] += 1
        element_counts[element] += 1

    result = pd.DataFrame(selected_rows).reset_index(drop=True)
    result['sample_idx'] = result['sample_idx'].astype(int)
    result['atom_idx'] = result['atom_idx'].astype(int)
    result['element'] = result['element'].astype(int)

    # Summary
    print(f"Selected {len(result)} sites (cap: {total})")
    print(f"\nSites per image:\n{pd.Series(image_counts).sort_values(ascending=False)}")
    print(f"\nSites per element:\n{pd.Series(element_counts).sort_values(ascending=False)}")

    return result


def construct_and_write(atoms_list, final_df, to_dir, rc=5, box_length=10):
    aaa = AseAtomsAdaptor()

    cell = np.array([[box_length, 0, 0],
                     [0, box_length, 0],
                     [0, 0, box_length]])

    for _, row in final_df.iterrows():
        index = int(row['atom_idx'])
        element = int(row['element'])
        image = int(row['sample_idx'])

        cell_obj = Cell(cell)
        final_atoms_list = extract_env(original_atoms=atoms_list[image],
                                       rc=rc,
                                       atom_inds=[index],
                                       new_cell=cell_obj,
                                       extract_cube=True,
                                       min_dist_delete=0.7)

        write_path = os.path.join(to_dir, str(image), str(element), str(index))
        os.makedirs(write_path, exist_ok=True)

        structure = aaa.get_structure(final_atoms_list[0]).get_sorted_structure()
        print(f'Writing to {write_path}')
        structure.to(fmt='poscar', filename=os.path.join(write_path, 'POSCAR'))

    return


def parse_args():
    parser = argparse.ArgumentParser(
        description="Screen atomic sites by uncertainty range and write POSCAR files for selected environments."
    )
    parser.add_argument("csv_path", type=str,
                        help="Path to CSV with columns [sample_idx, atom_idx, element, U_lower, U_upper].")
    parser.add_argument("xyz_path", type=str,
                        help="Path to xyz trajectory file.")
    parser.add_argument("to_dir", type=str,
                        help="Output directory where selected sites and POSCARs will be written.")
    parser.add_argument("--per_image", type=int, default=60,
                        help="Max sites to select from any one sample_idx (default: 60).")
    parser.add_argument("--per_element", type=int, default=100,
                        help="Max sites to select for any one element (default: 100).")
    parser.add_argument("--total", type=int, default=200,
                        help="Max total sites to return (default: 200).")
    parser.add_argument("--rc", type=float, default=5.0,
                        help="Cutoff radius for environment extraction (default: 5.0).")
    parser.add_argument("--box_length", type=float, default=10.0,
                        help="Cubic box length for the new cell (default: 10.0).")
    parser.add_argument("--df_filename", type=str, default="selected_sites.csv",
                        help="Filename for the selected-sites CSV written to to_dir (default: selected_sites.csv).")
    return parser.parse_args()

def main():
    args = parse_args()
    
    final_df = get_sorted_df(args.csv_path,
                             per_image=args.per_image,
                             per_element=args.per_element,
                             total=args.total)
    
    os.makedirs(args.to_dir, exist_ok=True) 
    df_out_path = os.path.join(args.to_dir, args.df_filename)
    final_df.sort_values('Range', ascending=False).to_csv(df_out_path, index=False)
    print(f'Wrote selected sites dataframe to {df_out_path}')
    
    atoms_list = read(args.xyz_path, index=":")
    
    construct_and_write(atoms_list,
                        final_df,
                        args.to_dir,
                        rc=args.rc,
                        box_length=args.box_length)

if __name__ == "__main__":
    main()
