"""
env_extraction.py
-----------------
Utilities for extracting and analyzing local atomic coordination environments
from ASE Atoms objects.

Functions
---------
extract_env         : Extract local environments around specified atom indices
find_central_atom   : Find the central atom in an extracted environment
get_ith_shell       : Get atom indices in the Nth nearest-neighbor shell

Internal helpers
----------------
_get_rdf            : Compute the radial distribution function
_find_peaks_and_valleys : Locate peaks and valleys in an RDF
_find_collisions    : Detect unphysically close atom pairs at boundaries
"""

from typing import Optional

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.neighborlist import NeighborList
from scipy.signal import argrelextrema, savgol_filter


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class CellTooSmallError(Exception):
    """Raised when the requested extraction cell is incompatible with the
    source structure or the specified cutoff radius."""
    pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_env(
    original_atoms: Atoms,
    rc: float,
    atom_inds: list[int],
    new_cell: np.ndarray,
    extract_cube: Optional[bool] = False,
    min_dist_delete: Optional[float] = 0.7,
) -> list[Atoms]:
    """Extract local environments around specified atoms.

    For each atom index in ``atom_inds``, builds a new Atoms object containing
    all neighbours within the cutoff radius ``rc``, embedded in ``new_cell``.
    Atoms in the central sphere are frozen via :class:`ase.constraints.FixAtoms`.

    Parameters
    ----------
    original_atoms : Atoms
        Source configuration from which environments are extracted.
    rc : float
        Cutoff radius in Ångströms. Only atoms within this distance of the
        central atom are included (and constrained) in the output.
    atom_inds : list[int]
        Zero-based indices of the atoms whose environments are to be extracted.
        A bare ``int`` is also accepted and will be wrapped in a list.
    new_cell : np.ndarray
        3×3 cell matrix for the output structures. Must be cubic (all diagonal
        norms equal).
    extract_cube : bool, optional
        If ``True``, include *all* atoms within the cubic cell in the output,
        not just those within ``rc``. Atoms outside ``rc`` are not constrained.
        Default ``False``.
    min_dist_delete : float, optional
        Minimum allowed interatomic distance in Ångströms. Pairs of unconstrained
        atoms closer than this threshold are considered collisions; the atom
        involved in the most collisions is iteratively removed until no
        collisions remain. Set to ``0`` to disable collision removal.
        Default ``0.7``.

    Returns
    -------
    list[Atoms]
        One Atoms object per entry in ``atom_inds``, each centred on the
        corresponding atom.

    Raises
    ------
    ValueError
        If ``new_cell`` is not 3×3, or if it is not cubic.
    CellTooSmallError
        If ``new_cell`` is larger than the source cell, if ``rc`` exceeds the
        maximum cell vector length, or if ``2 * rc > max_cell_len``.
    """
    if new_cell.size != 9:
        raise ValueError("Expected a 3×3 cell matrix.")

    cell_norms = np.linalg.norm(new_cell, 2, 1)
    max_cell_len = np.max(new_cell)

    if not (cell_norms[0] == cell_norms[1] == cell_norms[2]):
        raise ValueError(
            "new_cell is not cubic. Only cubic extraction cells are supported."
        )

    src_par = original_atoms.cell.cellpar()
    if (
        max_cell_len > src_par[0]
        or max_cell_len > src_par[1]
        or max_cell_len > src_par[2]
    ):
        raise CellTooSmallError(
            "Requested extraction cell is larger than the source structure."
        )
    if max_cell_len < rc:
        raise CellTooSmallError(
            "rc exceeds the maximum cell vector length of the extraction cell."
        )
    if rc * 2 > max_cell_len:
        raise CellTooSmallError(
            "2 * rc is greater than the extraction cell side length."
        )

    if isinstance(atom_inds, int):
        atom_inds = [atom_inds]

    n_atoms = len(original_atoms)
    cutoffs = (0.5 * max_cell_len * np.ones(n_atoms)).tolist()
    nl = NeighborList(cutoffs, self_interaction=True, bothways=True)
    nl.update(original_atoms)

    subcells = []

    for atom_ind in atom_inds:
        indices, offsets = nl.get_neighbors(atom_ind)

        neigh_disp = np.zeros((len(offsets), 3))
        neigh_symbols = []

        for neigh_idx, (i, offset) in enumerate(zip(indices, offsets)):
            neigh_disp[neigh_idx] = (
                original_atoms.positions[i]
                + offset @ original_atoms.get_cell()
                - original_atoms.positions[atom_ind]
            )
            neigh_symbols.append(original_atoms.symbols[i])

        # --- atoms within the cubic cell ---
        in_cube = np.where(
            np.all(np.abs(neigh_disp) <= max_cell_len / 2, axis=1)
        )[0]
        cube_disp = neigh_disp[in_cube]
        cube_symbols = [neigh_symbols[int(i)] for i in in_cube]

        # --- atoms within the spherical cutoff ---
        in_rc = np.where(np.linalg.norm(cube_disp, axis=1) <= rc)[0]
        sphere_disp = cube_disp[in_rc]
        sphere_symbols = [cube_symbols[int(i)] for i in in_rc]

        # --- build output Atoms object ---
        if extract_cube:
            new_pos = cube_disp
            new_symbols = cube_symbols
            ind_fix = in_rc
        else:
            new_pos = sphere_disp
            new_symbols = sphere_symbols
            ind_fix = np.arange(len(new_pos), dtype=int)

        new_pos = new_pos + cell_norms / 2  # shift origin to box centre

        new_atoms = Atoms(
            symbols=new_symbols,
            positions=new_pos,
            cell=new_cell,
            pbc=True,
        )
        new_atoms.set_constraint(FixAtoms(indices=ind_fix))

        # --- remove boundary collisions (cube mode only) ---
        if extract_cube and min_dist_delete > 0:
            total, per_atom = _find_collisions(new_atoms, min_dist_delete)
            while total > 0:
                del new_atoms[int(np.argmax(per_atom))]
                total, per_atom = _find_collisions(new_atoms, min_dist_delete)

        subcells.append(new_atoms)

    return subcells


def find_central_atom(config: Atoms, side_size: float) -> int:
    """Return the index of the central atom in an extracted environment.

    :func:`extract_env` places the central atom at the geometric centre of the
    cubic cell, i.e. at coordinates ``(side_size/2, side_size/2, side_size/2)``.
    This function locates that atom by position.

    Parameters
    ----------
    config : Atoms
        Extracted subcell returned by :func:`extract_env`.
    side_size : float
        Side length of the cubic cell in Ångströms.

    Returns
    -------
    int
        Index of the central atom within ``config``.
    """
    half = side_size / 2.0
    for i, pos in enumerate(config.positions):
        if all(coord == half for coord in pos):
            return i
    return None


def get_ith_shell(
    config: Atoms,
    central_atom_index: int,
    shell_index: int,
) -> np.ndarray:
    """Return the indices of atoms in the Nth nearest-neighbour shell.

    The shell boundary is determined from the radial distribution function
    (RDF) computed up to 10 Å. The RDF is smoothed with a Savitzky–Golay
    filter before peak/valley detection.

    Parameters
    ----------
    config : Atoms
        Atomic configuration (typically an extracted subcell).
    central_atom_index : int
        Index of the atom whose neighbours are sought.
    shell_index : int
        Shell number, starting from 1 (first nearest neighbours).

    Returns
    -------
    np.ndarray
        Indices of atoms belonging to the requested shell.

    Raises
    ------
    ValueError
        If ``shell_index < 1``.
    RuntimeError
        If fewer than ``shell_index`` shells are found within 10 Å.
    """
    if shell_index < 1:
        raise ValueError("shell_index must be at least 1.")

    r, rdf = _get_rdf(config, r_max=5.0, dr=0.1)
    rdf_smooth = savgol_filter(rdf, window_length=10, polyorder=3)
    _, valley_idxs = _find_peaks_and_valleys(rdf_smooth)

    if shell_index > len(valley_idxs):
        raise RuntimeError(
            f"Requested shell {shell_index} but only {len(valley_idxs)} "
            "shells found within 10 Å."
        )

    shell_distance = r[valley_idxs[shell_index - 1]]
    cutoffs = [0.5 * shell_distance] * len(config)
    nl = NeighborList(cutoffs, skin=0.0, bothways=True, self_interaction=True)
    nl.update(config)
    indices, _ = nl.get_neighbors(central_atom_index)
    return indices


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_rdf(
    config: Atoms,
    r_max: float,
    dr: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the radial distribution function for a configuration.

    Parameters
    ----------
    config : Atoms
        Atomic configuration.
    r_max : float
        Maximum distance in Ångströms.
    dr : float
        Bin width in Ångströms.

    Returns
    -------
    r : np.ndarray
        Bin centres.
    rdf : np.ndarray
        Normalised RDF values.
    """
    n_atoms = len(config)
    positions = config.get_positions()
    cell = config.get_cell()

    cutoffs = [0.5 * r_max] * n_atoms
    nl = NeighborList(cutoffs, skin=0.0, bothways=True, self_interaction=False)
    nl.update(config)

    bins = np.arange(0, r_max + dr, dr)
    rdf_hist = np.zeros(len(bins) - 1)

    for i in range(n_atoms):
        indices, offsets = nl.get_neighbors(i)
        for j, offset in zip(indices, offsets):
            if i == j:
                continue
            dist = np.linalg.norm(
                positions[j] + np.dot(offset, cell) - positions[i]
            )
            if dist < r_max:
                bin_idx = int(dist // dr)
                if bin_idx < len(rdf_hist):
                    rdf_hist[bin_idx] += 1

    r = 0.5 * (bins[:-1] + bins[1:])
    shell_volumes = 4.0 / 3.0 * np.pi * (bins[1:] ** 3 - bins[:-1] ** 3)
    number_density = n_atoms / config.get_volume()
    rdf = rdf_hist / (n_atoms * shell_volumes * number_density)
    return r, rdf


def _find_peaks_and_valleys(
    rdf: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Locate peaks and valleys in a smoothed RDF array.

    Searches only from the global maximum onwards, so transient low-r
    noise does not contaminate the shell detection.

    Parameters
    ----------
    rdf : np.ndarray
        Smoothed RDF values.

    Returns
    -------
    peak_indices : np.ndarray
        Indices of peaks at or after the global maximum.
    valley_indices : np.ndarray
        Indices of valleys after the global maximum.
    """
    max_peak_idx = int(np.argmax(rdf))
    peak_indices = argrelextrema(rdf, np.greater)[0]
    valley_indices = argrelextrema(rdf, np.less)[0]

    return (
        peak_indices[peak_indices >= max_peak_idx],
        valley_indices[valley_indices > max_peak_idx],
    )


def _find_collisions(
    atoms: Atoms,
    min_dist: float,
) -> tuple[int, np.ndarray]:
    """Detect unphysically close atom pairs, ignoring constrained atoms.

    Parameters
    ----------
    atoms : Atoms
        Configuration to inspect.
    min_dist : float
        Distance threshold in Ångströms; pairs at or below this value are
        flagged as collisions.

    Returns
    -------
    total_collisions : int
        Total number of unique colliding pairs.
    num_collisions_per_atom : np.ndarray
        Per-atom collision count (constrained atoms set to zero).
    """
    n_atoms = len(atoms)
    cutoffs = (0.5 * min_dist * np.ones(n_atoms)).tolist()
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True, skin=0)
    nl.update(atoms)

    connect_mat = nl.get_connectivity_matrix(sparse=False)
    num_collisions_per_atom = np.sum(connect_mat, axis=1)

    for constraint in atoms.constraints:
        num_collisions_per_atom[constraint.get_indices()] = 0

    total_collisions = int(np.sum(num_collisions_per_atom) / 2)
    return total_collisions, num_collisions_per_atom
