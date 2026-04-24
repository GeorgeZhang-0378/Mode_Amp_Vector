#!/usr/bin/env python3
"""
Mode_dir_Vector_search_v1.py

Version 1:
- Reads CASTEP-style .cell files.
- Supports %BLOCK LATTICE_CART and %BLOCK LATTICE_ABC.
- Reads %BLOCK POSITIONS_FRAC.
- Ignores SPIN=..., comments, SPECIES_POT, SYMMETRY_GENERATE, kpoints, etc.
- Checks that parent and distorted structures contain the same species and atom counts.
- Assumes atom i in parent maps to atom i in distorted.
- Computes displacement/mode direction vectors in Cartesian coordinates:

    r_distorted = r_parent + Q * mode_direction_vector

  so:

    mode_direction_vector = (r_distorted - r_parent) / Q

Future development:
- Add automatic atom mapping when atom order differs.
- Add support for multiple distorted structures.
- Add periodic-boundary-aware mapping using minimum-image convention.
- Add output to CSV / JSON / new .cell files.
"""

import numpy as np
from collections import Counter


# ============================================================
# Basic utilities
# ============================================================

def clean_line(line):
    """Remove inline comments and surrounding whitespace."""
    return line.split("#")[0].strip()


def find_block(lines, block_name):
    """
    Find a CASTEP block.

    Accepts:
        %BLOCK LATTICE_CART
        %ENDBLOCK LATTICE_CART

    Returns lines inside the block.
    """
    start_key = f"%BLOCK {block_name}".upper()
    end_key = f"%ENDBLOCK {block_name}".upper()

    inside = False
    block_lines = []

    for line in lines:
        stripped = line.strip()
        upper = stripped.upper()

        if upper.startswith(start_key):
            inside = True
            continue

        if inside and upper.startswith(end_key):
            return block_lines

        if inside:
            block_lines.append(line.rstrip("\n"))

    return None


def is_numeric_triplet(tokens):
    """Check whether first three tokens are floats."""
    if len(tokens) < 3:
        return False
    try:
        float(tokens[0])
        float(tokens[1])
        float(tokens[2])
        return True
    except ValueError:
        return False


# ============================================================
# Lattice parsing
# ============================================================

def lattice_abc_to_cart(a, b, c, alpha_deg, beta_deg, gamma_deg):
    """
    Convert lattice parameters to a 3x3 Cartesian lattice matrix.

    Rows are lattice vectors:
        a_vector
        b_vector
        c_vector
    """
    alpha = np.radians(alpha_deg)
    beta = np.radians(beta_deg)
    gamma = np.radians(gamma_deg)

    avec = np.array([a, 0.0, 0.0])

    bvec = np.array([
        b * np.cos(gamma),
        b * np.sin(gamma),
        0.0
    ])

    cx = c * np.cos(beta)

    cy = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)

    cz_squared = c**2 - cx**2 - cy**2
    if cz_squared < 0 and abs(cz_squared) < 1e-10:
        cz_squared = 0.0

    if cz_squared < 0:
        raise ValueError("Invalid LATTICE_ABC parameters: cannot construct real c-vector.")

    cvec = np.array([
        cx,
        cy,
        np.sqrt(cz_squared)
    ])

    return np.vstack([avec, bvec, cvec])


def parse_lattice(lines):
    """
    Parse either LATTICE_CART or LATTICE_ABC.

    Returns:
        lattice_matrix, lattice_type
    """
    cart_block = find_block(lines, "LATTICE_CART")
    abc_block = find_block(lines, "LATTICE_ABC")

    if cart_block is not None and abc_block is not None:
        raise ValueError("Both LATTICE_CART and LATTICE_ABC found. Please keep only one.")

    if cart_block is None and abc_block is None:
        raise ValueError("No LATTICE_CART or LATTICE_ABC block found.")

    if cart_block is not None:
        numeric_rows = []

        for line in cart_block:
            line_clean = clean_line(line)
            if not line_clean:
                continue

            tokens = line_clean.split()

            # Skip unit line such as:
            # ang
            # bohr
            if not is_numeric_triplet(tokens):
                continue

            numeric_rows.append([float(tokens[0]), float(tokens[1]), float(tokens[2])])

        if len(numeric_rows) != 3:
            raise ValueError(
                f"LATTICE_CART should contain exactly 3 numeric vector rows, found {len(numeric_rows)}."
            )

        return np.array(numeric_rows, dtype=float), "LATTICE_CART"

    if abc_block is not None:
        numeric_rows = []

        for line in abc_block:
            line_clean = clean_line(line)
            if not line_clean:
                continue

            tokens = line_clean.split()

            # Skip unit line such as:
            # ang
            if not is_numeric_triplet(tokens):
                continue

            numeric_rows.append([float(tokens[0]), float(tokens[1]), float(tokens[2])])

        if len(numeric_rows) != 2:
            raise ValueError(
                f"LATTICE_ABC should contain exactly 2 numeric rows, found {len(numeric_rows)}."
            )

        a, b, c = numeric_rows[0]
        alpha, beta, gamma = numeric_rows[1]

        lattice = lattice_abc_to_cart(a, b, c, alpha, beta, gamma)

        return lattice, "LATTICE_ABC"


# ============================================================
# Position parsing
# ============================================================

def parse_positions_frac(lines):
    """
    Parse %BLOCK POSITIONS_FRAC.

    Example accepted line:
        Mn 0.1 0.2 0.3 SPIN=3.000

    Returns list of dictionaries:
        {
            "species": "Mn",
            "frac": np.array([x, y, z])
        }
    """
    block = find_block(lines, "POSITIONS_FRAC")

    if block is None:
        raise ValueError("No POSITIONS_FRAC block found.")

    atoms = []

    for line in block:
        line_clean = clean_line(line)
        if not line_clean:
            continue

        tokens = line_clean.split()

        if len(tokens) < 4:
            continue

        species = tokens[0]

        try:
            x = float(tokens[1])
            y = float(tokens[2])
            z = float(tokens[3])
        except ValueError:
            continue

        atoms.append({
            "species": species,
            "frac": np.array([x, y, z], dtype=float)
        })

    if len(atoms) == 0:
        raise ValueError("POSITIONS_FRAC block was found, but no atoms were parsed.")

    return atoms


def frac_to_cart(frac, lattice):
    """
    Convert fractional coordinates to Cartesian coordinates.

    lattice rows are lattice vectors, so:
        r_cart = frac @ lattice
    """
    return frac @ lattice


# ============================================================
# Structure object
# ============================================================

def read_structure(filename):
    with open(filename, "r") as f:
        lines = f.readlines()

    lattice, lattice_type = parse_lattice(lines)
    atoms = parse_positions_frac(lines)

    species_counts = Counter(atom["species"] for atom in atoms)

    species_order = []
    for atom in atoms:
        if atom["species"] not in species_order:
            species_order.append(atom["species"])

    return {
        "filename": filename,
        "lattice": lattice,
        "lattice_type": lattice_type,
        "atoms": atoms,
        "species_counts": species_counts,
        "species_order": species_order,
        "total_atoms": len(atoms)
    }


def print_structure_summary(structure, label):
    print("\n" + "=" * 70)
    print(f"{label} structure read successfully.")
    print("=" * 70)
    print(f"File: {structure['filename']}")
    print(f"Lattice format: {structure['lattice_type']}")
    print(f"Number of elements: {len(structure['species_counts'])}")
    print(f"Total atoms: {structure['total_atoms']}")

    print("\nElement summary:")
    print(f"{'Element':<12}{'Atoms':>8}")
    print("-" * 20)

    for species in structure["species_order"]:
        print(f"{species:<12}{structure['species_counts'][species]:>8}")


# ============================================================
# Matching and mapping
# ============================================================

def compare_structures(parent, distorted):
    """
    Check species counts and total atom counts.
    """
    if parent["total_atoms"] != distorted["total_atoms"]:
        raise ValueError(
            f"ERROR: total atom number does not match. "
            f"Parent = {parent['total_atoms']}, distorted = {distorted['total_atoms']}."
        )

    if parent["species_counts"] != distorted["species_counts"]:
        raise ValueError(
            f"ERROR: species numbers do not match.\n"
            f"Parent counts: {dict(parent['species_counts'])}\n"
            f"Distorted counts: {dict(distorted['species_counts'])}"
        )

    for i, (p_atom, d_atom) in enumerate(zip(parent["atoms"], distorted["atoms"]), start=1):
        if p_atom["species"] != d_atom["species"]:
            raise ValueError(
                f"ERROR: line-by-line species mismatch at atom {i}.\n"
                f"Parent: {p_atom['species']}\n"
                f"Distorted: {d_atom['species']}\n"
                f"Version 1 requires the same atom ordering."
            )

    print("\nNumber match OK. Now mapping atoms line-by-line...")


def print_mapping_table(parent, distorted):
    print("\n" + "=" * 70)
    print("Line-by-line atom mapping")
    print("=" * 70)

    print(f"{'Index':>5}  {'Parent':<42}  {'Distorted':<42}")
    print("-" * 95)

    for i, (p_atom, d_atom) in enumerate(zip(parent["atoms"], distorted["atoms"]), start=1):
        sp = p_atom["species"]

        p = p_atom["frac"]
        d = d_atom["frac"]

        p_str = f"{sp} ({p[0]: .8f}, {p[1]: .8f}, {p[2]: .8f})"
        d_str = f"{sp} ({d[0]: .8f}, {d[1]: .8f}, {d[2]: .8f})"

        print(f"{i:5d}  {p_str:<42}  ->  {d_str:<42}")


def ask_for_Q():
    raw = input("\nWhat is the mode amplitude Q? Press Enter for default Q = 1: ").strip()

    if raw == "":
        return 1.0

    try:
        Q = float(raw)
    except ValueError:
        raise ValueError("Q must be a decimal number, e.g. 1, 0.5, 3.12. Fractions like 1/2 are not accepted.")

    if abs(Q) < 1e-14:
        raise ValueError("Q cannot be zero, because mode_direction_vector = displacement / Q.")

    return Q


def compute_mode_direction_vectors(parent, distorted, Q):
    """
    Compute:

        r_distorted_cart = r_parent_cart + Q * mode_direction_vector_cart

    Therefore:

        mode_direction_vector_cart = (r_distorted_cart - r_parent_cart) / Q
    """
    results = []

    for i, (p_atom, d_atom) in enumerate(zip(parent["atoms"], distorted["atoms"]), start=1):
        species = p_atom["species"]

        p_frac = p_atom["frac"]
        d_frac = d_atom["frac"]

        p_cart = frac_to_cart(p_frac, parent["lattice"])
        d_cart = frac_to_cart(d_frac, distorted["lattice"])

        displacement_cart = d_cart - p_cart
        mode_direction_vector_cart = displacement_cart / Q

        results.append({
            "index": i,
            "species": species,
            "parent_frac": p_frac,
            "distorted_frac": d_frac,
            "parent_cart": p_cart,
            "distorted_cart": d_cart,
            "displacement_cart": displacement_cart,
            "mode_direction_vector_cart": mode_direction_vector_cart
        })

    return results


def print_mode_direction_vectors(results, Q):
    print("\n" + "=" * 70)
    print("Calculated mode direction vectors")
    print("=" * 70)
    print(f"Using Q = {Q}")
    print("\nEquation used:")
    print("    r_distorted_cart = r_parent_cart + Q * mode_direction_vector_cart")
    print("    mode_direction_vector_cart = (r_distorted_cart - r_parent_cart) / Q")

    print("\nCartesian result in angstrom:")
    print(f"{'Index':>5} {'El':<4} {'Parent cart':<38} {'Mode direction vector cart':<38}")
    print("-" * 105)

    for item in results:
        i = item["index"]
        sp = item["species"]
        p = item["parent_cart"]
        v = item["mode_direction_vector_cart"]

        p_str = f"({p[0]: .8f}, {p[1]: .8f}, {p[2]: .8f})"
        v_str = f"({v[0]: .8f}, {v[1]: .8f}, {v[2]: .8f})"

        print(f"{i:5d} {sp:<4} {p_str:<38} + Q {v_str:<38}")


# ============================================================
# Main interactive program
# ============================================================

def main():
    print("=" * 70)
    print("Mode direction vector search - Version 1")
    print("=" * 70)

    parent_file = input("\nPlease input parent structure .cell file: ").strip()
    print("\nReading parent file...")
    parent = read_structure(parent_file)
    print_structure_summary(parent, "Parent")

    distorted_file = input("\nPlease input distorted structure .cell file: ").strip()
    print("\nReading distorted file...")
    distorted = read_structure(distorted_file)
    print_structure_summary(distorted, "Distorted")

    compare_structures(parent, distorted)
    print_mapping_table(parent, distorted)

    Q = ask_for_Q()

    results = compute_mode_direction_vectors(parent, distorted, Q)
    print_mode_direction_vectors(results, Q)

    print("\nDone.")


if __name__ == "__main__":
    main()
