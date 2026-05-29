import numpy as np
from pathlib import Path

amps = [-1.0, -2/3, -1/3, 0.0, 1/3, 2/3, 1.0]

parent_file = "parent.cell"
x2_file = "X2.cell"
x3_file = "X3.cell"

outdir = Path("mixed_X2_X3_7x7")
outdir.mkdir(exist_ok=True)


def read_cell(filename):
    """
    Read LATTICE_CART and POSITIONS_FRAC from a CASTEP .cell file.
    Assumes atoms are in the same order in parent, X2, and X3.
    """
    lines = Path(filename).read_text().splitlines()

    lattice_lines = []
    pos_data = []

    in_lattice = False
    in_pos = False

    for line in lines:
        stripped = line.strip()

        if stripped.upper().startswith("%BLOCK LATTICE_CART"):
            in_lattice = True
            lattice_lines.append(line)
            continue

        if stripped.upper().startswith("%ENDBLOCK LATTICE_CART"):
            lattice_lines.append(line)
            in_lattice = False
            continue

        if in_lattice:
            lattice_lines.append(line)
            continue

        if stripped.upper().startswith("%BLOCK POSITIONS_FRAC"):
            in_pos = True
            continue

        if stripped.upper().startswith("%ENDBLOCK POSITIONS_FRAC"):
            in_pos = False
            continue

        if in_pos:
            if not stripped or stripped.startswith("#"):
                continue

            parts = stripped.split()
            species = parts[0]
            xyz = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            extra = " ".join(parts[4:]) if len(parts) > 4 else ""
            pos_data.append((species, xyz, extra))

    return lattice_lines, pos_data


def min_image_delta(mode_pos, parent_pos):
    """
    Minimum-image fractional displacement.
    This handles atoms crossing 0/1 periodic boundaries.
    """
    return (mode_pos - parent_pos + 0.5) % 1.0 - 0.5


def amp_label(a):
    """
    Make safe filename labels.
    """
    if np.isclose(a, -1):
        return "m1"
    if np.isclose(a, -2/3):
        return "m2d3"
    if np.isclose(a, -1/3):
        return "m1d3"
    if np.isclose(a, 0):
        return "0"
    if np.isclose(a, 1/3):
        return "p1d3"
    if np.isclose(a, 2/3):
        return "p2d3"
    if np.isclose(a, 1):
        return "p1"
    return str(a).replace("-", "m").replace(".", "p")


lat_parent, parent = read_cell(parent_file)
lat_x2, x2 = read_cell(x2_file)
lat_x3, x3 = read_cell(x3_file)

if len(parent) != len(x2) or len(parent) != len(x3):
    raise ValueError("The three files do not contain the same number of atoms.")

species_parent = [p[0] for p in parent]
species_x2 = [p[0] for p in x2]
species_x3 = [p[0] for p in x3]

if species_parent != species_x2 or species_parent != species_x3:
    raise ValueError("Atom species/order do not match between parent, X2, and X3 files.")

parent_coords = np.array([p[1] for p in parent])
x2_coords = np.array([p[1] for p in x2])
x3_coords = np.array([p[1] for p in x3])

d_x2 = min_image_delta(x2_coords, parent_coords)
d_x3 = min_image_delta(x3_coords, parent_coords)

# Optional sanity check: print maximum fractional displacement in each mode.
print("Max |d_X2| fractional:", np.max(np.abs(d_x2)))
print("Max |d_X3| fractional:", np.max(np.abs(d_x3)))

for a in amps:
    for b in amps:
        mixed_coords = (parent_coords + a * d_x2 + b * d_x3) % 1.0

        fname = outdir / f"Ca3Mn2O7_X2_{amp_label(a)}_X3_{amp_label(b)}.cell"

        with open(fname, "w") as f:
            f.write("# Mixed-mode frozen structure generated from parent + alpha*X2 + beta*X3\n")
            f.write(f"# alpha_X2 = {a:.10f}\n")
            f.write(f"# beta_X3  = {b:.10f}\n\n")

            for line in lat_parent:
                f.write(line + "\n")

            f.write("\n%BLOCK POSITIONS_FRAC\n")

            for (species, _, extra), xyz in zip(parent, mixed_coords):
                if extra:
                    f.write(
                        f"{species:<2}  {xyz[0]: .15f}  {xyz[1]: .15f}  {xyz[2]: .15f}   {extra}\n"
                    )
                else:
                    f.write(
                        f"{species:<2}  {xyz[0]: .15f}  {xyz[1]: .15f}  {xyz[2]: .15f}\n"
                    )

            f.write("%ENDBLOCK POSITIONS_FRAC\n")

print(f"Generated {len(amps) * len(amps)} structures in: {outdir}")
