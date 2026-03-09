import numpy
import numpy as np
import re

def parse_urgent_2d_from_harmonics(filepath, flag_full_grid=True):
    """
    Parse an URGENT .out file into numpy arrays.

    Returns
    -------
    harmonics : list[int]
        Harmonic numbers present (e.g. [1, 2, ..., 20]).
        Index 0 of the first axis corresponds to harmonics[0].
    POWER_DENSITY : ndarray, shape (n_harmonics, nX, nY)
    ENERGY        : ndarray, shape (n_harmonics, nX, nY)
        NaN for the summed "HARMONICS 1 TO N" block, which has no E column.
    X_grid        : ndarray, shape (nX, nY)   — unique X values broadcast over Y
    Y_grid        : ndarray, shape (nX, nY)   — unique Y values broadcast over X
    """

    fortran_float = re.compile(r'([+-]?\d+\.\d+)[Dd]([+-]?\d+)')

    def to_float(s):
        return float(fortran_float.sub(r'\1e\2', s))

    # ------------------------------------------------------------------ #
    # 1. Read the file and split into blocks
    # ------------------------------------------------------------------ #
    with open(filepath) as fh:
        text = fh.read()

    # Each block starts at "ANGULAR DISTRIBUTION - HARMONIC"
    block_pattern = re.compile(
        r'ANGULAR DISTRIBUTION - (HARMONIC\s+\d+|HARMONICS\s+\d+\s+TO\s+\d+)',
        re.IGNORECASE
    )
    block_starts = [(m.start(), m.group(1)) for m in block_pattern.finditer(text)]

    blocks = []
    for i, (start, label) in enumerate(block_starts):
        end = block_starts[i + 1][0] if i + 1 < len(block_starts) else len(text)
        blocks.append((label.strip(), text[start:end]))

    # ------------------------------------------------------------------ #
    # 2. Parse each block
    # ------------------------------------------------------------------ #
    # data_by_harmonic: dict harmonic_number -> {'x':[], 'y':[], 'e':[], 'pd':[]}
    data_by_harmonic = {}

    for label, chunk in blocks:
        # Determine harmonic number(s)
        m_single = re.match(r'HARMONIC\s+(\d+)', label, re.I)
        m_sum    = re.match(r'HARMONICS\s+(\d+)\s+TO\s+(\d+)', label, re.I)

        if m_single:
            harm = int(m_single.group(1))
            has_energy = True
        elif m_sum:
            harm = 0          # use 0 as the "summed" key
            has_energy = False
        else:
            continue

        rows_x, rows_y, rows_e, rows_pd = [], [], [], []

        for line in chunk.splitlines():
            parts = line.split()
            # Data lines: all parts are numeric
            if len(parts) < 4:
                continue
            try:
                vals = [to_float(p) for p in parts]
            except ValueError:
                continue

            if has_energy and len(vals) >= 4:
                # X  Y  E  POWER_DENSITY  [FLUX ...]
                rows_x.append(vals[0])
                rows_y.append(vals[1])
                rows_e.append(vals[2])
                rows_pd.append(vals[3])
            elif not has_energy and len(vals) >= 3:
                # X  Y  POWER_DENSITY  [FLUX ...]
                rows_x.append(vals[0])
                rows_y.append(vals[1])
                rows_e.append(np.nan)
                rows_pd.append(vals[2])

        if rows_x:
            data_by_harmonic[harm] = {
                'x':  np.array(rows_x),
                'y':  np.array(rows_y),
                'e':  np.array(rows_e),
                'pd': np.array(rows_pd),
            }

    # ------------------------------------------------------------------ #
    # 3. Build the 2-D grid (X, Y axes) from harmonic 1
    # ------------------------------------------------------------------ #
    ref = data_by_harmonic[1]          # harmonic 1 always present
    x_unique = np.unique(ref['x'])
    y_unique = np.unique(ref['y'])
    nX, nY   = len(x_unique), len(y_unique)

    X_grid, Y_grid = np.meshgrid(x_unique, y_unique, indexing='ij')  # (nX, nY)

    # ------------------------------------------------------------------ #
    # 4. Assemble 3-D arrays  (harmonic, X, Y)
    # ------------------------------------------------------------------ #
    # Keep only individual harmonics (exclude the summed block key=0)
    harmonics = sorted(k for k in data_by_harmonic if k > 0)

    POWER_DENSITY = np.full((len(harmonics), nX, nY), np.nan)
    ENERGY        = np.full((len(harmonics), nX, nY), np.nan)

    x_idx = {v: i for i, v in enumerate(x_unique)}
    y_idx = {v: i for i, v in enumerate(y_unique)}

    for h_i, h in enumerate(harmonics):
        d = data_by_harmonic[h]
        for x, y, e, pd in zip(d['x'], d['y'], d['e'], d['pd']):
            xi = x_idx[x]
            yi = y_idx[y]
            POWER_DENSITY[h_i, xi, yi] = pd
            ENERGY[h_i, xi, yi]        = e

    if not (numpy.isfinite(POWER_DENSITY)).all():
        print("parse_urgent_2d_from_harmonics: Cleaned infinities in POWER_DENSITY")
        POWER_DENSITY[~numpy.isfinite(POWER_DENSITY)] = 0.0
    if not (numpy.isfinite(ENERGY)).all():
        print("parse_urgent_2d_from_harmonics: Cleaned infinities in ENERGY")
        ENERGY[~numpy.isfinite(ENERGY)] = 0.0

    if flag_full_grid:
        POWER_DENSITY_f = _expand_quadrant(POWER_DENSITY)
        ENERGY_f = _expand_quadrant(ENERGY)
        X_f, Y_f = _full_grid(X_grid[:, 0], Y_grid[0, :])
        return harmonics, POWER_DENSITY_f, ENERGY_f, X_f, Y_f
    else:
        return harmonics, POWER_DENSITY, ENERGY, X_grid, Y_grid


# ------------------------------------------------------------------ #
# Four-quadrant expansion
# ------------------------------------------------------------------ #

def _expand_quadrant(arr: np.ndarray) -> np.ndarray:
    """
    Mirror a one-quadrant array (..., nX, nY) to all four quadrants.

    Assumes the last two axes are X (index 0 = x=0) and Y (index 0 = y=0).
    The origin row/column is included exactly once.

    Returns ndarray (..., 2*nX-1, 2*nY-1).
    """
    # Mirror X: flip rows [1:] and prepend  ->  -xmax ... 0 ... +xmax
    full_x = np.concatenate([np.flip(arr[..., 1:, :], axis=-2), arr], axis=-2)
    # Mirror Y: flip cols [1:] and prepend  ->  -ymax ... 0 ... +ymax
    full   = np.concatenate([np.flip(full_x[..., 1:], axis=-1), full_x], axis=-1)
    return full


def _full_grid(x_pos: np.ndarray, y_pos: np.ndarray):
    """
    Build full (X, Y) meshgrids from one-quadrant 1-D coordinate arrays.

    Parameters
    ----------
    x_pos : 1-D array  (x >= 0, first element = 0)
    y_pos : 1-D array  (y >= 0, first element = 0)

    Returns
    -------
    X, Y : ndarray  (2*nX-1, 2*nY-1)
    """
    x_full = np.concatenate([-x_pos[1:][::-1], x_pos])
    y_full = np.concatenate([-y_pos[1:][::-1], y_pos])
    return np.meshgrid(x_full, y_full, indexing='ij')


# ------------------------------------------------------------------ #
#
# ------------------------------------------------------------------ #
if __name__ == '__main__':
    import os

    filepath = 'urgent.out'
    harmonics, POWER_DENSITY, ENERGY, X_grid, Y_grid = parse_urgent_2d_from_harmonics(filepath)

    print(f"Harmonics      : {harmonics}")
    print(f"POWER_DENSITY  : shape={POWER_DENSITY.shape}, dtype={POWER_DENSITY.dtype}")
    print(f"ENERGY         : shape={ENERGY.shape},        dtype={ENERGY.dtype}")
    print(f"X_grid         : shape={X_grid.shape},        dtype={X_grid.dtype}")
    print(f"Y_grid         : shape={Y_grid.shape},        dtype={Y_grid.dtype}")

    # Quick sanity checks
    print(f"\nX unique values (first 5): {X_grid[:, 0][:5]}")
    print(f"Y unique values (first 5): {Y_grid[0, :][:5]}")
    print(f"POWER_DENSITY[0, 0, 0] (harm 1, x=0, y=0): {POWER_DENSITY[0, 0, 0]:.4f}")
    print(f"ENERGY[0, 0, 0]        (harm 1, x=0, y=0): {ENERGY[0, 0, 0]:.3f} eV")


    print("Load with:  data = np.load('urgent_arrays.npz')")
    print("            POWER_DENSITY = data['POWER_DENSITY']   # (harmonic, X, Y)")
    print("            ENERGY        = data['ENERGY']          # (harmonic, X, Y)")
    print("            X             = data['X']               # (X, Y)")
    print("            Y             = data['Y']               # (X, Y)")
    print("            harmonics     = data['harmonics']       # [1..20]")

    # print(X_grid.shape, Y_grid.shape, POWER_DENSITY.shape, ENERGY.shape)
    from srxraylib.plot.gol import plot_image

    plot_image(POWER_DENSITY.sum(axis=0), X_grid[:,0], Y_grid[0,:])
    # print(X_grid[:,0], Y_grid[0,:])


