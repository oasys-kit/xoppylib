import numpy as np
import re


def parse_urgent_2d_from_harmonics(filepath, flag_full_grid=True):
    """
    Parse an URGENT .out file into numpy arrays.

    Returns
    -------
    harmonics     : list[int]               e.g. [1, 2, ..., 20] or [10]
    POWER_DENSITY : ndarray (H, nX, nY)     W/mm² or W/mrad²
    ENERGY        : ndarray (H, nX, nY)     eV  (NaN for summed block)
    FLUX          : ndarray (H, nX, nY)     ph/s/0.1%BW
    X_grid        : ndarray (nX, nY)        mm
    Y_grid        : ndarray (nX, nY)        mm
    """

    fortran_float = re.compile(r'([+-]?\d+\.\d+)[Dd]([+-]?\d+)')

    def to_float(s):
        return float(fortran_float.sub(r'\1e\2', s))

    # ------------------------------------------------------------------ #
    # 1. Read the file and split into blocks
    # ------------------------------------------------------------------ #
    with open(filepath) as fh:
        text = fh.read()

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
    data_by_harmonic = {}

    for label, chunk in blocks:
        m_single = re.match(r'HARMONIC\s+(\d+)', label, re.I)
        m_sum    = re.match(r'HARMONICS\s+(\d+)\s+TO\s+(\d+)', label, re.I)

        if m_single:
            harm = int(m_single.group(1))
            has_energy = True
        elif m_sum:
            harm = 0
            has_energy = False
        else:
            continue

        rows_x, rows_y, rows_e, rows_pd, rows_fl = [], [], [], [], []

        for line in chunk.splitlines():
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                vals = [to_float(p) for p in parts]
            except ValueError:
                continue

            if has_energy and len(vals) >= 5:
                # X  Y  E  POWER_DENSITY  FLUX
                rows_x.append(vals[0]);  rows_y.append(vals[1])
                rows_e.append(vals[2]);  rows_pd.append(vals[3])
                rows_fl.append(vals[4])
            elif not has_energy and len(vals) >= 4:
                # X  Y  POWER_DENSITY  FLUX
                rows_x.append(vals[0]);  rows_y.append(vals[1])
                rows_e.append(np.nan);   rows_pd.append(vals[2])
                rows_fl.append(vals[3])

        if rows_x:
            data_by_harmonic[harm] = {
                'x':  np.array(rows_x),
                'y':  np.array(rows_y),
                'e':  np.array(rows_e),
                'pd': np.array(rows_pd),
                'fl': np.array(rows_fl),
            }

    # ------------------------------------------------------------------ #
    # 3. Build the 2-D grid from the first available harmonic
    # ------------------------------------------------------------------ #
    first_harm = sorted(k for k in data_by_harmonic if k > 0)[0]
    ref = data_by_harmonic[first_harm]

    x_unique = np.unique(ref['x'])
    y_unique = np.unique(ref['y'])
    nX, nY   = len(x_unique), len(y_unique)

    X_grid, Y_grid = np.meshgrid(x_unique, y_unique, indexing='ij')

    # ------------------------------------------------------------------ #
    # 4. Assemble 3-D arrays  (harmonic, X, Y)
    # ------------------------------------------------------------------ #
    harmonics = sorted(k for k in data_by_harmonic if k > 0)

    POWER_DENSITY = np.full((len(harmonics), nX, nY), np.nan)
    ENERGY        = np.full((len(harmonics), nX, nY), np.nan)
    FLUX          = np.full((len(harmonics), nX, nY), np.nan)

    x_idx = {v: i for i, v in enumerate(x_unique)}
    y_idx = {v: i for i, v in enumerate(y_unique)}

    for h_i, h in enumerate(harmonics):
        d = data_by_harmonic[h]
        for x, y, e, pd, fl in zip(d['x'], d['y'], d['e'], d['pd'], d['fl']):
            xi = x_idx[x]; yi = y_idx[y]
            POWER_DENSITY[h_i, xi, yi] = pd
            ENERGY       [h_i, xi, yi] = e
            FLUX         [h_i, xi, yi] = fl

    for name, arr in [('POWER_DENSITY', POWER_DENSITY),
                      ('ENERGY',        ENERGY),
                      ('FLUX',          FLUX)]:
        if not np.isfinite(arr).all():
            print(f"parse_urgent_2d_from_harmonics: Cleaned infinities in {name}")
            arr[~np.isfinite(arr)] = 0.0

    # ------------------------------------------------------------------ #
    # 5. Optionally expand to four quadrants
    # ------------------------------------------------------------------ #
    if flag_full_grid:
        POWER_DENSITY = _expand_quadrant(POWER_DENSITY)
        ENERGY        = _expand_quadrant(ENERGY)
        FLUX          = _expand_quadrant(FLUX)
        X_grid, Y_grid = _full_grid(x_unique, y_unique)

    return harmonics, POWER_DENSITY, ENERGY, FLUX, X_grid, Y_grid


# ------------------------------------------------------------------ #
# Four-quadrant helpers
# ------------------------------------------------------------------ #

def _expand_quadrant(arr: np.ndarray) -> np.ndarray:
    """Mirror (..., nX, nY) → (..., 2*nX-1, 2*nY-1) by even symmetry."""
    full_x = np.concatenate([np.flip(arr[..., 1:, :], axis=-2), arr], axis=-2)
    return     np.concatenate([np.flip(full_x[..., 1:], axis=-1), full_x], axis=-1)


def _full_grid(x_pos: np.ndarray, y_pos: np.ndarray):
    """Build full meshgrids from one-quadrant 1-D coordinate arrays."""
    x_full = np.concatenate([-x_pos[1:][::-1], x_pos])
    y_full = np.concatenate([-y_pos[1:][::-1], y_pos])
    return np.meshgrid(x_full, y_full, indexing='ij')


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
if __name__ == '__main__':
    filepath = 'urgent.out'
    harmonics, POWER_DENSITY, ENERGY, FLUX, X_grid, Y_grid = \
        parse_urgent_2d_from_harmonics(filepath)

    print(f"harmonics     : {harmonics}")
    print(f"POWER_DENSITY : shape={POWER_DENSITY.shape}, dtype={POWER_DENSITY.dtype}")
    print(f"ENERGY        : shape={ENERGY.shape}")
    print(f"FLUX          : shape={FLUX.shape}")
    print(f"X_grid        : shape={X_grid.shape}  range {X_grid[0,0]} -> {X_grid[-1,0]}")
    print(f"Y_grid        : shape={Y_grid.shape}  range {Y_grid[0,0]} -> {Y_grid[0,-1]}")
    print(f"\nAt centre (x=0, y=0):")
    cx, cy = X_grid.shape[0]//2, X_grid.shape[1]//2
    print(f"  POWER_DENSITY[0] = {POWER_DENSITY[0, cx, cy]:.4f}")
    print(f"  ENERGY[0]        = {ENERGY[0, cx, cy]:.3f} eV")
    print(f"  FLUX[0]          = {FLUX[0, cx, cy]:.4e}")

    # print(X_grid.shape, Y_grid.shape, POWER_DENSITY.shape, ENERGY.shape)
    from srxraylib.plot.gol import plot_image

    plot_image(POWER_DENSITY.sum(axis=0), X_grid[:,0], Y_grid[0,:])
    # print(X_grid[:,0], Y_grid[0,:])