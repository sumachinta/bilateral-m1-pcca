import scipy.io
import numpy as np
import re

TIME_STEP = 0.002 # 2 ms per frame (500 Hz sampling rate)

# ── Helper: parse "Col2~79..." into a numeric range ───────────────────────────
def _parse_col_range(s):
    """
    Extract 1-indexed start and end from strings like
    'Col2~79spike counts per frame for left hemisphere'.
    Used for both column ranges (NeuralData) and row ranges (Location).
    Returns (start, end) as 1-indexed integers.
    """
    match = re.search(r'Col(\d+)~(\d+)', s)
    if not match:
        raise ValueError(f"Could not parse range from description: '{s}'")
    return int(match.group(1)), int(match.group(2))


# ── Helper: normalize FSRS codes ──────────────────────────────────────────────
def fix_fsrs_mappinf(fsrs_array):
    """
    Normalize FSRS waveform classification codes.
    Confirmed coding:
        1  = RS  (regular spiking)
       -1  = FS  (fast spiking)
        0  = UN  (unclassified)
       48  = UN  (unclassified — alternate code, confirmed by colleague)
    Converts all 48s to 0 so the array has a single code for unclassified.
    """
    out = fsrs_array.copy()
    out[out == 48] = 0
    return out


# ── Session inspector: print a human-readable summary ─────────────────────────
def inspect_session(filepath):
    """
    Load a session .mat file and print a structured summary of its contents.
    Run this first on any new session file before analysis.
    Both NeuralData and Location are parsed dynamically — neuron counts and
    LH/RH boundaries differ across sessions.
    """
    mat = scipy.io.loadmat(filepath, simplify_cells=True)

    neural   = mat['NeuralData']
    expinfo  = mat['ExpInfo']
    location = mat['Location']
    desc     = mat['Description']

    neural_desc = desc['NeuralData']
    loc_desc    = desc['Location']

    # ── Parse LH/RH boundaries from NeuralData description ────────────────
    lh_start, lh_end = _parse_col_range(neural_desc['LH'])
    rh_start, rh_end = _parse_col_range(neural_desc['RH'])
    n_lh = lh_end - lh_start + 1
    n_rh = rh_end - rh_start + 1

    # ── Parse LH/RH row boundaries from Location description ──────────────
    loc_lh_start, loc_lh_end = _parse_col_range(loc_desc['NeuronNumber'])
    n_loc_lh = loc_lh_end - loc_lh_start + 1
    n_loc_rh = location.shape[0] - n_loc_lh

    print(f"\n{'='*60}")
    print(f"  Session file : {filepath}")
    print(f"{'='*60}")

    print(f"\n── NeuralData {neural.shape} ──")
    print(f"  Total frames   : {neural.shape[0]}")
    print(f"  LH neurons     : {n_lh}  (NeuralData cols {lh_start}–{lh_end})")
    print(f"  RH neurons     : {n_rh}  (NeuralData cols {rh_start}–{rh_end})")
    print(f"  Total neurons  : {n_lh + n_rh}")
    print(f"  Description.LH : '{neural_desc['LH']}'")
    print(f"  Description.RH : '{neural_desc['RH']}'")

    print(f"\n── Location {location.shape} ──")
    print(f"  LH rows (Location): rows {loc_lh_start}–{loc_lh_end}  ({n_loc_lh} neurons)")
    print(f"  RH rows (Location): rows {loc_lh_end+1}–{location.shape[0]}  ({n_loc_rh} neurons)")
    print(f"  Description.NeuronNumber: '{loc_desc['NeuronNumber']}'")
    fsrs_raw = location[:, 3]
    fsrs_norm = fix_fsrs_mappinf(fsrs_raw)
    fsrs_vals, counts = np.unique(fsrs_norm, return_counts=True)
    print(f"\n  FSRS value counts:")
    labels = {1: 'RS (regular spiking)', -1: 'FS (fast spiking)', 0: 'UN (unclassified)'}
    for v, c in zip(fsrs_vals, counts):
        print(f"    {v:5.0f}  →  {c} neurons  [{labels.get(v, f'UNKNOWN ({v})')}]")
    if (fsrs_raw == 48).any():
        print(f"    (note: {int((fsrs_raw == 48).sum())} neurons had raw FSRS=48, remapped to 0)")

    print(f"\n── ExpInfo {expinfo.shape} ──")
    print(f"  (columns fixed across all sessions)")
    exp_desc = desc['ExpInfo']
    for field, val in exp_desc.items():
        print(f"  {field:20s}: {val}")

    print(f"\n── Behavioral outcomes (frame-level counts) ──")
    print(f"  Hit frames              : {int(expinfo[:, 23].sum())}")
    print(f"  Miss frames             : {int(expinfo[:, 24].sum())}")
    print(f"  False Alarm frames      : {int(expinfo[:, 25].sum())}")
    print(f"  Correct Rejection frames: {int(expinfo[:, 26].sum())}")
    print(f"  Lick frames             : {int(expinfo[:, 27].sum())}")
    print(f"  Trial start frames      : {int((neural[:, 0] == 1).sum())}")

    print(f"\n{'='*60}\n")


# ── Session loader: returns a clean dict of named arrays ──────────────────────
def load_session(filepath):
    """
    Load a session .mat file into a structured dict.

    Both NeuralData and Location are parsed dynamically per session:
      - NeuralData: LH/RH column boundaries read from Description.NeuralData
      - Location:   LH/RH row boundaries read from Description.Location
    This is necessary because neuron counts differ across mice/sessions.

    ExpInfo column mappings are fixed across all sessions and hardcoded here.

    FSRS codes are normalized: 48 → 0 (both mean UN = unclassified).

    Returns
    -------
    session : dict with keys:
        filepath, n_lh, n_rh,
        frames, spikes_LH, spikes_RH,
        first_touch, piston_frames,
        touch_frames, whisker_angle, curvature, phase,
        run_speed,
        hits, misses, false_alarms, correct_rejs, licks,
        neuron_num, shank, depth, fsrs,
        loc_lh_mask, loc_rh_mask   ← boolean masks into Location rows
    """
    mat = scipy.io.loadmat(filepath, simplify_cells=True)

    neural      = mat['NeuralData']
    expinfo     = mat['ExpInfo']
    location    = mat['Location']
    desc        = mat['Description']
    neural_desc = desc['NeuralData']
    loc_desc    = desc['Location']

    # ── Dynamically parse LH/RH column boundaries in NeuralData ───────────
    lh_start, lh_end = _parse_col_range(neural_desc['LH'])
    rh_start, rh_end = _parse_col_range(neural_desc['RH'])
    lh_idx = slice(lh_start - 1, lh_end)   # 0-indexed Python slice
    rh_idx = slice(rh_start - 1, rh_end)
    n_lh   = lh_end - lh_start + 1
    n_rh   = rh_end - rh_start + 1

    # ── Dynamically parse LH/RH row boundaries in Location ────────────────
    loc_lh_start, loc_lh_end = _parse_col_range(loc_desc['NeuronNumber'])
    loc_lh_mask = np.zeros(location.shape[0], dtype=bool)
    loc_lh_mask[loc_lh_start - 1 : loc_lh_end] = True   # LH rows
    loc_rh_mask = ~loc_lh_mask                            # RH rows

    session = {
        # ── Metadata ──────────────────────────────────────────────────────
        'filepath'    : filepath,
        'n_lh'        : n_lh,
        'n_rh'        : n_rh,

        # ── Neural (session-variable columns) ─────────────────────────────
        'frames'      : neural[:, 0].astype(int),   # camera frame number
        'absolute_frames': np.arange(len(neural)),  # unique, monotonic frame index across entire session
        'time'        : np.arange(len(neural)) * TIME_STEP,  # seconds per frame
        'spikes_LH'   : neural[:, lh_idx],          # (n_frames, n_lh)
        'spikes_RH'   : neural[:, rh_idx],          # (n_frames, n_rh)

        # ── Trial / stimulus (fixed columns across sessions) ───────────────
        'first_touch'   : expinfo[:, 1],            # binary: first touch frame
        'piston_frames' : expinfo[:, 2:6],          # (F, 4): rightC, rightD, leftC, leftD
        'touch_frames'  : expinfo[:, 6:10],         # (F, 4): rightC, rightD, leftC, leftD
        'whisker_angle' : expinfo[:, 10:14],        # (F, 4): rightC, rightD, leftC, leftD
        'curvature'     : expinfo[:, 14:18],        # (F, 4): rightC, rightD, leftC, leftD
        'phase'         : expinfo[:, 18:22],        # (F, 4): rightC, rightD, leftC, leftD
        'run_speed'     : expinfo[:, 22],           # cm/s per frame

        # ── Behavioral outcomes (binary, per frame) ────────────────────────
        'hits'        : expinfo[:, 23],
        'misses'      : expinfo[:, 24],
        'false_alarms': expinfo[:, 25],
        'correct_rejs': expinfo[:, 26],
        'licks'       : expinfo[:, 27],

        # ── Neuron metadata (session-variable rows) ────────────────────────
        'neuron_num'  : location[:, 0].astype(int),
        'shank'       : location[:, 1],
        'depth'       : location[:, 2],             # µm from brain surface
        'fsrs'        : fix_fsrs_mappinf(location[:, 3]),  # 1=RS, -1=FS, 0=UN
        'loc_lh_mask' : loc_lh_mask,                # boolean: True = LH neuron
        'loc_rh_mask' : loc_rh_mask,                # boolean: True = RH neuron
    }

    return session


# ── Quick usage example ───────────────────────────────────────────────────────
if __name__ == '__main__':

    # Step 1: always inspect a new session file first
    inspect_session('P1.mat')

    # Step 2: load into a clean dict for analysis
    s = load_session('P1.mat')

    print(f"LH neurons    : {s['n_lh']}")
    print(f"RH neurons    : {s['n_rh']}")
    print(f"spikes_LH     : {s['spikes_LH'].shape}")    # (F, N_L)
    print(f"spikes_RH     : {s['spikes_RH'].shape}")    # (F, N_R)
    print(f"LH depth range: {s['depth'][s['loc_lh_mask']].min():.0f}–"
          f"{s['depth'][s['loc_lh_mask']].max():.0f} µm")
    print(f"RH depth range: {s['depth'][s['loc_rh_mask']].min():.0f}–"
          f"{s['depth'][s['loc_rh_mask']].max():.0f} µm")

    # Load all sessions — all mappings handled automatically per session
    # P*.mat = bilateral, U*.mat = unilateral (confirmed naming convention)
    import glob, os
    all_files = sorted(glob.glob('*.mat'))
    for f in all_files:
        group = 'bilateral' if os.path.basename(f).startswith('P') else 'unilateral'
        s = load_session(f)
        print(f"{f}  [{group}]  LH: {s['n_lh']} neurons, RH: {s['n_rh']} neurons")
