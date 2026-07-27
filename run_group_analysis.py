import matplotlib
matplotlib.use('Agg')  # non-interactive backend; suppresses plt.show()

from tools.session_group_analysis import run_group_analysis

# ── Configuration ─────────────────────────────────────────────────────────────
SESSION_IDS   = None            # None = auto-discover every results/*<WINDOW_SUFFIX>.joblib
WINDOW_SUFFIX = '_w0.0-1.0'
OUT_DIR       = 'group_analysis'
SAVE_FIGURES  = True            # per-session heatmaps / psv-scatter / latent-heatmap PNGs
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    neuron_tuning_all, latent_tuning_all, failed = run_group_analysis(
        session_ids=SESSION_IDS,
        window_suffix=WINDOW_SUFFIX,
        out_dir=OUT_DIR,
        save_figures=SAVE_FIGURES,
    )
