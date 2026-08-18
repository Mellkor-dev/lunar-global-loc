#!/usr/bin/env bash
# Parallel, resumable Apollo 14 localization pipeline.
#
# Optional environment controls:
#   DARCES_WORKERS=6          independent site processes
#   RESOLUTIONS="5m"          run only selected profiles
#   START_PHASE=4             resume at DARCES (phases 1--8 below)
#   FORCE_DARCES=1            ignore an already complete aggregate result
#   LOG_DIR=/some/directory   override the timestamped log directory

set -Eeuo pipefail

if (( $# > 0 )); then
    case "$1" in
        -h|--help)
            echo "Usage: ./run_apollo14_parallel_pipeline.sh"
            echo "Environment: DARCES_WORKERS=6 RESOLUTIONS=\"...\" START_PHASE=1..8 FORCE_DARCES=0|1 LOG_DIR=..."
            exit 0
            ;;
        *)
            echo "Unexpected argument: $1 (use --help)" >&2
            exit 2
            ;;
    esac
fi

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$WORKSPACE/LargeScale_Implement"
PYTHON="$WORKSPACE/.venv/bin/python"
START_PHASE="${START_PHASE:-1}"
DARCES_WORKERS="${DARCES_WORKERS:-6}"
FORCE_DARCES="${FORCE_DARCES:-0}"
LOG_DIR="${LOG_DIR:-$WORKSPACE/pipeline_logs/parallel_$(date +%Y%m%d_%H%M%S)}"

if [[ ! -x "$PYTHON" ]]; then
    echo "Python environment not found: $PYTHON" >&2
    exit 1
fi
if [[ ! "$START_PHASE" =~ ^[1-8]$ ]]; then
    echo "START_PHASE must be an integer from 1 through 8" >&2
    exit 1
fi
if [[ ! "$DARCES_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
    echo "DARCES_WORKERS must be a positive integer" >&2
    exit 1
fi
if [[ "$FORCE_DARCES" != "0" && "$FORCE_DARCES" != "1" ]]; then
    echo "FORCE_DARCES must be 0 or 1" >&2
    exit 1
fi

# Prevent each process from creating its own nested BLAS/OpenMP thread pool.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

read -r -a RESOLUTION_LIST <<< "${RESOLUTIONS:-0p25m 0p5m 1m 2m 5m 10m}"
VALID_RESOLUTIONS=" 0p25m 0p5m 1m 2m 5m 10m "
for resolution in "${RESOLUTION_LIST[@]}"; do
    if [[ "$VALID_RESOLUTIONS" != *" $resolution "* ]]; then
        echo "Unsupported resolution: $resolution" >&2
        exit 1
    fi
done
LOCALIZATION_RESOLUTIONS=("${RESOLUTION_LIST[@]}")

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/apollo14_parallel_pipeline.log") 2>&1
trap 'status=$?; echo "FAILED (exit $status) at line $LINENO. Log: $LOG_DIR/apollo14_parallel_pipeline.log"; exit $status' ERR

cd "$WORKSPACE"

run() {
    echo
    printf '>>> '
    printf '%q ' "$@"
    echo
    "$@"
}

phase_enabled() {
    (( START_PHASE <= $1 ))
}

darces_result_is_complete() {
    local resolution="$1"
    "$PYTHON" -c "
import json
from pathlib import Path
import sys
sys.path.insert(0, r'$PROJECT')
from pipeline_config import load_resolution_config
c = load_resolution_config('$resolution')
p = c.results_path / 'darces_all_sites.json'
if not p.is_file():
    raise SystemExit(1)
try:
    payload = json.loads(p.read_text(encoding='utf-8'))
    settings = payload['settings']
    expected = sorted(
        int(path.stem.rsplit('_', 1)[1])
        for path in c.gridded_maps_path.glob('grid_site_*.npz')
    )
    actual = sorted(int(record['site']) for record in payload['sites'])
    compatible = (
        actual == expected
        and int(settings['trials_per_site']) == 100000
        and int(settings['seed']) == 29
        and float(settings['heading_tolerance_deg']) == 5.0
        and int(settings['minimum_cluster_size']) == 2
        and float(settings['cluster_position_radius_m']) == 50.0
        and int(settings['top_hypothesis_count']) == 5
        and float(settings['control_rms_tolerance_m']) == 10.0
        and float(settings['side_ratio_tolerance']) == 0.12
        and float(settings['minimum_triangle_angle_deg']) == 10.0
        and float(settings['covariance_sigma_multiplier']) == 2.0
        and float(settings['distance_tolerance_sigma_multiplier']) == 3.0
        and float(settings['z_residual_tolerance_sigma_multiplier']) == 3.0
        and float(settings['minimum_overlap']) == 0.5
        and settings['maximum_terrain_mae_m'] is None
        and float(settings['cluster_heading_radius_deg']) == 5.0
        and float(settings['fitness_reference_spacing_factor']) == 0.5
        and float(settings['consensus_xy_tolerance_m']) == 15.0
        and int(settings['minimum_consensus_features']) == 4
        and not bool(settings['use_feature_consensus'])
    )
except (KeyError, TypeError, ValueError, json.JSONDecodeError):
    compatible = False
raise SystemExit(0 if compatible else 1)
" >/dev/null
}

echo "Apollo 14 parallel localization pipeline"
echo "Workspace:       $WORKSPACE"
echo "Resolutions:     ${RESOLUTION_LIST[*]}"
echo "Localization:    ${LOCALIZATION_RESOLUTIONS[*]:-(none)}"
echo "DARCES workers:  $DARCES_WORKERS"
echo "Start phase:     $START_PHASE"
echo "Force DARCES:    $FORCE_DARCES"
echo "Log:             $LOG_DIR/apollo14_parallel_pipeline.log"

run "$PYTHON" -c \
    "import sys; sys.path.insert(0, '$PROJECT'); from pipeline_config import load_pipeline_config; c=load_pipeline_config(); assert c.config_path.name == 'apollo14.yaml'; assert c.truth_source_profile == '2m'; print('Config:', c.config_path); print('Truth:', c.truth_source_label)"

# Create once (or validate and reuse) the experiment-wide random site sample.
# Every resolution and every local/downstream stage consumes this manifest.
run "$PYTHON" "$PROJECT/Scripts/20_select_sites.py" --resolution 0p25m

# Phase 1: DEM, catalogue, uncertainty, and capture validation.
if phase_enabled 1; then
    for resolution in "${RESOLUTION_LIST[@]}"; do
        run "$PYTHON" "$PROJECT/Scripts/00_inspect_DEM.py" --resolution "$resolution"
        run "$PYTHON" "$PROJECT/Scripts/02_validate_dem_allignment.py" --resolution "$resolution"
    done
    run "$PYTHON" "$PROJECT/Scripts/03_feature_detector.py" --resolution all
    for resolution in "${LOCALIZATION_RESOLUTIONS[@]}"; do
        run "$PYTHON" "$PROJECT/Scripts/04_global_feature_uncertainty.py" --resolution "$resolution"
    done
    run "$PYTHON" "$PROJECT/Scripts/06_validate_capture.py" --resolution 0p25m
fi

# Phase 2: level and grid each synchronized LiDAR capture.
if phase_enabled 2; then
    for resolution in "${RESOLUTION_LIST[@]}"; do
        run "$PYTHON" "$PROJECT/Scripts/05_local_scan_processing.py" --resolution "$resolution"
    done
fi

# Phase 3: local crater extraction.
if phase_enabled 3; then
    for resolution in "${RESOLUTION_LIST[@]}"; do
        run "$PYTHON" "$PROJECT/Scripts/07_local_feature_detection.py" --resolution "$resolution"
    done
fi

# Phase 4: deterministic site-parallel DARCES with automatic checkpoints.
if phase_enabled 4; then
    for resolution in "${LOCALIZATION_RESOLUTIONS[@]}"; do
        if [[ "$FORCE_DARCES" == "0" ]] && darces_result_is_complete "$resolution"; then
            echo
            echo ">>> $resolution DARCES already complete and compatible; skipping"
            continue
        fi
        run "$PYTHON" "$PROJECT/Scripts/09_darces_all_sites.py" \
            --resolution "$resolution" \
            --trials 100000 \
            --seed 29 \
            --heading-tolerance 5 \
            --minimum-cluster-size 2 \
            --cluster-position-radius 50 \
            --top-hypotheses 5 \
            --control-rms-tolerance 10 \
            --side-ratio-tolerance 0.12 \
            --minimum-triangle-angle 10 \
            --covariance-sigma-multiplier 2 \
            --distance-tolerance-sigma 3 \
            --z-residual-tolerance-sigma 3 \
            --minimum-overlap 0.5 \
            --cluster-heading-radius 5 \
            --reference-spacing-factor 0.5 \
            --consensus-radius 15 \
            --minimum-consensus-features 4 \
            --workers "$DARCES_WORKERS"
    done
fi

# Phase 5: traversal overlays.
if phase_enabled 5; then
    for resolution in "${LOCALIZATION_RESOLUTIONS[@]}"; do
        run "$PYTHON" "$PROJECT/Scripts/10_plot_traverse_map.py" --resolution "$resolution"
    done
fi

# Phase 6: correspondence-level RANSAC.
if phase_enabled 6; then
    for resolution in "${LOCALIZATION_RESOLUTIONS[@]}"; do
        run "$PYTHON" "$PROJECT/Scripts/11_ransac_all_sites.py" --resolution "$resolution"
    done
fi

# Phase 7: independent single-frame 2.5-D MOGA.
if phase_enabled 7; then
    for resolution in "${LOCALIZATION_RESOLUTIONS[@]}"; do
        run "$PYTHON" "$PROJECT/Scripts/12_moga_all_sites.py" \
            --resolution "$resolution" \
            --heading-sigma-deg 1 \
            --max-evaluations 300 \
            --relative-tolerance 1e-10
    done
fi

# Phase 8: combined diagnostics and presentation summary.
if phase_enabled 8; then
    run "$PYTHON" "$PROJECT/Scripts/13_pipeline_diagnostics.py" --resolution all
    run "$PYTHON" "$PROJECT/Scripts/14_results_presentation.py"
fi

echo
echo "Apollo 14 parallel pipeline completed."
echo "Results: $PROJECT/results"
echo "Plots:   $PROJECT/plots"
echo "Log:     $LOG_DIR/apollo14_parallel_pipeline.log"
echo "DARCES checkpoints remain under results/<resolution>_px/darces_checkpoints/."
