#!/usr/bin/env bash
# Run the Haworth localization experiment from DEM validation through summary.
#
# Optional environment controls:
#   RESOLUTIONS="5m"          run only selected profiles
#   START_PHASE=4             resume at DARCES (see phase numbers below)
#   LOG_DIR=/some/directory   override the timestamped log directory

set -Eeuo pipefail

if (( $# > 0 )); then
    case "$1" in
        -h|--help)
            echo "Usage: ./run_haworth_pipeline.sh"
            echo "Environment: RESOLUTIONS=\"...\" START_PHASE=1..8 LOG_DIR=..."
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
LOG_DIR="${LOG_DIR:-$WORKSPACE/pipeline_logs/$(date +%Y%m%d_%H%M%S)}"

if [[ ! -x "$PYTHON" ]]; then
    echo "Python environment not found: $PYTHON" >&2
    exit 1
fi
if [[ ! "$START_PHASE" =~ ^[1-8]$ ]]; then
    echo "START_PHASE must be an integer from 1 through 8" >&2
    exit 1
fi

read -r -a RESOLUTION_LIST <<< "${RESOLUTIONS:-0p25m 0p5m 1m 2m 5m 10m}"
VALID_RESOLUTIONS=" 0p25m 0p5m 1m 2m 5m 10m "
LOCALIZATION_RESOLUTIONS=()
for resolution in "${RESOLUTION_LIST[@]}"; do
    if [[ "$VALID_RESOLUTIONS" != *" $resolution "* ]]; then
        echo "Unsupported resolution: $resolution" >&2
        exit 1
    fi
    if [[ "$resolution" != "10m" ]]; then
        LOCALIZATION_RESOLUTIONS+=("$resolution")
    fi
done

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/haworth_pipeline.log") 2>&1
trap 'status=$?; echo "FAILED (exit $status) at line $LINENO. Log: $LOG_DIR/haworth_pipeline.log"; exit $status' ERR

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

echo "Haworth localization pipeline"
echo "Workspace:    $WORKSPACE"
echo "Resolutions:  ${RESOLUTION_LIST[*]}"
echo "Localization: ${LOCALIZATION_RESOLUTIONS[*]:-(none)}"
echo "Start phase:  $START_PHASE"
echo "Log:          $LOG_DIR/haworth_pipeline.log"

# Abort early if this workspace is not using the Haworth configuration.
run "$PYTHON" -c \
    "import sys; sys.path.insert(0, '$PROJECT'); from pipeline_config import load_pipeline_config; c=load_pipeline_config(); assert c.config_path.name == 'haworth.yaml'; assert c.truth_source_profile == '1m'; print('Config:', c.config_path); print('Truth:', c.truth_source_label)"

# Validate and reuse the experiment-wide random site sample.
run "$PYTHON" "$PROJECT/Scripts/20_select_sites.py" --resolution 0p25m

# Phase 1: source DEM, catalogue, uncertainty, and capture validation.
if phase_enabled 1; then
    for resolution in "${RESOLUTION_LIST[@]}"; do
        run "$PYTHON" "$PROJECT/Scripts/00_inspect_DEM.py" \
            --resolution "$resolution"
        run "$PYTHON" "$PROJECT/Scripts/02_validate_dem_allignment.py" \
            --resolution "$resolution"
    done

    run "$PYTHON" "$PROJECT/Scripts/03_feature_detector.py" --resolution all

    for resolution in "${LOCALIZATION_RESOLUTIONS[@]}"; do
        run "$PYTHON" "$PROJECT/Scripts/04_global_feature_uncertainty.py" \
            --resolution "$resolution"
    done

    # All resolution workspaces point to this same synchronized capture set.
    run "$PYTHON" "$PROJECT/Scripts/06_validate_capture.py" \
        --resolution 0p25m
fi

# Phase 2: level and grid every LiDAR frame at each requested resolution.
if phase_enabled 2; then
    for resolution in "${RESOLUTION_LIST[@]}"; do
        run "$PYTHON" "$PROJECT/Scripts/05_local_scan_processing.py" \
            --resolution "$resolution"
    done
fi

# Phase 3: extract local crater descriptors.
if phase_enabled 3; then
    for resolution in "${RESOLUTION_LIST[@]}"; do
        run "$PYTHON" "$PROJECT/Scripts/07_local_feature_detection.py" \
            --resolution "$resolution"
    done
fi

# Phase 4: independent DARCES localization. These explicit values are the
# shared, finalized defaults in matching/darces.py. Feature consensus remains
# disabled unless a separate controlled experiment explicitly enables it.
if phase_enabled 4; then
    for resolution in "${LOCALIZATION_RESOLUTIONS[@]}"; do
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
            --minimum-consensus-features 4
    done
fi

# Phase 5: DARCES traversal overlays.
if phase_enabled 5; then
    for resolution in "${LOCALIZATION_RESOLUTIONS[@]}"; do
        run "$PYTHON" "$PROJECT/Scripts/10_plot_traverse_map.py" \
            --resolution "$resolution"
    done
fi

# Phase 6: correspondence-level RANSAC refinement.
if phase_enabled 6; then
    for resolution in "${LOCALIZATION_RESOLUTIONS[@]}"; do
        run "$PYTHON" "$PROJECT/Scripts/11_ransac_all_sites.py" \
            --resolution "$resolution"
    done
fi

# Phase 7: independent 2.5-D single-frame MOGA refinement.
if phase_enabled 7; then
    for resolution in "${LOCALIZATION_RESOLUTIONS[@]}"; do
        run "$PYTHON" "$PROJECT/Scripts/12_moga_all_sites.py" \
            --resolution "$resolution" \
            --heading-sigma-deg 1 \
            --max-evaluations 300 \
            --relative-tolerance 1e-10
    done
fi

# Phase 8: combined machine-readable diagnostics and presentation figure.
if phase_enabled 8; then
    run "$PYTHON" "$PROJECT/Scripts/13_pipeline_diagnostics.py" \
        --resolution all
    run "$PYTHON" "$PROJECT/Scripts/14_results_presentation.py"
fi

echo
echo "Haworth pipeline completed."
echo "Results: $PROJECT/results"
echo "Plots:   $PROJECT/plots"
echo "Log:     $LOG_DIR/haworth_pipeline.log"
echo "Note: 10 m preprocessing is included, but 10 m localization is skipped"
echo "because its global DEM contains fewer than three usable crater minima."
