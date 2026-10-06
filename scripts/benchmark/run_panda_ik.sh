#!/bin/bash
# Run the clutter grasping protocol with the Panda + parallel-gripper baseline:
# for every object USD, isolated + C0_easy + C1_medium + C2_hard, NUM_TRIALS trials each.
# Needs no trained policy (scripted IK, GG-CNN grasp pose), so it is the quickest end-to-end check.
#
# Usage (inside the Isaac Lab container, from anywhere):
#   bash scripts/benchmark/run_panda_ik.sh
#   bash scripts/benchmark/run_panda_ik.sh --objects "A24_0" --conditions "isolated C1_medium" --num_trials 5 --yes

# ── Repository paths ───────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RESULTS_DIR="${REPO_ROOT}/results"
OBJECTS_DIR="${REPO_ROOT}/source/clutter_grasp/clutter_grasp/assets/objects"
PYTHON="${PYTHON:-/isaac-sim/python.sh}"

# ── Defaults (paper protocol: 100 trials per object and condition) ────────────
NUM_TRIALS=100
OBJECTS=""
CONDITIONS=(isolated C0_easy C1_medium C2_hard)
ASSUME_YES=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num_trials) NUM_TRIALS="$2"; shift 2 ;;
        --objects)    OBJECTS="$2"; shift 2 ;;
        --conditions) read -r -a CONDITIONS <<< "$2"; shift 2 ;;
        --yes|-y)     ASSUME_YES=true; shift ;;
        -h|--help)    sed -n '2,8p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $1 (see --help)"; exit 1 ;;
    esac
done

[ -z "${OBJECTS}" ] && OBJECTS=$(ls -1 "${OBJECTS_DIR}" | sed 's/\.usd$//')
OBJECTS=(${OBJECTS})
LOG_DIR="${RESULTS_DIR}/benchmark/panda_ik/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "========================================"
echo "Controller:  panda_ik"
echo "Objects:     ${#OBJECTS[@]} (${OBJECTS[*]})"
echo "Conditions:  ${CONDITIONS[*]}"
echo "Trials:      ${NUM_TRIALS} per object and condition"
echo "Results:     ${RESULTS_DIR}/benchmark/panda_ik/"
echo "========================================"
if [ "${ASSUME_YES}" != true ]; then
    read -p "Proceed? (y/n) " -n 1 -r; echo
    [[ $REPLY =~ ^[Yy]$ ]] || { echo "Aborted."; exit 1; }
fi
mkdir -p "${LOG_DIR}"

TOTAL=$(( ${#OBJECTS[@]} * ${#CONDITIONS[@]} )); N=0; OK=0; FAILED=()
START=$(date +%s)

for object_id in "${OBJECTS[@]}"; do
    for condition in "${CONDITIONS[@]}"; do
        N=$((N + 1))
        if [ "${condition}" = "isolated" ]; then
            COND_ARGS=(--isolated)
        else
            COND_ARGS=(--target_complexity "${condition}")
        fi
        LOG_FILE="${LOG_DIR}/${object_id}_${condition}_${TIMESTAMP}.log"
        echo ""
        echo "── [${N}/${TOTAL}] ${object_id} / ${condition} ──  log: ${LOG_FILE}"

        LIVESTREAM="${LIVESTREAM:-2}" "${PYTHON}" "${SCRIPT_DIR}/run_panda_ik.py" \
            --target_object "${object_id}" --num_trials "${NUM_TRIALS}" \
            --object_usd_dir "${OBJECTS_DIR}" \
            "${COND_ARGS[@]}" --enable_cameras --headless \
            2>&1 | tee "${LOG_FILE}"

        if [ "${PIPESTATUS[0]}" -eq 0 ]; then
            OK=$((OK + 1))
        else
            FAILED+=("${object_id}/${condition}")
            echo "✗ ${object_id}/${condition} failed (see ${LOG_FILE})"
            if [ "${ASSUME_YES}" != true ]; then
                read -p "Continue with next run? (y/n) " -n 1 -r; echo
                [[ $REPLY =~ ^[Yy]$ ]] || exit 1
            fi
        fi
    done
done

ELAPSED=$(( $(date +%s) - START ))
echo ""
echo "========================================"
echo "Done: ${OK}/${TOTAL} runs succeeded in $((ELAPSED / 3600))h $(((ELAPSED % 3600) / 60))m"
[ ${#FAILED[@]} -gt 0 ] && echo "Failed: ${FAILED[*]}"
echo "Results: ${RESULTS_DIR}/benchmark/panda_ik/"
echo "========================================"
