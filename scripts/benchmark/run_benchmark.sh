#!/bin/bash
# Run the full clutter grasping protocol with the Contactile hand:
# for every object, isolated + C0_easy + C1_medium + C2_hard, NUM_TRIALS trials each.
#
# Usage (inside the Isaac Lab container, from anywhere):
#   bash scripts/benchmark/run_benchmark.sh                               # per-object PPO (rl), all objects
#   bash scripts/benchmark/run_benchmark.sh --policy distilled            # rl | rl_clutter | transformer | distilled
#   bash scripts/benchmark/run_benchmark.sh --objects "A24_0 D25_3" --num_trials 10
#   bash scripts/benchmark/run_benchmark.sh --controller my_pkg.ctrl:make_controller --controller_name mine
#   bash scripts/benchmark/run_benchmark.sh --yes                         # no confirmation prompts

# ── Repository paths ───────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WEIGHTS_DIR="${REPO_ROOT}/weights"
RESULTS_DIR="${REPO_ROOT}/results"
OBJECTS_DIR="${REPO_ROOT}/source/clutter_grasp/clutter_grasp/assets/objects"
PYTHON="${PYTHON:-/isaac-sim/python.sh}"

# ── Defaults (paper protocol: 100 trials per object and condition) ────────────
NUM_TRIALS=100
OBJECTS=""
CONTROLLER=""
CONTROLLER_NAME=""
POLICY="rl"
TRAINED_POLICIES_DIR=""
CLASSIFIER_MODEL_DIR="${WEIGHTS_DIR}/classifier"
CONDITIONS=(isolated C0_easy C1_medium C2_hard)
ASSUME_YES=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num_trials)            NUM_TRIALS="$2"; shift 2 ;;
        --objects)               OBJECTS="$2"; shift 2 ;;
        --conditions)            read -r -a CONDITIONS <<< "$2"; shift 2 ;;
        --policy)                POLICY="$2"; shift 2 ;;
        --controller)            CONTROLLER="$2"; shift 2 ;;
        --controller_name)       CONTROLLER_NAME="$2"; shift 2 ;;
        --trained_policies_dir)  TRAINED_POLICIES_DIR="$2"; shift 2 ;;
        --classifier_model_dir)  CLASSIFIER_MODEL_DIR="$2"; shift 2 ;;
        --yes|-y)                ASSUME_YES=true; shift ;;
        -h|--help)
            sed -n '2,10p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $1 (see --help)"; exit 1 ;;
    esac
done

case "${POLICY}" in
    rl)          DEFAULT_POLICIES_DIR="${WEIGHTS_DIR}/ppo_policies" ;;
    rl_clutter)  DEFAULT_POLICIES_DIR="${WEIGHTS_DIR}/ppo_clutter_policies" ;;
    transformer|distilled) DEFAULT_POLICIES_DIR="" ;;
    *) echo "Unknown --policy ${POLICY} (rl, rl_clutter, transformer, distilled)"; exit 1 ;;
esac
TRAINED_POLICIES_DIR="${TRAINED_POLICIES_DIR:-${DEFAULT_POLICIES_DIR}}"

# ── Object list: PPO policies available, or every object USD for a transformer / external controller ──
if [ -z "${OBJECTS}" ]; then
    if [ -n "${CONTROLLER}" ] || [ -z "${TRAINED_POLICIES_DIR}" ]; then
        OBJECTS=$(ls -1 "${OBJECTS_DIR}" | sed 's/\.usd$//')
    else
        if [ ! -d "${TRAINED_POLICIES_DIR}" ]; then
            echo "ERROR: ${TRAINED_POLICIES_DIR} not found. Run scripts/download_weights.sh first."
            exit 1
        fi
        OBJECTS=$(ls -1 "${TRAINED_POLICIES_DIR}")
    fi
fi
OBJECTS=(${OBJECTS})

LABEL="${CONTROLLER_NAME:-${CONTROLLER:+${CONTROLLER##*:}}}"
LABEL="${LABEL:-${POLICY}}"
LOG_DIR="${RESULTS_DIR}/benchmark/${LABEL}/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "========================================"
echo "Controller:  ${LABEL}"
echo "Objects:     ${#OBJECTS[@]} (${OBJECTS[*]})"
echo "Conditions:  ${CONDITIONS[*]}"
echo "Trials:      ${NUM_TRIALS} per object and condition"
echo "Results:     ${RESULTS_DIR}/benchmark/${LABEL}/"
echo "========================================"
if [ "${ASSUME_YES}" != true ]; then
    read -p "Proceed? (y/n) " -n 1 -r; echo
    [[ $REPLY =~ ^[Yy]$ ]] || { echo "Aborted."; exit 1; }
fi
mkdir -p "${LOG_DIR}"

EXTRA_ARGS=(--policy "${POLICY}" --classifier_model_dir "${CLASSIFIER_MODEL_DIR}")
[ -n "${TRAINED_POLICIES_DIR}" ] && EXTRA_ARGS+=(--trained_policies_dir "${TRAINED_POLICIES_DIR}")
[ -n "${CONTROLLER}" ] && EXTRA_ARGS+=(--controller "${CONTROLLER}")
[ -n "${CONTROLLER_NAME}" ] && EXTRA_ARGS+=(--controller_name "${CONTROLLER_NAME}")

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

        LIVESTREAM="${LIVESTREAM:-2}" "${PYTHON}" "${SCRIPT_DIR}/run_benchmark.py" \
            --target_object "${object_id}" --num_trials "${NUM_TRIALS}" \
            "${COND_ARGS[@]}" "${EXTRA_ARGS[@]}" --headless \
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
echo "Results: ${RESULTS_DIR}/benchmark/${LABEL}/"
echo "Next:    python analysis/analyze_sim.py --controller ${LABEL}=${RESULTS_DIR}/benchmark/${LABEL} ..."
echo "========================================"
