#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-run_01}"
RESULT_ROOT="/mnt/gds/results"
RUNNER="experiments/frequency_momentum_ablation/run.py"
ANALYZER="experiments/frequency_momentum_ablation/analyze_phase1.py"
SPLITS="experiments/frequency_momentum_ablation/splits.json"

COMMON=(
  --requests 30000
  --frequency-bytes 8192
  --momentum-bytes 1024
  --frequency-window 32768
  --frequency-thresholds 3 4
  --momentum-thresholds 1 2 3 4
  --score-modes max weighted multiplicative lexicographic
  --hit-ns 11915
  --bypass-ns 65977
  --repeats 5
  --splits-file "${SPLITS}"
)

COST_DIR="${RESULT_ROOT}/groute_phase1_cost_sensitivity/${RUN_ID}"
MOMENTUM_DIR="${RESULT_ROOT}/groute_phase1_momentum_threshold1/${RUN_ID}"
SCORE_DIR="${RESULT_ROOT}/groute_phase1_score_ablation/${RUN_ID}"
SHADOW_DIR="${RESULT_ROOT}/groute_phase1_shadow_pollution/${RUN_ID}"
HOLDOUT_DIR="${RESULT_ROOT}/groute_phase1_holdout_evaluation/${RUN_ID}"

for directory in "${COST_DIR}" "${MOMENTUM_DIR}" "${SCORE_DIR}" \
                 "${SHADOW_DIR}" "${HOLDOUT_DIR}"; do
  if [[ -e "${directory}" ]]; then
    echo "ERROR: ${directory} already exists; choose another run_XX" >&2
    exit 1
  fi
done

# Tuning is restricted to stable_zipf and phase_shift_aba with tuning seeds.
python "${RUNNER}" "${COMMON[@]}" \
  --experiment cost_sensitivity_tuning \
  --split tuning \
  --cache-lines 2 4 8 16 \
  --momentum-windows 128 256 512 1024 2048 \
  --fill-ns-values 52784 58000 62000 72000 90000 \
  --output-dir "${COST_DIR}/tuning"

python "${ANALYZER}" select \
  --inputs "${COST_DIR}/tuning" \
  --split tuning \
  --output "${HOLDOUT_DIR}/selection_retuned.json"

python "${ANALYZER}" select \
  --inputs "${COST_DIR}/tuning" \
  --split tuning \
  --fixed-from-fill 52784 \
  --output "${HOLDOUT_DIR}/selection_fixed_52784.json"

# Paired M=1/2/3/4 comparison on held-out scan and two-reference-burst traces.
python "${RUNNER}" "${COMMON[@]}" \
  --experiment momentum_threshold1_paired \
  --split final_heldout_trace \
  --cache-lines 2 4 \
  --momentum-windows 128 512 \
  --fill-ns-values 72000 \
  --policies hybrid dual \
  --output-dir "${MOMENTUM_DIR}/paired_holdout"

# Same-threshold score pairing. No winner is selected from these held-out rows.
python "${RUNNER}" "${COMMON[@]}" \
  --experiment score_ablation_paired \
  --split final_heldout_trace \
  --cache-lines 2 4 \
  --momentum-windows 128 512 \
  --fill-ns-values 72000 90000 \
  --policies dual \
  --output-dir "${SCORE_DIR}/paired_holdout"

# Dedicated output for auditing ghost reaccess versus the shadow counterfactual.
python "${RUNNER}" "${COMMON[@]}" \
  --experiment shadow_pollution \
  --split final_heldout_trace \
  --cache-lines 2 4 \
  --momentum-windows 128 \
  --fill-ns-values 72000 \
  --policies cache_all hybrid dual \
  --output-dir "${SHADOW_DIR}/heldout"

# Locked retuned envelope, evaluated on unseen seeds and unseen trace classes.
for split in validation final_in_domain final_heldout_trace; do
  python "${RUNNER}" "${COMMON[@]}" \
    --experiment holdout_retuned \
    --split "${split}" \
    --cache-lines 2 4 8 16 \
    --fill-ns-values 52784 58000 62000 72000 90000 \
    --selection-file "${HOLDOUT_DIR}/selection_retuned.json" \
    --output-dir "${HOLDOUT_DIR}/retuned/${split}"
done

# Fixed policy selected at fill=52784, then evaluated without retuning.
for split in final_in_domain final_heldout_trace; do
  python "${RUNNER}" "${COMMON[@]}" \
    --experiment holdout_fixed_policy \
    --split "${split}" \
    --cache-lines 2 4 8 16 \
    --fill-ns-values 52784 58000 62000 72000 90000 \
    --selection-file "${HOLDOUT_DIR}/selection_fixed_52784.json" \
    --output-dir "${HOLDOUT_DIR}/fixed_policy/${split}"
done

python "${ANALYZER}" report \
  --inputs "${COST_DIR}" "${MOMENTUM_DIR}" "${SCORE_DIR}" \
           "${SHADOW_DIR}" "${HOLDOUT_DIR}" \
  --output-dir "${HOLDOUT_DIR}/report"

echo "Phase-one results saved under ${RESULT_ROOT}/groute_phase1_*/*/${RUN_ID}"
