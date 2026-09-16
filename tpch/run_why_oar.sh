#!/usr/bin/env bash
set -euo pipefail

# Submit after adapting the resource expression to your cluster, for example:
#   oarsub -l 'host=1/gpu=1,walltime=04:00:00' -S ./tpch/run_why_oar.sh
OLLAMA_BIN="${OLLAMA_BIN:-/home/daisy/cicciara/ollama/bin/ollama}"
VENV_PATH="${VENV_PATH-/home/daisy/cicciara/venvs/rag-why}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${LOG_DIR:-${REPO_DIR}/tpch/oar_logs}"
JOB_LABEL="${OAR_JOB_ID:-manual}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
mkdir -p "${LOG_DIR}"

if [[ -n "${VENV_PATH:-}" ]]; then
    # Set VENV_PATH when the required packages are installed in a virtualenv.
    source "${VENV_PATH}/bin/activate"
    PYTHON_BIN="${VENV_PATH}/bin/python"
fi

cd "${REPO_DIR}"
export TPCH_DATA_DIR="${TPCH_DATA_DIR:-${REPO_DIR}/tpch/tpch-data}"
export QUESTIONS_FILE="${QUESTIONS_FILE:-${REPO_DIR}/tpch/questions.json}"
export FAISS_INDEX_DIR="${FAISS_INDEX_DIR:-${REPO_DIR}/tpch/faiss_index}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/tpch/runs/tpch_${JOB_LABEL}}"
# Fail before starting Ollama if any input rows or questions are malformed.
"${PYTHON_BIN}" "${REPO_DIR}/tpch/tpch_data.py" \
    --data-dir "${TPCH_DATA_DIR}" --questions "${QUESTIONS_FILE}"

# Large models can take several minutes to allocate and load on a cold start.
export OLLAMA_LOAD_TIMEOUT="${OLLAMA_LOAD_TIMEOUT:-10m}"
"${OLLAMA_BIN}" serve >"${LOG_DIR}/ollama_${JOB_LABEL}.log" 2>&1 &
OLLAMA_PID=$!
cleanup() {
    kill "${OLLAMA_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Wait until Ollama accepts requests, and fail clearly if it exits during startup.
for _ in $(seq 1 60); do
    if curl --silent --fail http://127.0.0.1:11434/api/tags >/dev/null; then
        break
    fi
    if ! kill -0 "${OLLAMA_PID}" 2>/dev/null; then
        echo "Ollama failed to start; see ${LOG_DIR}/ollama_${JOB_LABEL}.log" >&2
        exit 1
    fi
    sleep 1
done

if ! curl --silent --fail http://127.0.0.1:11434/api/tags >/dev/null; then
    echo "Timed out waiting for Ollama" >&2
    exit 1
fi

# A rebuild is required to measure CSV embedding. Set REBUILD_FAISS_INDEX=0 to
# reuse an existing index; embedding duration will then intentionally be null.
REBUILD_FAISS_INDEX="${REBUILD_FAISS_INDEX:-1}" \
LLM_MODEL_NAME="${LLM_MODEL_NAME:-llama3:70b}" \
REQUIRE_CUDA="${REQUIRE_CUDA:-1}" \
    "${PYTHON_BIN}" "${REPO_DIR}/tpch/Why.py" \
    >"${LOG_DIR}/why_${JOB_LABEL}.log" 2>&1

"${PYTHON_BIN}" "${REPO_DIR}/tpch/analyze_results.py" \
    --run-dir "${OUTPUT_DIR}" \
    --ground-truth "${GROUND_TRUTH_FILE:-${REPO_DIR}/tpch/ground_truthTpch.json}" \
    >"${LOG_DIR}/analysis_${JOB_LABEL}.log" 2>&1

echo "Pipeline and analysis complete."
echo "Results: ${OUTPUT_DIR}/test_pipeline.json"
echo "Timings: ${OUTPUT_DIR}/timing_metrics.json"
echo "Run log: ${LOG_DIR}/why_${JOB_LABEL}.log"
echo "Analysis: ${OUTPUT_DIR}/analysis/summary.md"
echo "Dashboard: ${OUTPUT_DIR}/analysis/dashboard.svg"
echo "Analysis log: ${LOG_DIR}/analysis_${JOB_LABEL}.log"
