#!/usr/bin/env bash
set -euo pipefail

# Submit after adapting the resource expression to your cluster, for example:
#   oarsub -l 'host=1/gpu=1,walltime=04:00:00' -S ./tpch/run_why_oar.sh

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${REPO_DIR}/tpch/oar_logs"
JOB_LABEL="${OAR_JOB_ID:-manual}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
mkdir -p "${LOG_DIR}"

if [[ -n "${VENV_PATH:-}" ]]; then
    # Set VENV_PATH when the required packages are installed in a virtualenv.
    source "${VENV_PATH}/bin/activate"
    PYTHON_BIN="python3"
fi

cd "${REPO_DIR}"

ollama serve >"${LOG_DIR}/ollama_${JOB_LABEL}.log" 2>&1 &
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
    "${PYTHON_BIN}" "${REPO_DIR}/tpch/Why.py" \
    >"${LOG_DIR}/why_${JOB_LABEL}.log" 2>&1

echo "Pipeline complete."
echo "Results: ${REPO_DIR}/tpch/test_pipeline.json"
echo "Timings: ${REPO_DIR}/tpch/timing_metrics.json"
echo "Run log: ${LOG_DIR}/why_${JOB_LABEL}.log"
