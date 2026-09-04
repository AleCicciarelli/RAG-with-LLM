# TPCH RAG results analysis

## Executive summary

- Answer exact match: **23/49 (46.9%)**.
- Exact provenance (`why`): **19/49 (38.8%)**.
- Full ground-truth evidence retrieved: **25/49 (51.0%)**; mean evidence recall: **73.8%**.
- With complete retrieval, answer accuracy was **92.0%**. With incomplete retrieval, it was **0.0%**.
- All 49 outputs parsed successfully; 14 contained an empty answer.

## Performance

- Cold question: **585.92 s**, accounting for **85.1%** of cumulative question latency. Model loading alone took 576.34 s.
- Warm requests: mean **2.13 s**, median **1.99 s**, p95 **3.43 s**.
- Embedding/index build: **33.23 s** for 77 documents on NVIDIA H100 NVL.
- Warm GPU utilization median: **87.8%**; peak GPU memory was approximately **52.7 GiB**.
- No prompt exceeded the configured 8,192-token context window.

## Interpretation

Retrieval is the dominant quality bottleneck: every answer with incomplete ground-truth evidence was incorrect, while 23 of 25 questions with complete evidence were answered correctly. Questions 11 and 48 had complete evidence but incorrect answers, so those are the clearest generation/reasoning failures. Provenance trails answer quality because several correct answers omitted supporting join rows.

Answer mismatches: 2, 6, 8, 9, 11, 12, 13, 14, 15, 16, 19, 20, 21, 23, 26, 28, 30, 31, 32, 36, 37, 40, 42, 45, 47, 48.
