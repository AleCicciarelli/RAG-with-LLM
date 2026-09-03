# Why.py measurements

This document describes the data collected by `tpch/Why.py`, how each value is
measured, and how it should be interpreted.

## Output files

The pipeline produces two JSON files:

- `tpch/test_pipeline.json` contains the answer and the measurements associated
  with each question.
- `tpch/timing_metrics.json` contains run-level configuration, embedding
  measurements, and a checkpointed copy of the per-question measurements.

`timing_metrics.json` is rewritten after indexing and after every completed
question. Consequently, measurements for completed questions normally survive
an interrupted OAR job. `test_pipeline.json` is written after all questions
finish.

The OAR runner also writes:

- `tpch/oar_logs/why_<job-id>.log`: pipeline standard output and errors.
- `tpch/oar_logs/ollama_<job-id>.log`: Ollama server output and errors.

All durations use a monotonic high-resolution clock (`time.perf_counter`) and
are stored in seconds. GPU memory values are in MiB.

## Run metadata

| Field | Collection method | Meaning |
| --- | --- | --- |
| `started_at_utc` | UTC clock when the Python process starts | Start timestamp for the experiment. |
| `finished_at_utc` | UTC clock after all results are written | Completion timestamp. It is absent if the job is interrupted. |
| `oar_job_id` | `OAR_JOB_ID` environment variable | OAR job that produced the measurements. It is `null` outside OAR. |
| `question_count` | Number of completed questions | Added when the complete run finishes. |
| `generation_model.name` | Pipeline configuration | Ollama model used for generation. |
| `generation_model.context_window_tokens` | `num_ctx` supplied to Ollama | Configured context-window size. |

## CSV embedding and FAISS index construction

CSV rows are loaded as LangChain documents. Documents are embedded in batches
with `sentence-transformers/all-mpnet-base-v2`, and the vectors are added to the
FAISS index.

When CUDA is available, the embedding model is explicitly placed on CUDA.
`torch.cuda.synchronize()` is called around measured sections so queued CUDA
work is complete before the wall-clock measurement stops.

| Field | Collection method | Meaning |
| --- | --- | --- |
| `embedding.model` | Pipeline configuration | Embedding model identifier. |
| `embedding.batch_size` | Pipeline configuration | Maximum documents sent to one embedding call. |
| `embedding.device` | `torch.cuda.is_available()` | `cuda` or `cpu`. Always check this field before treating a run as a GPU experiment. |
| `embedding.gpu_name` | `torch.cuda.get_device_name(0)` | CUDA device reported to the Python process. |
| `embedding.index_action` | FAISS index existence and `REBUILD_FAISS_INDEX` | `created_index` means embedding was performed; `loaded_existing_index` means it was skipped. |
| `embedding.documents_embedded` | Sum of document batches | Total number of CSV rows embedded during this run. |
| `embedding.wall_seconds` | Sum of synchronized per-file embedding durations | Time spent embedding documents and adding their vectors to FAISS. It excludes CSV loading between files and final index serialization. |
| `embedding.index_build_wall_seconds` | Clock around the complete file-processing loop | Overall indexing-loop duration, including CSV loading and orchestration overhead. It does not include the final `save_local` call. |
| `embedding.files[].file` | CSV filename | File being measured. |
| `embedding.files[].document_count` | Number of loaded CSV documents | Rows embedded from that file. |
| `embedding.files[].embedding_wall_seconds` | Synchronized clock around that file's embedding batches | Embedding and FAISS insertion time for the file. |

Embedding measurements are only produced while building an index. The OAR
runner defaults to `REBUILD_FAISS_INDEX=1` for this reason. If an existing index
is loaded, durations and document counts that cannot be measured are left
`null` or zero rather than inferred.

## Retrieval

Each question is embedded with the same embedding model and passed to
`FAISS.similarity_search_with_score`. The top 10 documents become the context
sent to the LLM. Ground-truth rows are not used by the active pipeline.

| Field | Collection method | Meaning |
| --- | --- | --- |
| `retrieval.method` | Pipeline configuration | `faiss_similarity_search_with_score`. |
| `retrieval.k` | Pipeline configuration | Maximum number of neighbours requested per question. |
| `retrieval_seconds` or `timing_seconds.retrieval` | Synchronized clock around query embedding and FAISS search | Complete retrieval latency for one question. |
| `retrieved_document_count` | Length of the returned document list | Actual number of documents returned; it can be less than `k`. |
| `retrieved_documents[].rank` | FAISS result order | One-based retrieval rank. |
| `retrieved_documents[].score` | Score returned by FAISS | With the current default Euclidean FAISS configuration, a lower score means a closer vector. Scores should only be compared between runs using the same model, normalization, and distance configuration. |
| `retrieved_documents[].metadata` | Metadata stored with the FAISS document | Identifies the source CSV and row used as context. |

Retrieval time includes query embedding. It does not include prompt construction
or LLM generation.

## Generation and parsing

The prompt is built from the question, retrieved documents, and prompt template.
The Ollama response is streamed so latency to the first non-empty output chunk
can be measured. The complete text is then parsed with `JsonOutputParser`.

| Field | Collection method | Meaning |
| --- | --- | --- |
| `timing_seconds.generation` | Clock around prompt preparation, streamed Ollama call, and JSON parsing | Complete generation-stage latency visible to the pipeline. |
| `generation_request.time_to_first_token_seconds` | Time from immediately before `llm.stream()` to the first non-empty chunk | Initial response latency. It includes request and prompt-processing latency. |
| `generation_request.llm_stream_seconds` | Time from immediately before streaming until the complete response arrives | LLM request duration measured by the client. It excludes JSON parsing. |
| `generation_request.ollama_total_seconds` | Ollama `total_duration` response metadata | Ollama's server-side total request duration, converted from nanoseconds. |
| `generation_request.model_load_seconds` | Ollama `load_duration` response metadata | Server time spent loading or preparing the model, converted from nanoseconds. |
| `parsing_succeeded` | Result of `JsonOutputParser.parse()` | `true` when the response is valid under the expected JSON structure; otherwise `false`, and the stored answer is empty. |
| `timing_seconds.total` | Clock from immediately before retrieval until generation and parsing finish | End-to-end question latency. Small orchestration overhead means it can be slightly larger than retrieval plus generation. |

### Cold and warm requests

`sequence_state` labels question 1 as `cold_first_request` and later questions
as `warm_subsequent_request`.

A cold request may need to load model weights, allocate GPU memory, initialize
the execution backend, and populate caches. A warm request normally reuses the
resident model and is therefore faster. The sequence label alone does not prove
that a load happened: `model_load_seconds` is the measured Ollama value to use
when evaluating cold-start overhead. A model eviction or Ollama restart can also
make a later request behave like a cold request.

## Token counts and prompt truncation

| Field | Collection method | Meaning |
| --- | --- | --- |
| `token_counts.original_prompt` | Full prompt encoded locally with the Llama 3 tokenizer before submission | Number of tokens requested by the pipeline before Ollama can truncate anything. |
| `token_counts.ollama_prompt_processed` | Ollama `prompt_eval_count` metadata | Prompt tokens actually evaluated and reported by Ollama. Chat-template tokens can make this differ slightly from the local count. |
| `token_counts.output` | Ollama `eval_count` metadata | Generated tokens reported by Ollama. |
| `prompt_truncation.exceeds_configured_context` | Local prompt count compared with `LLM_CONTEXT_WINDOW` | Whether the untruncated prompt is larger than the configured context window. |
| `prompt_truncation.detected` | Same context-window comparison | `true` establishes that the prompt cannot fit and truncation is required. |
| `prompt_truncation.check` | Tokenizer availability status | Describes how the check was performed, or says `unavailable_tokenizer`. |

The tokenizer is initialized before question timing begins so its download or
startup does not distort the first question. If the matching tokenizer cannot
be loaded, local prompt count and truncation fields are `null`; the pipeline does
not substitute an approximate token count.

A `false` truncation value means the locally tokenized prompt fits within the
configured window. Ollama's chat template and any backend-specific token
reservation can still affect its processed count, so both original and reported
prompt counts are retained for comparison.

## GPU observations

GPU measurements are collected independently during CSV embedding and during
each LLM generation. A background thread calls `nvidia-smi` every 0.2 seconds
and reads the first GPU visible to the OAR job.

| Field | Collection method | Meaning |
| --- | --- | --- |
| `average_gpu_utilization_percent` | Arithmetic mean of observed `nvidia-smi` utilization values | Average of the samples captured during that phase. It is not converted into estimated GPU-seconds. |
| `peak_gpu_memory_mib` | Maximum observed `nvidia-smi memory.used` sample | Highest memory value seen by the sampler. This is an observed maximum, not a continuous hardware high-water mark. |
| `average_gpu_memory_mib` | Arithmetic mean of observed memory values | Average allocated GPU memory across captured samples. |
| `gpu_samples` | Number of successful `nvidia-smi` reads | Sample population used by the aggregate fields. |
| `gpu_sample_interval_seconds` | Sampler configuration | Target time between observations. |
| `gpu_observations[].elapsed_seconds` | Monotonic sample time minus phase start | Sample position within embedding or generation. |
| `gpu_observations[].utilization_percent` | Direct `nvidia-smi` reading | GPU utilization at that observation. |
| `gpu_observations[].memory_used_mib` | Direct `nvidia-smi` reading | Device memory in use at that observation. |
| `gpu_observations[].memory_total_mib` | Direct `nvidia-smi` reading | Total device memory reported at that observation. |

Raw observations are saved so aggregate values can be audited. Sampling cannot
observe a spike shorter than the sampling interval. Values also describe the
visible GPU as a whole, so the OAR allocation should provide an exclusive GPU if
the measurements must represent only this pipeline. If `nvidia-smi` is missing
or no samples succeed, aggregate fields are `null` and `gpu_samples` is zero.

## Running under OAR

Adapt the resource expression to the cluster and submit the provided runner:

```bash
oarsub -l 'host=1/gpu=1,walltime=04:00:00' -S ./tpch/run_why_oar.sh
```

If dependencies are installed in a virtual environment, export its path before
submission according to the cluster's environment-forwarding policy:

```bash
export VENV_PATH=/path/to/venv
oarsub -l 'host=1/gpu=1,walltime=04:00:00' -S ./tpch/run_why_oar.sh
```

To reuse an existing FAISS index instead of measuring embedding again, set
`REBUILD_FAISS_INDEX=0`.
