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

## Evaluation and dashboard metrics

`tpch/analyze_results.py` evaluates a completed run and creates three files in
`tpch/results_analysis/`:

- `summary.md`: run-level results and interpretation.
- `dashboard.svg`: dependency-free visualization of quality, retrieval,
  latency, and GPU utilization.
- `per_question.csv`: the derived metrics for every question.

These are post-run evaluation products. They are not generated by `Why.py`, and
their quality metrics are not fields from Ollama or FAISS. The analysis compares
`test_pipeline.json` with the expected answers and evidence in
`ground_truth2.json`. The entries in these two files must be in the same question
order and have the same length; otherwise the analysis stops with an error.

Regenerate all three outputs from the repository root with:

```bash
python3 tpch/analyze_results.py
```

### Answer exact match

For each question, the predicted `answer.answer` list is compared with the
ground-truth `answer` list. Before comparison, each element is converted to a
string, leading and trailing whitespace is removed, consecutive whitespace is
collapsed, text is converted to lowercase, and the lists are sorted.

Consequently, capitalization, extra whitespace, answer order, and numeric JSON
types do not affect the result. The comparison remains strict: alternative date
formats, additional answers, missing answers, abbreviations, and semantically
equivalent wording can still count as mismatches.

The run-level answer accuracy is:

```text
answer accuracy = questions with answer_exact=true / total questions
```

### Evidence references and exact provenance

Evidence identifiers have the form `<table>_<row number>`, for example
`students_0`. The analyzer extracts these identifiers from both the predicted
`answer.why` and ground-truth `why` fields. Identifier order and grouping inside
braces do not affect the comparison.

`why_exact` is true only when the set of predicted evidence identifiers exactly
equals the ground-truth set. It is false when any required row is missing or any
unsupported row is added. This is stricter than answer correctness: a correct
answer can have incomplete provenance.

```text
provenance accuracy = questions with why_exact=true / total questions
```

### Evidence retrieval recall

The analyzer also compares the ground-truth evidence identifiers with the
documents returned by FAISS. A retrieved document identifier is constructed
from `retrieved_documents[].metadata.source` and
`retrieved_documents[].metadata.row`.

For one question:

```text
evidence recall = required ground-truth rows present in the top-k results
                  -------------------------------------------------------
                  total required ground-truth rows
```

`full evidence retrieved` means evidence recall is exactly 100%. `Any evidence
found` means evidence recall is greater than zero. Mean evidence recall is the
arithmetic mean of the per-question recall values.

For questions requiring joins, every ground-truth row in the join path counts
as required evidence. For example, if the ground truth requires a course,
enrollment, and student row, retrieving only the course and student gives
evidence recall `2/3` and does not count as full evidence.

This metric measures whether the expected rows are available to the LLM. It does
not establish that every retrieved row is relevant, that the LLM used the rows,
or that the ground truth is the only valid evidence set. Because retrieval uses
`k=10` in the current run, “full evidence” means full evidence within the top 10.

`first_evidence_rank` in `per_question.csv` is the best (lowest) FAISS rank of
any required ground-truth row. It is empty when none of the required rows was
retrieved.

### Conditional answer accuracy

The report separates questions into two groups:

```text
accuracy with full retrieval = correct answers among questions with recall=1
                                ----------------------------------------------
                                questions with recall=1

accuracy with incomplete retrieval = C.A. among questions with recall<1
                                      --------------------------------------
                                      questions with recall<1
```

This comparison helps distinguish retrieval failures from likely generation or
reasoning failures. It demonstrates association, not causation: question
difficulty and the number of required join rows may affect both retrieval and
answer accuracy.

### Empty answers and parsing

`empty_answer` is true when the parsed `answer.answer` list is empty. It is
separate from `parsing_succeeded`: a syntactically valid JSON response can parse
successfully while intentionally or incorrectly containing an empty answer.

If generation fails after the configured retries, `generation_error` contains
the final exception text and `generation_attempts` contains the number of
requests made. A successful first request has one attempt and a null error.

### Cold- and warm-latency aggregates

The dashboard treats question 1 as cold and questions 2 through the end as warm,
following `generation_request.sequence_state`. Warm statistics therefore exclude
question 1:

- **Warm mean** is the arithmetic mean of `timing_seconds.total` for questions
  2 through N.
- **Warm median** is the median of those warm total latencies.
- **Warm p95** is the nearest-rank 95th percentile: sort warm latencies and
  select item `ceil(0.95 * N)` using one-based indexing.
- **Cold-start share** is question 1 total latency divided by the sum of total
  latency for all questions.

These calculations describe the observed sequence. A later model eviction or
server restart can produce another cold-like request even though it is labelled
`warm_subsequent_request`; inspect `model_load_seconds` when this is suspected.

### Dashboard interpretation

The dashboard contains four summary cards and the following plots:

- **Quality and retrieval:** run-level answer exact match, exact provenance,
  full evidence retrieval, and any-evidence retrieval rates.
- **Latency by question:** total question latency on a logarithmic vertical
  scale so the cold-start outlier and warm requests remain visible together.
  Question 1 is red and later questions are blue.
- **Evidence recall by question:** green means 100%, yellow means partial, and
  red means no required evidence was retrieved. Each tile also prints the exact
  recall percentage.
- **Average GPU utilization:** per-request `nvidia-smi` sample averages for warm
  requests only. This is device-wide observed utilization, with the same
  sampling limitations described in the GPU observations section.

The key-finding panel compares answer accuracy conditional on full versus
incomplete evidence retrieval. The dashboard currently labels the model,
retrieval `k`, question count, and GPU for this experiment; when analyzing a
different experimental configuration, those labels should be read from the
input metadata rather than assumed to remain constant.

### `per_question.csv` fields

| Field | Meaning |
| --- | --- |
| `question_number`, `question` | One-based position and question text. |
| `answer_exact` | Normalized answer-list exact match. |
| `why_exact` | Exact set match for predicted and required evidence identifiers. |
| `empty_answer` | Whether the parsed answer list is empty. |
| `retrieval_recall` | Fraction of required ground-truth rows found in top-k. |
| `first_evidence_rank` | Best rank of any required evidence row, or empty if absent. |
| `retrieval_seconds` | Query embedding plus FAISS search latency. |
| `generation_seconds` | Complete generation-stage latency. |
| `total_seconds` | End-to-end latency for the question. |
| `prompt_tokens`, `output_tokens` | Local prompt count and Ollama output count. |
| `gpu_utilization` | Average sampled GPU utilization during generation. |
| `peak_gpu_memory_mib` | Highest sampled device-memory use during generation. |

