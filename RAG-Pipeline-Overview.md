# RAG Pipeline Overview
Pipeline for explainable querying over tabular data.
## 1. Datasets

### 1.1 Toy dataset
Manually created.
- Size: 44k, 5 to 10 rows per file.
- Stored in `csv_data/`
- Files include:
  - `classrooms.csv`
  - `courses.csv`
  - `departments.csv`
  - `enrollments.csv`
  - `exams.csv`
  - `grades.csv`
  - `students.csv`
  - `teacherResearchAreas.csv`
  - `teachers.csv`
  - `thesis.csv`
- Schema definition in `schemaTOY.txt`
  - Relational tables for students, courses, departments, enrollments, exams, grades, teachers, and thesis
- Used by:
  - `iterativeWithPlan.py`
  - `LlmWithRag.py`
  - `planGenerator.py`

### 1.2 TPC-H dataset
From the TPC-H benchmark. 
- Size: 0.1GB 
- Schema definition in `schemaTPCH.txt`
  - Tables include: `CUSTOMER`, `ORDERS`, `LINEITEM`, `PART`, `SUPPLIER`, `PARTSUPP`, `NATION`, `REGION`
- Approximate table sizes:

| Table | Approximate rows |
|---|---|
| `REGION` | 5 |
| `NATION` | 25 |
| `SUPPLIER` | 1,000 |
| `CUSTOMER` | 15,000 |
| `PART` | 20,000 |
| `PARTSUPP` | 80,000 |
| `ORDERS` | 150,000 |
| `LINEITEM` | Approximately 600,000 |

- Used by:
  - `tpch/LlmWithRagTpch.py`
  - `tpch/iterativeRAG.py`

## 2. Pipeline types in the repository

### 2.1 Iterative RAG pipeline 

This pipeline follows the iterative design from the image: the system performs repeated retrieval and generation cycles, progressively extending the question with previous answers.

#### 2.1.1 How it works

- At each iteration, the system retrieves the top-k most relevant documents for the current query.
- The LLM receives the current retrieved context and generates a structured answer.
- The generated answer is appended to the original question to form the next iteration's query.
- The process repeats for a fixed number of steps (empirically set to 5).

#### 2.1.2 Adaptive retrieval

- The pipeline can increase the retrieval window `k` using the number of distinct tuples referenced by the previous witness sets.
- Let `W` be the witness sets produced by the model in the previous iteration, where each set is `{{table_row1, table_row2, ...}}`.
- Extract unique table-row identifiers and compute `∆k` as the number of distinct rows.
- Update retrieval size with `k_next = k_current + ∆k`.
- This keeps the context compact while adding evidence proportional to the model's actual dependencies.
- The loop is repeated until `max_iteration` (=5) is reached.

#### 2.1.3 State and loop structure

- The internal state includes:
  - `original_question`
  - `current_question`
  - `context`
  - `answer`
  - `k`
- The loop is implemented using a RAG graph in LangGraph.
- Each iteration performs:
  1. Retrieval using the current query.
  2. Generation using the retrieved context and the updated prompt.

#### 2.1.4 Implementation notes

- The iterative design is represented most directly in `tpch/iterativeRAG.py`.
- `iterativeWithPlan.py` also implements an iterative loop, but with a different logic.

### 2.2 Plan-based RAG pipeline

This pipeline first generates a reasoning plan from the question, then executes the plan step-by-step with retrieval and generation.

#### 2.2.1 Core behavior

- A separate plan generator LLM decomposes the original question into atomic natural language steps.
- The plan is a list of step descriptions returned as a JSON-style array.
- For each step:
  - Retrieve top-k documents relevant to the step's sub-question.
  - Generate an answer based only on the retrieved context.
  - Use the step output to refine the next sub-question.

#### 2.2.2 Plan generation prompt

- The prompt asks the model to output a list of atomic steps with no extra text.
- The expected format is:
  ```json
  [
      "Step 1: <action>",
      "Step 2: <action>",
      ...
  ]
  ```
- The prompt is conditioned on the question and the database schema.

#### 2.2.3 Implementation notes

- Plan generation is implemented in `planGenerator.py`.
- The plan-based execution is implemented in `iterativeWithPlan.py`.
- The system uses `max_iterations` equal to the plan length and executes each subtask sequentially.

### 2.3 Standard RAG pipeline with context retrieval

Implemented in `LlmWithRag.py`

#### 2.3.1 Key behavior

- Uses `csv_data/` and FAISS index in `faiss_index/`
- Reads questions from `question.txt`
- Retrieves top-10 documents with `vector_store.similarity_search(question, k=10)`
- Sends the retrieved context + question to a prompt pulled from `rlm/rag-prompt`
- Expects JSON output and parses it with `JsonOutputParser`

#### 2.3.2 LLM details

- Default LLM: `ChatOllama(model="llama3:70b", temperature=0)`
- Alternative option: `llama3-8b-8192`
- Embeddings: `sentence-transformers/all-mpnet-base-v2`

### 2.4 TPCH RAG pipeline

Implemented in `tpch/LlmWithRagTpch.py`

#### 2.4.1 Key behavior

- Uses TPCH CSV data in `tpch/csv_data_tpch/`
- Loads all CSV rows into documents
- Uses BM25 retrieval instead of FAISS (tried both)
- Retrieves the top 10 BM25 documents
- Builds a prompt that demands answer + witness sets
- Parses output into `answer` and `why`

#### 2.4.2 LLM and retrieval details

- LLM: `llama3-70b-8192`, `llama3-8b-8192`
- Embeddings: `sentence-transformers/all-mpnet-base-v2`
- Retriever: `BM25Retriever.from_documents(documents)`

### 2.5 Empty-context baseline

Implemented in `LlmEmptyContext.py`

#### 2.5.1 Key behavior

- Does not use retrieved context or vector store
- Sends only the question to the LLM
- Prompt requests an answer and explanation as JSON
- Useful as a baseline for comparing pure LLM reasoning versus RAG

#### 2.5.2 LLM details

- LLM: `llama3-70b-8192`,`llama3-8b-8192`
- Prompt from `hub.pull("rlm/rag-prompt")`

## 3. Data flow model for the iterative RAG pipeline

### 3.1 Input

- `question` from `questions.json`
- Toy schema from `schemaTOY.txt`
- CSV data from `csv_data/`

### 3.2 Preprocessing

- Load CSV files with `CSVLoader`
- Convert each CSV row into a LangChain `Document`
- Build or load FAISS index (`faiss_index/`)
- **LLM calls in preprocessing:** None (unless using plan generation)

### 3.3 Retrieval

- Encode `current_question` via `sentence-transformers/all-mpnet-base-v2`
- Query the FAISS index for `k=10`
- Return the top-10 similar document chunks
- **LLM calls in retrieval:** None

### 3.4 LLM call

- Build prompt with:
  - the retrieved context
  - the current question
  - explicit format instructions for JSON output
- Invoke the model via `llm.invoke(final_prompt)`
- Extract the JSON block from the model output
- Parse `answer` and optionally use it to update the next step
- **LLM calls per iteration:** 1 call to `llm.invoke()` with variable token count

### 3.5 Iteration

- If a plan exists, repeat the retrieve/generate loop for each step
- On each iteration, the question is extended with the previous answer
- This can improve answer accuracy by feeding intermediate conclusions back into retrieval
- **Total LLM calls:** 1 per iteration × number of iterations (typically 1–5)

### 3.6 Token accounting for carbon footprint

- **Input tokens:** Retrieved documents + question + system prompt (typically 500–3,000 tokens)
- **Output tokens:** Generated answer (typically 50–500 tokens)
- **Total per iteration:** 550–3,500 tokens
- **Total per question:** 550 tokens × iterations = 550–17,500 tokens
- **Batch effect:** Processing multiple questions sequentially (no batching implemented)


## 5. External APIs and cloud services

This section details all external services and API calls made during pipeline execution, which are relevant for carbon footprint analysis.

### 5.1 LLM providers

#### 5.1.1 Groq API (Cloud-based, paid service)

- **Endpoints used:** `model_provider="groq"` via LangChain `init_chat_model()`
- **Models:**
  - `llama3-70b-8192` (70 billion parameters)
  - `llama3-8b-8192` (8 billion parameters)
  - `mistral-saba-24b` (24 billion parameters)
- **API Key:** `GROQ_API_KEY` environment variable
- **Authentication:** API key based
- **Usage:**
  - Used in `LlmWithRag.py`, `tpch/LlmWithRagTpch.py`, `LlmEmptyContext.py`
  - Each LLM call makes an HTTP POST to Groq's inference API


#### 5.1.2 Ollama (Local/self-hosted, no cloud calls) --> FINAL CHOICE FOR THE PIPELINE

- **Endpoints:** Local HTTP server (default: http://localhost:11434)
- **Models:**
  - `llama3:8b` (8 billion parameters)
  - `llama3:70b` (70 billion parameters)
  - `llama3.1-8b-ft` (fine-tuned variant)
  - `mixtral:latest` (Mistral mixture of experts)
- **Usage:**
  - Used in `iterativeWithPlan.py`, `LlmWithRag.py`, `tpch/iterativeRAG.py`, `planGenerator.py`
  - No external API calls; runs locally on user hardware
- Run on lig GPUs (gpu10)

### 5.2 Embedding models

#### 5.2.1 HuggingFace embeddings (Partially cloud-based)

- **Model:** `sentence-transformers/all-mpnet-base-v2`
- **Provider:** Hugging Face Model Hub
- **Library:** `HuggingFaceEmbeddings` from `langchain_huggingface`
- **First-run behavior:**
  - Downloads model weights (~140 MB) from Hugging Face hub on first use
  - Caches locally in `~/.cache/huggingface/hub/`
  - Subsequent runs use local cached version (no API call)
- **Inference:**
  - Runs locally on user hardware after download
  - Embeddings computed in-process, not via API


### 5.3 Vector stores and retrievers

#### 5.3.1 FAISS (Facebook AI Similarity Search)

- **Location:** Local file-based index (`faiss_index/`, `tpch/faiss_index/`)
- **Indexing:** Built from documents using `FAISS.from_documents()`
- **Serialization:** Saved locally with `vector_store.save_local()`
- **Retrieval:** `similarity_search()` uses L2 distance (default) or cosine similarity
- **No external API calls**

- One-time indexing cost (CPU-intensive)
- Queries use local memory and CPU

#### 5.3.2 BM25 Retriever

- **Implementation:** `BM25Retriever.from_documents()` from LangChain
- **Algorithm:** Okapi BM25 (probabilistic ranking function)
- **Location:** In-memory, no persistent storage
- **No external API calls**

- CPU-based scoring, no GPU acceleration

### 5.4 Monitoring and observability

#### 5.4.1 LangSmith API (Cloud-based tracing)

- **Endpoint:** LangSmith cloud service
- **API Key:** `LANGSMITH_API_KEY` environment variable
- **Configuration:**
  - `LANGSMITH_TRACING` flag (true/false) controls whether traces are sent
  - Files set this variably:
    - `iterativeWithPlan.py`: `LANGSMITH_TRACING = "false"` (tracing disabled)
    - `LlmEmptyContext.py`: `LANGSMITH_TRACING = "true"` (tracing enabled)
    - Other files vary by use case
- **Purpose:** Logs LLM calls, intermediate steps, and chains for debugging/monitoring
- **API calls made:**
  - Each LLM invocation (if tracing enabled) sends metadata to LangSmith
  - Includes prompt, response, latency, token count

### 5.5 Data access and I/O

#### 5.5.1 Local CSV data

- **Location:** `csv_data/` (toy dataset) or `tpch/csv_data_tpch/` (TPC-H dataset)
- **I/O:** All disk-based
- **CSVLoader:** Parses CSV files using LangChain's `CSVLoader`
- **No external API calls**

### 5.6 Dependency libraries and frameworks

| Library | Purpose | Cloud calls | Execution |
|---------|---------|------------|--------|
| **LangChain** | Core orchestration | Depends on use | Depends on config |
| **LangGraph** | State graph execution | None | Local CPU |
| **Pydantic** | Data validation/parsing | None | Local CPU |
| **Scikit-learn** | BM25 vectorizer | None | Local CPU |
| **sentence-transformers** | Embedding inference | Initial download only | Local GPU/CPU |
| **FAISS** | Vector search | None | Local CPU/GPU |
| **NumPy/SciPy** | Numerical operations | None | Local hardware |

