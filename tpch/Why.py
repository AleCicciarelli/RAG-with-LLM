import os
from pathlib import Path
from itertools import islice
from tpch_data import data_files, iter_rows, index_manifest, load_questions
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict, Set
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain import hub
import json
import csv
import re
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List
import time
import subprocess
import threading
from datetime import datetime, timezone
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOllama

try:
    import torch
except ImportError:
    torch = None

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

os.environ["LANGSMITH_TRACING"] = "false" 
os.environ["LANGSMITH_API_KEY"] = "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"


# MISTRAL by Groq
#llm = init_chat_model("mistral-saba-24b", model_provider="groq", temperature = 0)
#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 token hugging-face

LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "llama3:70b")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
LLM_CONTEXT_WINDOW = 8192
EMBEDDING_BATCH_SIZE = 200
OLLAMA_MAX_ATTEMPTS = max(1, int(os.environ.get("OLLAMA_MAX_ATTEMPTS", "3")))
OLLAMA_RETRY_DELAY_SECONDS = max(
    0.0, float(os.environ.get("OLLAMA_RETRY_DELAY_SECONDS", "10"))
)
repo_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tpch_folder = Path(__file__).resolve().parent
csv_folder = Path(os.environ.get("TPCH_DATA_DIR", tpch_folder / "tpch-data")).resolve()
faiss_index_folder = Path(os.environ.get("FAISS_INDEX_DIR", tpch_folder / "faiss_index")).resolve()
output_folder = Path(os.environ.get("OUTPUT_DIR", tpch_folder / "runs" / "tpch")).resolve()
questions_path = Path(os.environ.get("QUESTIONS_FILE", tpch_folder / "questions.json")).resolve()
output_filename = output_folder / "test_pipeline.json"
timing_filename = output_folder / "timing_metrics.json"
output_folder.mkdir(parents=True, exist_ok=True)
all_files = data_files(csv_folder)
data = load_questions(questions_path)
questions = list(data)
expected_manifest = index_manifest(all_files, EMBEDDING_MODEL_NAME)
manifest_path = faiss_index_folder / "manifest.json"


llm = ChatOllama(model=LLM_MODEL_NAME, temperature=0, num_ctx=LLM_CONTEXT_WINDOW)

# Embedding model: Hugging Face
#embedding_model = HuggingFaceEmbeddings(model_name="/home/ciccia/.cache/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2/snapshots/12e86a3c702fc3c50205a8db88f0ec7c0b6b94a0")
REQUIRE_CUDA = os.environ.get("REQUIRE_CUDA", "1").lower() not in {"0", "false", "no"}
cuda_available = bool(torch is not None and torch.cuda.is_available())
if REQUIRE_CUDA and not cuda_available:
    torch_version = getattr(torch, "__version__", "not installed")
    torch_cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
    raise RuntimeError(
        "CUDA is required for embeddings, but PyTorch cannot access a GPU. "
        f"torch={torch_version}, torch CUDA build={torch_cuda_version}, "
        f"CUDA_VISIBLE_DEVICES={visible_devices}. Install a PyTorch wheel compatible "
        "with the compute node's NVIDIA driver (the reported driver supports CUDA "
        "12.8, so use the cu128 wheel), and run inside a GPU allocation. Set "
        "REQUIRE_CUDA=0 only when an intentional CPU fallback is desired."
    )
embedding_device = "cuda" if cuda_available else "cpu"
if cuda_available:
    print(
        f"PyTorch GPU enabled: {torch.cuda.get_device_name(0)} "
        f"(torch={torch.__version__}, CUDA={torch.version.cuda})"
    )
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": embedding_device},
)
#embedding_model = HuggingFaceEmbeddings(
#    model_name="BAAI/bge-small-en-v1.5",
#    model_kwargs={"device": "cuda"},  
#    encode_kwargs={"normalize_embeddings": True}
#)
""" Indexing part """


timing_metrics = {
    "dataset": {"name": "tpch", "data_dir": str(csv_folder),
                "questions_file": str(questions_path), "index_dir": str(faiss_index_folder),
                "provenance": "explicit_table_rownum"},
    "started_at_utc": datetime.now(timezone.utc).isoformat(),
    "oar_job_id": os.environ.get("OAR_JOB_ID"),
    "retrieval": {
        "method": "faiss_similarity_search_with_score",
        "k": 10,
        "score_interpretation": "lower_is_more_similar_for_default_faiss_euclidean_distance",
    },
    "embedding": {
        "model": EMBEDDING_MODEL_NAME,
        "batch_size": EMBEDDING_BATCH_SIZE,
        "device": embedding_device,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "index_action": None,
        "wall_seconds": None,
        "index_build_wall_seconds": None,
        "average_gpu_utilization_percent": None,
        "peak_gpu_memory_mib": None,
        "average_gpu_memory_mib": None,
        "gpu_samples": 0,
        "documents_embedded": 0,
        "files": [],
    },
    "generation_model": {
        "name": LLM_MODEL_NAME,
        "context_window_tokens": LLM_CONTEXT_WINDOW,
    },
    "gpu_measurement": {
        "source": "nvidia-smi",
        "sample_interval_seconds": 0.2,
        "scope": "first GPU visible to the OAR job",
        "peak_definition": "maximum observed sample",
    },
    "questions": [],
}


def save_timing_metrics():
    """Checkpoint timings so an interrupted OAR job keeps completed measurements."""
    with open(timing_filename, "w", encoding="utf-8") as timing_file:
        json.dump(timing_metrics, timing_file, indent=2, ensure_ascii=False)


def sample_gpu(stop_event, samples, interval_seconds=0.2):
    """Capture raw nvidia-smi utilization and memory observations."""
    while not stop_event.is_set():
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=2,
                check=True,
            )
            # CUDA_VISIBLE_DEVICES normally exposes one GPU in an OAR allocation.
            utilization, memory_used, memory_total = completed.stdout.splitlines()[0].split(",")
            samples.append({
                "elapsed_seconds": time.perf_counter(),
                "utilization_percent": float(utilization.strip()),
                "memory_used_mib": float(memory_used.strip()),
                "memory_total_mib": float(memory_total.strip()),
            })
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            pass
        stop_event.wait(interval_seconds)


def summarize_gpu_samples(samples, phase_started):
    if not samples:
        return {
            "average_gpu_utilization_percent": None,
            "peak_gpu_memory_mib": None,
            "average_gpu_memory_mib": None,
            "gpu_samples": 0,
            "gpu_sample_interval_seconds": 0.2,
            "gpu_observations": [],
        }
    observations = [
        {
            **sample,
            "elapsed_seconds": sample["elapsed_seconds"] - phase_started,
        }
        for sample in samples
    ]
    return {
        "average_gpu_utilization_percent": sum(
            sample["utilization_percent"] for sample in samples
        ) / len(samples),
        "peak_gpu_memory_mib": max(sample["memory_used_mib"] for sample in samples),
        "average_gpu_memory_mib": sum(
            sample["memory_used_mib"] for sample in samples
        ) / len(samples),
        "gpu_samples": len(samples),
        "gpu_sample_interval_seconds": 0.2,
        "gpu_observations": observations,
    }


def get_token_count(text):
    """Count the untruncated prompt with the matching Llama 3 tokenizer, if available."""
    global prompt_tokenizer
    if prompt_tokenizer is False:
        return None
    if prompt_tokenizer is None:
        if AutoTokenizer is None:
            prompt_tokenizer = False
            return None
        try:
            prompt_tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Meta-Llama-3-8B-Instruct"
            )
        except Exception as exc:
            print(f"Prompt tokenizer unavailable; truncation cannot be checked exactly: {exc}")
            prompt_tokenizer = False
            return None
    return len(prompt_tokenizer.encode(text, add_special_tokens=True))


prompt_tokenizer = None

# Verify if the FAISS files already exist
index_exists = all(
    os.path.isfile(os.path.join(faiss_index_folder, filename))
    for filename in ("index.faiss", "index.pkl")
)
force_index_rebuild = os.environ.get("REBUILD_FAISS_INDEX", "0") == "1"
manifest_matches = False
if manifest_path.is_file():
    try:
        manifest_matches = json.loads(manifest_path.read_text()) == expected_manifest
    except (ValueError, OSError):
        pass
if index_exists and not force_index_rebuild and not manifest_matches:
    print("Index metadata does not match TPC-H inputs; rebuilding the index.")
if index_exists and manifest_matches and not force_index_rebuild:
    # Load the FAISS index folder (allow_dangerous_deserialization=True just because we create the files and so we can trust them)
    vector_store = FAISS.load_local(str(faiss_index_folder), embedding_model, allow_dangerous_deserialization=True)
    timing_metrics["embedding"]["index_action"] = "loaded_existing_index"
    print("FAISS index successfully loaded")
else:
    batch_size = EMBEDDING_BATCH_SIZE

    # Initialize vector_store before the loop
    vector_store = None

    # Synchronization makes the wall clock include queued CUDA work.
    gpu_samples = []
    gpu_sampler_stop = threading.Event()
    gpu_sampler = None
    if cuda_available:
        torch.cuda.synchronize()
        gpu_sampler = threading.Thread(
            target=sample_gpu,
            args=(gpu_sampler_stop, gpu_samples),
            daemon=True,
        )
        gpu_sampler.start()
    embedding_started = time.perf_counter()

    for file_path in all_files:
        rows = iter_rows(file_path)
        document_count = 0
        if cuda_available:
            torch.cuda.synchronize()
        file_embedding_started = time.perf_counter()
        while batch_rows := list(islice(rows, batch_size)):
            batch_docs = [Document(page_content=content, metadata=metadata)
                          for content, metadata in batch_rows]
            document_count += len(batch_docs)
            timing_metrics["embedding"]["documents_embedded"] += len(batch_docs)
            if vector_store is None: # Only create for the first batch
                vector_store = FAISS.from_documents(batch_docs, embedding=embedding_model)
            else:
                vector_store.add_documents(batch_docs)
        if cuda_available:
            torch.cuda.synchronize()
        timing_metrics["embedding"]["files"].append({
            "file": file_path.name,
            "document_count": document_count,
            "embedding_wall_seconds": time.perf_counter() - file_embedding_started,
        })

    if cuda_available:
        torch.cuda.synchronize()
        gpu_sampler_stop.set()
        gpu_sampler.join(timeout=3)
    embedding_wall_seconds = time.perf_counter() - embedding_started
    timing_metrics["embedding"].update({
        "index_action": "created_index",
        "wall_seconds": sum(
            item["embedding_wall_seconds"]
            for item in timing_metrics["embedding"]["files"]
        ),
        "index_build_wall_seconds": embedding_wall_seconds,
        **summarize_gpu_samples(gpu_samples, embedding_started),
    })

    # Save after full processing
    if vector_store is None:
        raise ValueError(f"No TPC-H rows found in {csv_folder}")
    vector_store.save_local(str(faiss_index_folder))
    manifest_path.write_text(json.dumps(expected_manifest, indent=2), encoding="utf-8")
    print("FAISS vector store created and saved successfully!")

save_timing_metrics()

# Initialize outside per-question timing so tokenizer setup is not charged to
# the first question. A failure is recorded later as an unavailable check.
get_token_count("")


""" Retrieve and Generate part """
# Define prompt for question-answering

''' old prompt'''
class AnswerItem(BaseModel):
    answer: List[str]
    why: List[str] 

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: AnswerItem
def definePrompt():
    prompt = """
        Answer QUESTION_HERE using ONLY the retrieved TPC-H tuples in CONTEXT_HERE.
        Database schema (relationships only, not evidence): SCHEMA_HERE
        For each answer, explain WHY using Witness Sets: minimal sets of input
        tuples that justify that answer. Each answer has one corresponding why string.
        Copy tuple IDs exactly from metadata tuple_id or the <table>_rownum field.
        IDs are supplied by the dataset; never infer them from primary keys or CSV positions.
        A witness containing two joined tuples is "{{supplier_212,nation_3}}".
        Alternative witnesses for one answer are "{{orders_10,customer_14},{orders_20,customer_14}}".
        Use only IDs present in the retrieved context. Do not invent missing rows.
        Return ONLY a JSON object with "answer" and "why" arrays of strings.
        If the context cannot answer the question, return {"answer": [], "why": []}.

        EXAMPLE (illustrative only, never evidence for the actual question):
        Context: tuple_id: region_2; r_regionkey: 1; r_name: AMERICA
        Question: What is the name of the region with region key 1?
        Output: {"answer": ["AMERICA"], "why": ["{{region_2}}"]}
    """
    return prompt

# Step 1: Define Explanation Class: composed by file and row

parser = JsonOutputParser(pydantic_schema=AnswerItem)    
schema_path = os.path.join(repo_folder, "schemaTPCH.txt")
# Load the schema from the file
with open(schema_path, "r") as f:
    schema = f.read().strip()
# Print the schema to verify it has been loaded correctly
print(f"Schema loaded from {schema_path}:\n{schema}\n")
# Define application steps
# Retrieved the most k relevant docs in the vector store, embedding also the question and computing the similarity function

def retrieve(state: State):
    print(f"Retrieving for question: {state['question']}")
    documents_with_scores = vector_store.similarity_search_with_score(
        state["question"], k=timing_metrics["retrieval"]["k"]
    )
    return {
        "context": [document for document, _ in documents_with_scores],
        "retrieved_documents": [
            {
                "rank": rank,
                "score": float(score),
                "metadata": document.metadata,
            }
            for rank, (document, score) in enumerate(documents_with_scores, start=1)
        ],
    }
r''' Disabled legacy ground-truth retrieval implementation.
def get_rows_from_ground_truth(ground_f2: str, csv_folder: str) -> List[Document]:
    """
    Estrae le righe specificate in f2, gestendo Witness Sets multipli e duplicati.
    Supporta anche formati annidati come:
    [
        "{{courses_0,enrollments_0,students_0},{courses_3,enrollments_3,students_0}}",
        "{{courses_0,enrollments_9,students_1}}"
    ]
    """
    documents = []
    seen_entries: Set[str] = set()

    if isinstance(ground_f2, str):
        ground_f2 = [ground_f2]

    # Regex per catturare tutte le occorrenze tipo table_row
    pattern = re.compile(r'(\w+_\d+)')

    for witness_set in ground_f2:
        matches = pattern.findall(witness_set)

        for entry in matches:
            if entry in seen_entries:
                continue
            seen_entries.add(entry)

            try:
                table_name, row_number = entry.rsplit("_", 1)
                row_number = int(row_number)
                csv_path = os.path.join(csv_folder, f"{table_name}.csv")
                #print(table_name, row_number)   
                with open(csv_path, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    header = next(reader)
                    for idx, row in enumerate(reader):
                        if idx == row_number:
                            content = ",".join(row)
                            metadata = {"source": table_name, "row": row_number}
                            documents.append(Document(page_content=content, metadata=metadata))
                            break
            except Exception as e:
                print(f"⚠️ Errore nel parsing di '{entry}': {e}")

    return documents
'''
def tryParseOutput(output_text: str):
    try:
        # Esegui il modello LLM con la catena
        
        # Regex: estrae il primo oggetto JSON, tra ```json ... ``` o solo {}
        json_match = re.search(r"```json\s*([\s\S]*?)\s*```", output_text)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            # Fallback: qualsiasi blocco tra { }
            json_match = re.search(r"\{[\s\S]*?\}", output_text)
            if not json_match:
                json_match = re.search(r"\{\s*\"answer\"\s*:\s*\[.*?\],\s*\"why\"\s*:\s*\[.*?\]\s*\}", output_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0).strip()
            else:
                return None

        # Parse JSON
        parsed_output = json.loads(json_str)

         # Validazione: devono esserci entrambi i campi richiesti
        if not isinstance(parsed_output, dict):
            return None

        if "answer" not in parsed_output:
            return None
        if "why" not in parsed_output:
            return None
        # Validazione finale: tipo corretto dei campi
        if not isinstance(parsed_output["answer"], list):
            return None
        if not isinstance(parsed_output["why"], list):
            return None
        
        return parsed_output
    except Exception as e:
        print(f"Error parsing output: {e}")
        return None
# Generate the answer invoking the LLM with the context joined with the question
def generate(state: State):
  
    print("\n[DEBUG] CONTEXT USED:")
    for doc in state["context"]:
        print(f"- Source: {doc.metadata} \n  Content: {doc.page_content[:300]}...\n")
   
    docs_content = "\n\n".join(str(doc.metadata) + "\n" + doc.page_content for doc in state["context"])
    raw_prompt = definePrompt()
    final_prompt = raw_prompt.replace("QUESTION_HERE", state["question"]).replace("CONTEXT_HERE", docs_content).replace("SCHEMA_HERE", schema)
    original_prompt_tokens = get_token_count(final_prompt)
    prompt_exceeds_context = (
        original_prompt_tokens > LLM_CONTEXT_WINDOW
        if original_prompt_tokens is not None else None
    )

    gpu_samples = []
    gpu_sampler_stop = threading.Event()
    gpu_sampler = None
    generation_started = time.perf_counter()
    if cuda_available:
        gpu_sampler = threading.Thread(
            target=sample_gpu,
            args=(gpu_sampler_stop, gpu_samples),
            daemon=True,
        )
        gpu_sampler.start()

    output_parts = []
    first_token_seconds = None
    response_metadata = {}
    usage_metadata = {}
    generation_error = None
    generation_attempts = 0
    try:
        for attempt in range(1, OLLAMA_MAX_ATTEMPTS + 1):
            generation_attempts = attempt
            # Discard a partial response before retrying so outputs are never duplicated.
            output_parts = []
            response_metadata = {}
            usage_metadata = {}
            try:
                for chunk in llm.stream(final_prompt):
                    if chunk.content:
                        if first_token_seconds is None:
                            first_token_seconds = time.perf_counter() - generation_started
                        output_parts.append(str(chunk.content))
                    if getattr(chunk, "response_metadata", None):
                        response_metadata.update(chunk.response_metadata)
                    if getattr(chunk, "usage_metadata", None):
                        usage_metadata.update(chunk.usage_metadata)
                generation_error = None
                break
            except (ConnectionError, TimeoutError, ValueError) as exc:
                generation_error = str(exc)
                if attempt == OLLAMA_MAX_ATTEMPTS:
                    print(
                        f"Ollama generation failed after {attempt} attempts: {exc}"
                    )
                    break
                delay = OLLAMA_RETRY_DELAY_SECONDS * attempt
                print(
                    f"Ollama generation attempt {attempt}/{OLLAMA_MAX_ATTEMPTS} "
                    f"failed: {exc}. Retrying in {delay:.0f}s..."
                )
                time.sleep(delay)
    finally:
        gpu_sampler_stop.set()
        if gpu_sampler is not None:
            gpu_sampler.join(timeout=3)

    generation_seconds = time.perf_counter() - generation_started
    output_text = "".join(output_parts).strip()
    print(f"\n[DEBUG] LLM RESPONSE:\n{output_text}\n")
    
    
    # Prova a parsare l'output JSON
    if generation_error is not None:
        parsed_output = None
    else:
        try:
            parsed_output = parser.parse(output_text)
        except Exception as e:
            print(f"Error parsing output: {e}")
            parsed_output = None
    return {
        "answer": parsed_output if parsed_output else [],
        "metrics": {
            "parsing_succeeded": parsed_output is not None,
            "generation_attempts": generation_attempts,
            "generation_error": generation_error,
            "time_to_first_token_seconds": first_token_seconds,
            "llm_stream_seconds": generation_seconds,
            "original_prompt_tokens": original_prompt_tokens,
            "ollama_prompt_tokens_processed": response_metadata.get(
                "prompt_eval_count", usage_metadata.get("input_tokens")
            ),
            "output_tokens": response_metadata.get(
                "eval_count", usage_metadata.get("output_tokens")
            ),
            "prompt_exceeds_configured_context": prompt_exceeds_context,
            "prompt_truncation_detected": prompt_exceeds_context,
            "prompt_truncation_check": (
                "matching_model_tokenizer_count_vs_configured_context"
                if original_prompt_tokens is not None
                else "unavailable_tokenizer"
            ),
            "model_load_seconds": (
                response_metadata["load_duration"] / 1_000_000_000
                if response_metadata.get("load_duration") is not None else None
            ),
            "ollama_total_seconds": (
                response_metadata["total_duration"] / 1_000_000_000
                if response_metadata.get("total_duration") is not None else None
            ),
            **summarize_gpu_samples(gpu_samples, generation_started),
        },
        }
    '''
    if parsed_output is None:
        response = llm.invoke(correction_prompt = f"""
                The previous output is not a valid JSON object. Please extract and return only a valid JSON with the following structure:
                 ```json
                    {{
                        "answer": ["<answer_1>", "<answer_2>", ...],
                        "why": ["{{{{<table_name>_<row>}},{{<table_name>_<row>}}}}", "{{{{<table_name>_<row>}}}}", ...]
                    }}
                    ```

                Do not include any text outside the JSON block.
                Here is the previous output:

                {output_text}
                """
            )
        output_text = response.content.strip()
        print(f"\n[DEBUG] LLM RESPONSE:\n{output_text}\n")
    
    
        # Prova a parsare l'output JSON
        parsed_output = tryParseOutput(output_text)
        if parsed_output:
            return parsed_output
        else:
            print("Error: Failed to parse corrected output.")
            return {
                "answer": [],
                "why": []
            }
'''

# Build the graph structure once
#graph_builder = StateGraph(State).add_sequence([retrieve, generate])
#graph_builder.add_edge(START, "retrieve")
#graph = graph_builder.compile()

all_results = []
for i, question in enumerate(questions):
    print(f"Processing question n. {i+1}")

    # Retrieve actual nearest neighbours, including query embedding and FAISS search.
    question_started = time.perf_counter()
    if cuda_available:
        torch.cuda.synchronize()
    retrieval_started = time.perf_counter()
    retrieval_result = retrieve({"question": question})
    if cuda_available:
        torch.cuda.synchronize()
    retrieval_seconds = time.perf_counter() - retrieval_started
    context_docs = retrieval_result["context"]
    
    print(f" Processing question n. {i+1}")
    #full_result = graph.invoke({"question": question})
    
    state = {
        "question": question,
        "context": context_docs
    }
    
    generation_started = time.perf_counter()
    full_result = generate(state)
    generation_seconds = time.perf_counter() - generation_started
    total_seconds = time.perf_counter() - question_started
    generation_metrics = full_result["metrics"]
 
    result = {
        "question": question,
        "question_type": data[question],
        "dataset": "tpch",
        "models": {
            "llm": LLM_MODEL_NAME,
            "embedding": EMBEDDING_MODEL_NAME,
            "embedding_batch_size": EMBEDDING_BATCH_SIZE,
        },
        "answer": full_result.get("answer", []),
        "retrieved_document_count": len(context_docs),
        "retrieved_documents": retrieval_result["retrieved_documents"],
        "parsing_succeeded": generation_metrics["parsing_succeeded"],
        "token_counts": {
            "original_prompt": generation_metrics["original_prompt_tokens"],
            "ollama_prompt_processed": generation_metrics["ollama_prompt_tokens_processed"],
            "output": generation_metrics["output_tokens"],
        },
        "prompt_truncation": {
            "detected": generation_metrics["prompt_truncation_detected"],
            "exceeds_configured_context": generation_metrics["prompt_exceeds_configured_context"],
            "check": generation_metrics["prompt_truncation_check"],
        },
        "generation_request": {
            "sequence_state": "cold_first_request" if i == 0 else "warm_subsequent_request",
            **generation_metrics,
        },
        "timing_seconds": {
            "retrieval": retrieval_seconds,
            "generation": generation_seconds,
            "total": total_seconds,
        },
    }
    all_results.append(result)
    with open(output_filename, "w", encoding="utf-8") as output_file:
        json.dump(all_results, output_file, indent=2, ensure_ascii=False)
    timing_metrics["questions"].append({
        "question_number": i + 1,
        "question": question,
        "retrieved_document_count": len(context_docs),
        "retrieved_documents": retrieval_result["retrieved_documents"],
        "parsing_succeeded": generation_metrics["parsing_succeeded"],
        "retrieval_seconds": retrieval_seconds,
        "generation_seconds": generation_seconds,
        "total_seconds": total_seconds,
        "generation_request": {
            "sequence_state": "cold_first_request" if i == 0 else "warm_subsequent_request",
            **generation_metrics,
        },
    })
    save_timing_metrics()
example_output_txt = output_folder / "example_readable_output.txt"
with open(example_output_txt, "w", encoding="utf-8") as f:
    for idx, result in enumerate(all_results, 1):
        f.write(f"--- Question {idx} ---\n")
        f.write(result["question"] + "\n\n")
        f.write("Answer:\n")
        if isinstance(result["answer"], list):
            f.write(", ".join(result["answer"]))
        else:
            f.write(str(result["answer"]))
        f.write("\n\n")
print(f"Readable example saved to {example_output_txt}")

# Save the results for the current value of k to a JSON file for later analysis
with open(output_filename, "w") as output_file:
    json.dump(all_results, output_file, indent=2, ensure_ascii=False)
    #for i,result in enumerate(all_results,1):
    #    output_file.write(f"----Results {i}---- \n")
    #    output_file.write(f"Question:{result['question']} \n")
    #    output_file.write(f"Answer:{result['answer']} \n")
    #    output_file.write("\n\n")			
print(f"Results saved to {output_filename}")
timing_metrics["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
timing_metrics["question_count"] = len(timing_metrics["questions"])
save_timing_metrics()
print(f"Timing metrics saved to {timing_filename}")
