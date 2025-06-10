import os
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict, Set
from langchain_core.documents import Document
from langchain import hub
import json
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List
import csv
from langchain_community.chat_models import ChatOllama
import re
os.environ["LANGSMITH_TRACING"] = "true" 
os.environ["LANGSMITH_API_KEY"] = "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"
#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 old old token langsmith
# olt token langsmith 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
if not os.environ.get("GROQ_API_KEY"):
  os.environ["GROQ_API_KEY"] = "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"

# LLM: Llama3-8b by Groq
#llm = init_chat_model("llama3-8b-8192", model_provider="groq", temperature = 0)
# MISTRAL by Groq
llm = init_chat_model("mistral-saba-24b", model_provider="groq", temperature = 0)
#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 token hugging-face
#llm = ChatOllama(model="llama3:8b", temperature=0)
# Embedding model: Hugging Face
#embedding_model = HuggingFaceEmbeddings(model_name="/home/ciccia/.cache/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2/snapshots/12e86a3c702fc3c50205a8db88f0ec7c0b6b94a0")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

""" Indexing part """

csv_folder = "csv_data_tpch"


""" Retrieve and Generate part """
# Define prompt for question-answering
prompt = hub.pull("rlm/rag-prompt")
# Step 1: Define Explanation Class: composed by file and row

class AnswerItem(BaseModel):
    answer: List[str]
    why: List[str]

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: List[AnswerItem]
    k: int
   
parser = JsonOutputParser(pydantic_schema=AnswerItem)    
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

# Generate the answer invoking the LLM with the context joined with the question
def generate(state: State):
    prompt_with_explanation = f"""
        Question: {state["question"]}

        Your task is to:
        1. Provide the correct answer(s) based only on the context.
        2. For each answer, explain WHY it appears using **Witness Sets**: minimal sets of input tuples that justify the result.

        Each Witness Set must be a string like:
            "{{{{<table_name>_<row>}}}}"
        (use `source`(return just the table_name which correspond to the name of the file,WITHOUT extension and path csv_data/TABLENAME.csv
) and `row` metadata from the context).

        If an answer has multiple Witness Sets, list each one in the `"why"` array "{{{{WitnessSet1}}, {{WitnessSet2}}}}". A result is valid if at least one Witness Set supports it.

        Return the output as a **stringified JSON array**, with NO extra text: (avoid the comments, or the additional information)

        [
            {{
            "answer": ["<answer_1>","<answer_2>"],
            "why": [
            "{{{{table_row_a, table_row_b}}}}",  
            "{{{{table_row_e, table_row_f}}}}"    
            }}
        ]

        Example:

        CONTEXT:
        - source: <table_1>, row: <row_idx_1>
        (<col_a>:<val_a>, <col_b>:<val_b>, ...)
        - source: <table_1>, row: <row_idx_2>
        (<col_a>:<val_a1>, <col_b>:<val_b1>, ...)
        - source: <table_2>, row: <row_idx_3>
        (<col_c>:<val_c>, <col_d>:<val_d>, ...)
        - source: <table_2>, row: <row_idx_4>
        (<col_c>:<val_c1>, <col_d>:<val_d1>, ...)
        - source: <table_3>, row: <row_idx_1>
        (<col_e>:<val_e>, <col_f>:<val_f>, ...)
        - source: <table_3>, row: <row_idx_2>
        (<col_e>:<val_e1>, <col_f>:<val_f1>, ...)

        QUESTION:
            "Which are the <entity_type> (specify <col_a> and <col_b>) involved in <condition_1> OR <condition_2>?"

        EXPECTED ANSWER:

        [
            {{
                "answer": ["<col_a_val> <col_b_val>", "<col_a_val> <col_b_val>"],
                "why": [
                    "{{{{<table_1>_<row_idx_1>,<table_2>_<row_idx_3>,<table_3>_<row_idx_1>}},{{<table_1>_<row_idx_2>,<table_2>_<row_idx_4>,<table_3>_<row_idx_1>}}}}", 
                    "{{{{<table_1>_<row_idx_1>,<table_2>_<row_idx_4>,<table_3>_<row_idx_2>}}}}"
                ]
            }}
        

        ]
"""


    docs_content = "\n\n".join(str(doc.metadata) + "\n" + doc.page_content for doc in state["context"])

    messages = prompt.invoke({"question": prompt_with_explanation, "context": docs_content})
    response = llm.invoke(messages)
     
    try:
        parsed = parser.parse(response.content)
    except Exception as e:
        print(f"Errore nel parsing: {e}")
        parsed = None

    return {
        "answer": parsed if parsed else response.content.strip()
    }

# Create a dictionary to store results for each k
results_by_k = {}
# Leggi le domande dal file JSON
with open("tpch/questions.json", "r") as f:
    data = json.load(f)
    questions = list(data.keys())
# Ora 'questions' contiene solo le domande (le chiavi del dizionario)
for q in questions:
    print(q)
with open("tpch/ground_truthTpch.json", "r", encoding="utf-8") as f:
    ground_truth = json.load(f)   

all_results = []

for i, question in enumerate(questions):
    print(f"Processing question n. {i+1}")
    gt = ground_truth[i]
    gt_source_info = gt["why"]
    
    # Step 2: Costruisci contesto perfetto a partire dalle righe vere
    context_docs = get_rows_from_ground_truth(gt_source_info, csv_folder="tpch/csv_data_tpch")
    
    # Step 3: Costruisci manualmente lo stato
    state = {
        "question": question,
        "context": context_docs
    }

    full_result = generate(state)
    result = {
        "question": question,
        "answer": full_result.get("answer", ""),
    }
    all_results.append(result)

output_filename = f"tpch/outputs_mixtral8x7b/full_context/outputs_mixtral8x7bGroq.json"
# Save the results for the current value of k to a JSON file for later analysis
with open(output_filename, "w") as output_file:
    json.dump(all_results, output_file, indent=4, ensure_ascii=False)
print(f"Results saved to {output_filename}")
