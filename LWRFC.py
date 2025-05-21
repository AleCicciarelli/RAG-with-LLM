import os
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict, Union, Set
from langchain_core.documents import Document
from langchain import hub
import json
import re
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel
from typing import List, Dict
import csv
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "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"
#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 olt token langsmith

if not os.environ.get("GROQ_API_KEY"):
  os.environ["GROQ_API_KEY"] = "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"
#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 previous token groq
# LLM: Llama3-8b by Groq
#llm = init_chat_model("llama3-70b-8192", model_provider="groq", temperature = 0)
# MISTRAL by Groq
llm = init_chat_model("mistral-saba-24b", model_provider="groq", temperature = 0)

# Prendi le variabili di ambiente model_name = os.environ["OLLAMA_MODEL"] output_file = os.environ["OUTPUT_FILE"]
# Inizializza l'LLM
#llm = ChatOllama(model="llama3:70b", temperature=0)
# Ollama LLM
#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 token hugging-face
# Embedding model: Hugging Face
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

""" Indexing part """

csv_folder = "csv_data"
faiss_index_folder = "faiss_index"


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
    answer: AnswerItem

parser = JsonOutputParser(pydantic_schema=AnswerItem)    
from typing import List
import os
import csv

class Document:
    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata

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
    
    # Creare il contesto per i documenti (contenuto e metadati)
    docs_content = "\n\n".join(str(doc.metadata) + "\n" + doc.page_content for doc in state["context"])
    
    prompt_with_explanation = prompt_with_explanation = f"""
        Question: {state["question"]}
        Context: {docs_content}

        Your task is to:
        1. Provide the correct answer(s) based only on the context.
        2. For each answer, explain WHY it appears using **Witness Sets**: minimal sets of input tuples that justify the result.

        Each Witness Set must be a string like:
            "{{{{<table_name>_<row>, <table_name>_<row>, ...}}}}"
        (use `source` and `row` metadata from the context).

        If an answer has multiple Witness Sets, list each one in the `"why"` array "{{{{WitnessSet1}}, {{WitnessSet2}}}}". A result is valid if at least one Witness Set supports it.

        Return the output as a **stringified JSON array**, with no extra text:

        [
            {{
            "answer": ["<answer_1>","<answer_2>"],
            "why": [
            "{{{{table_row_a, table_row_b}}}}",  //answer1
            "{{{{table_row_c, table_row_d}}}}",  //answer1
            "{{{{table_row_e, table_row_f}}}}"    //answer2
            ]
            }}
        ]

        Example:

        CONTEXT:
            - source: courses.csv, row: 0  
            (course_id:101, course_name:Machine Learning, ...)  
            - source: courses.csv, row: 3  
            (course_id:104, course_name:Advanced Algorithms, ...)  
            - source: enrollments.csv, row: 0  
            (enrollment_id:1, student_id:1, course_id:101, ...)  
            - source: enrollments.csv, row: 3  
            (enrollment_id:4, student_id:1, course_id:104, ...)  
            - source: enrollments.csv, row: 9  
            (enrollment_id:10, student_id:2, course_id:101, ...)  
            - source: students.csv, row: 0  
            (student_id:1, name:Giulia, surname:Rossi, ...)  
            - source: students.csv, row: 1  
            (student_id:2, name:Marco, surname:Bianchi, ...)  

        QUESTION:  
            "Which are the students (specify name and surname) enrolled in Machine Learning or in Advanced Algorithm courses?"

        EXPECTED ANSWER:

        [
            {{
            "answer": ["Giulia Rossi","Marco Bianchi"],
            "why": [
            "{{{{courses_0,enrollments_0,students_0}}}}", 
            "{{{{courses_3,enrollments_3,students_0}}}}",  
            "{{{{courses_0,enrollments_9,students_1}}}}"   
            ]
            }}
 
        ]
"""


    # Preparare il messaggio finale per l'LLM
    messages = prompt.invoke({"question": prompt_with_explanation, "context": docs_content})
    
    # Pulire e analizzare la risposta dell'LLM
    #cleaned_response = response.content.strip()
    #print(cleaned_response)
    response = llm.invoke(messages)
    
    try:
        parsed = parser.parse(response.content)
    except Exception as e:
        print(f"Errore nel parsing: {e}")
        parsed = None

    return {
        "answer": parsed if parsed else response.content.strip()
    }
'''
f2 = [
    "{{courses_0,enrollments_0,students_0},{courses_3,enrollments_3,students_0}}",
    "{{courses_0,enrollments_9,students_1}}"
]
'''

for l in range(1, 4):  # Da 1 a 3
    print(f"Esecuzione {l}...")
    # Leggi le domande da un file di testo
    with open("question.txt", "r") as f:
        questions = [line.strip() for line in f.readlines() if line.strip()]

    # Inizializza una lista per i risultati
    all_results = []
    # carica il ground truth
    with open("ground_truth.json", "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    # Loop per invocare LLM su tutte le domande
    for i, question in enumerate(questions):
        print(f"Processing question n. {i+1}")
        # Step 1: Ottieni la risposta attesa (facoltativo, puoi usarla per confronto)
        gt = ground_truth[i]
        #gt_answer = gt["f1"]
        gt_source_info = gt["why"]  # "{{students_0,exams_3}}"
        
        # Step 2: Costruisci contesto perfetto a partire dalle righe vere
        context_docs = get_rows_from_ground_truth(gt_source_info, csv_folder="csv_data")
        
        # Step 3: Costruisci manualmente lo stato
        state = {
            "question": question,
            "context": context_docs
        }

        # Step 4: Esegui solo il passo di generazione
        full_result = generate(state)
        
        result = {
            "question": question,
            "answer": full_result.get("answer", "")
        }
        # Aggiungere il risultato alla lista
        all_results.append(result)


    output_filename = f"output_FC_mistral24b_{l}.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    print(f"Risultati salvati in {output_filename}\n")


