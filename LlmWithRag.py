import os
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langchain import hub
import json
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List
import time

os.environ["LANGSMITH_TRACING"] = "true" 
os.environ["LANGSMITH_API_KEY"] = "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"
#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 old old token langsmith
#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 olt token langsmith
if not os.environ.get("GROQ_API_KEY"):
  os.environ["GROQ_API_KEY"] = "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"

# LLM: Llama3-8b by Groq
llm = init_chat_model("llama3-8b-8192", model_provider="groq", temperature = 0)
# MISTRAL by Groq
#llm = init_chat_model("mistral-saba-24b", model_provider="groq", temperature = 0)
#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 token hugging-face
# Embedding model: Hugging Face
#embedding_model = HuggingFaceEmbeddings(model_name="/home/ciccia/.cache/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2/snapshots/12e86a3c702fc3c50205a8db88f0ec7c0b6b94a0")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

""" Indexing part """

csv_folder = "csv_data"
faiss_index_folder = "faiss_index"

# Verify if the FAISS files already exist
if os.path.exists(faiss_index_folder):
    # Load the FAISS index folder ( allow_dangerous_deserialization=True just because we create the files and so we can trust them)
    vector_store = FAISS.load_local(faiss_index_folder, embedding_model, allow_dangerous_deserialization=True)
    print("FAISS index successfully loaded")
else:
    # if don't exist, load the csv files
    documents = []
    for file in os.listdir(csv_folder):
        if file.endswith(".csv"):
            file_path = os.path.join(csv_folder, file)
            loader = CSVLoader(file_path=file_path)
            docs = loader.load()
            documents.extend(docs)

    print(f"Loaded {len(documents)} documents from {len(os.listdir(csv_folder))} CSV files.")

    # Create vector store with the embedding model 
    # (if we want other similarity strategies: distance_strategy = DistanceStrategy.COSINE, the default is L2 distance)
    vector_store = FAISS.from_documents(documents=documents, embedding=embedding_model)

    # Save FAISS vector store 
    vector_store.save_local(faiss_index_folder)
    print("FAISS vector store created and saved successfully!")

""" Retrieve and Generate part """
# Define prompt for question-answering
prompt = hub.pull("rlm/rag-prompt")
# Step 1: Define Explanation Class: composed by file and row
class AnswerItem(BaseModel):
    file: str
    row: int

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: List[AnswerItem]
    k: int
   
parser = JsonOutputParser(pydantic_schema=AnswerItem)    
 
# Define application steps
# Retrieved the most k relevant docs in the vector store, embedding also the question and computing the similarity function
def retrieve(state: State):
   
    retrieved_docs = vector_store.similarity_search(state["question"], k = state["k"])
    #for doc in retrieved_docs:
    #    print(f"Source: {doc.metadata}\nContent: {doc.page_content}\n")
    return {"context": retrieved_docs}

# Generate the answer invoking the LLM with the context joined with the question
def generate(state: State):
    prompt_with_explanation = f"""
        You are a question-answering system. Given a question and a context, you must provide the answer(s) to the question.
        Question: {state["question"]}
        Your task is to:
        1. Provide the correct answer(s) based only on the context.
        2. For each answer, explain WHY it appears using **Witness Sets**: minimal sets of input tuples that justify the result.

        Each Witness Set must be a string like:
            "{{{{<table_name>_<row>}}}}"
        (use `source` and `row` metadata from the context).

        If an answer has multiple Witness Sets, list each one in the `"why"` array "{{{{WitnessSet1}}, {{WitnessSet2}}}}". A result is valid if at least one Witness Set supports it.

        Return the output as a **stringified JSON array**, with no extra text:

        [
            {{
            "answer": ["<answer_1>","<answer_2>"],
            "why": [
            "{{{{table_row_a, table_row_b}}, {{table_row_c, table_row_d}}}}",   //answer1        
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
            "{{{{courses_0,enrollments_0,students_0}},{{courses_3,enrollments_3,students_0}}}}",
            "{{{{courses_0,enrollments_9,students_1}}}}"
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
'''k analysis'''
# Create a dictionary to store results for each k
results_by_k = {}
# Process questions from a txt file
with open("question.txt", "r") as f:
    questions = [line.strip() for line in f.readlines() if line.strip()]
    
# Build the graph structure once
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

for k in range(19, 70):  # k da 0 a 20
    print(f"\n=== Running evaluation with k={k} ===")
    all_results = []

    for i, question in enumerate(questions):
        print(f"[k={k}] Processing question n. {i+1}")
        full_result = graph.invoke({"question": question, "k": k})
        result = {
            "question": question,
            "answer": full_result.get("answer", []),
        }
        all_results.append(result)

    output_filename = f"outputs_llama8b/outputs_k_{k}_llama8b.json"
    # Save the results for the current value of k to a JSON file for later analysis
    with open(output_filename, "w") as output_file:
        json.dump(all_results, output_file, indent=4, ensure_ascii=False)
    print(f"Results for k={k} saved to {output_filename}")
