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
import faiss 
import re
from langchain_community.chat_models import ChatOllama
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "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"


if not os.environ.get("GROQ_API_KEY"):
  os.environ["GROQ_API_KEY"] = "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"
#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 previous token groq
# LLM: Llama3-8b by Groq
#llm = init_chat_model("llama3-70b-8192", model_provider="groq", temperature = 0)
# Ollama LLM
llm = ChatOllama(model="llama3-70b-8192", temperature=0)
#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 token hugging-face
# Embedding model: Hugging Face
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
class ExplanationItem(BaseModel):
    file: str
    row: int

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    
# Define application steps
# Retrieved the most k relevant docs in the vector store, embedding also the question and computing the similarity function
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"], k = 76)
    #for doc in retrieved_docs:
    #    print(f"Source: {doc.metadata}\nContent: {doc.page_content}\n")
    return {"context": retrieved_docs}

# Generate the answer invoking the LLM with the context joined with the question
def generate(state: State):
    prompt_with_explanation = f"""
   You are an assistant that answers questions over structured data, the data came from tabular data and some tables are related
   with foreign keys(same attributes in more tables). For example the field "id" in students.csv is binded with "student_id" field in enrollments.csv,
   exams.csv, grades.csv, thesis.csv. So if "id = 1, name = Giulia, surname = Rossi,.." in students.csv, and "thesis_id=1, title=Deep Learning for Image Recognition, student_id=1, ..."
   in thesis.csv; it means that Giulia Rossi is doing the thesis with id 1, title= Deep Learning for Image Recognition,..
   
    You must not only answer the question, but for each result item, explain **WHY** it appears in the result.
    This explanation must be in terms of **WITNESSES SET**: the minimal sets of input tuples that justify the result.
    IMPORTANT:
        - **Inside** each witness set: tuples are combined using **AND** (all are required).
        - **Between** different witness sets: use **OR** (any one is enough).
    This means: 
        > Result X is in the output if (WitnessSet1) OR (WitnessSet2) OR ...

        And each WitnessSet has form:
        > {{Tuple1 AND Tuple2 AND ...}}
   

    Answer in the following format:
    - Result: <Result Value>
    WHY: {{ <witness 1> }}, {{ <witness 2> }}, ...

    Only include witnesses that are minimal and sufficient for deriving the result.
    Group equivalent witnesses together if possible.

    EXAMPLE(CONTEXT DATA):
        Input Data:
        courses.csv(course_id,course_name,department,credits,teacher,semester)
        101,Machine Learning,Computer Science,6,Carlo Rossi,Fall
        102,Database Systems,Computer Science,6,Laura Bianchi,Spring
        
        exams.csv(exam_id,course_id,date,time,classroom_id)
        1,101,2023-01-10,09:00,1
        2,102,2023-02-15,14:00,2
        
        QUESTION: "When is the Machine Learning exam?"

      - Result: 2023-01-10,09:00  
        WHY: 
            {{ courses(course_id:101, course_name:Machine Learning) AND exams(exam_id:1,course_id:101,date:2023-01-10,time:09:00) }}
    When referencing tuples from the input, use a readable format like:

    <table>(row=<index>): attribute1=value1, attribute2=value2, ...
    Avoid internal object representations (e.g., Document(...)). Focus on clarity.
    
    Data:
    {state["context"]}

    Question:
    {state["question"]}
    """

    # Creare il contesto per i documenti (contenuto e metadati)
    docs_content = "\n\n".join(str(doc.metadata) + "\n" + doc.page_content for doc in state["context"])

    # Preparare il messaggio finale per l'LLM
    messages = prompt.invoke({"question": prompt_with_explanation, "context": docs_content})
    
    # Eseguire la chiamata all'LLM
    response = llm.invoke(messages)
    
    # Pulire e analizzare la risposta dell'LLM
    cleaned_response = response.content.strip()
    print(cleaned_response)


    # Restituiamo la risposta intera come testo (rispetta il nuovo formato con Result/WHY)
    return {
        "answer": cleaned_response
    }


# Control flow: Compile the application into a graph
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# Leggi le domande da un file di testo
with open("question.txt", "r") as f:
    questions = [line.strip() for line in f.readlines() if line.strip()]

# Inizializza una lista per i risultati
all_results = []

# Loop per invocare LLM su tutte le domande
for i, question in enumerate(questions):
    print(f"Processing question n. {i+1}")
    
    # Eseguire l'invocazione del grafo
    full_result = graph.invoke({"question": question})
    
    result = {
        "question": question,
        "answer": full_result.get("answer", "")
    }
    # Aggiungere il risultato alla lista
    all_results.append(result)

with open("all_outputs_WHY_FOREIGNKEYS2_k76_llama70B.txt", "w", encoding="utf-8") as f:
    for result in all_results:
        f.write(f"Question: {result['question']}\n")
        f.write(result["answer"].strip() + "\n")
        f.write("\n" + "="*60 + "\n\n")