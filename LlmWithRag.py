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

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "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"


if not os.environ.get("GROQ_API_KEY"):
  os.environ["GROQ_API_KEY"] = "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"

# LLM: Llama3-8b by Groq
llm = init_chat_model("llama3-70b-8192", model_provider="groq", temperature = 0)

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
class ExplanationItem(BaseModel):
    file: str
    row: int

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: List[str]
    explanation: List[ExplanationItem]
    
# Define application steps
# Retrieved the most k relevant docs in the vector store, embedding also the question and computing the similarity function
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"], k = 17)
    #for doc in retrieved_docs:
    #    print(f"Source: {doc.metadata}\nContent: {doc.page_content}\n")
    return {"context": retrieved_docs}

# Generate the answer invoking the LLM with the context joined with the question
def generate(state: State):
    prompt_with_explanation = f"""
    Question: {state["question"]}
    Given this question and the context provided , provide the answer including the explanation on how you get the information: 
    - the name of the file
    - the row of the file
    (You can find this two information in the metadata of the document you use for the answer.)
    The answer must respect the following structure, but return it as a string representation of a JSON:

    {{
        "answer": ["<answer_1>", "<answer_2>", "..."],  
        "explanation": [  
            {{  "file": "<file_name>",
                "row": <row_number>}},  
            {{  "file": "<file_name>", 
                "row": <row_number>}}  
        ]
    }}

    ### IMPORTANT ###

    - The output must be a valid JSON object, without extra text.

    ### Example ###
    {{
        "answer": [
            "Computer Science"
        ],
        "explanation": [
            {{
                "file": "teachers.csv",
                "row": 1
            }}
        ]}}
        
    """

    docs_content = "\n\n".join(str(doc.metadata) + "\n" + doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": prompt_with_explanation, "context": docs_content})
    response = llm.invoke(messages)
    #print(response.content)
    cleaned_response = response.content.strip()
    #print(cleaned_response)
    '''Parse generated answer in a JSON format'''
    try:
        parsed_response = json.loads(cleaned_response)
    except json.JSONDecodeError as e:
        print(f"Error parsing response: {cleaned_response}")
        print(f"JSONDecodeError: {str(e)}")
        return {"answer": [], "explanation": []}
    print(parsed_response)
   
    return {
        "answer": parsed_response.get("answer", []),
        "explanation": parsed_response.get("explanation", [])
    }
    
'''k analysis'''
'''
import time

results_by_k = {}
# Process questions from a txt file
with open("question.txt", "r") as f:
    questions = [line.strip() for line in f.readlines() if line.strip()]
    
for k in range(12, 21):  # k da 0 a 20
    print(f"\n=== Running evaluation with k={k} ===")
    all_results = []

    def retrieve(state: State):
        retrieved_docs = vector_store.similarity_search(state["question"], k=k)
        return {"context": retrieved_docs}

    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    for i, question in enumerate(questions):
        print(f"[k={k}] Processing question n. {i+1}")
        full_result = graph.invoke({"question": question})
        result = {
            "question": question,
            "answer": full_result.get("answer", []),
            "explanation": full_result.get("explanation", [])
        }
        all_results.append(result)

    output_filename = f"outputs_k_{k}.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    results_by_k[k] = output_filename
    time.sleep(1)  # opzionale: per evitare throttling dell'API

''' 
# Control flow: Compile the application into a graph
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# Process questions from a txt file
with open("question.txt", "r") as f:
    questions = [line.strip() for line in f.readlines() if line.strip()]

all_results = []

''' Loop for LLM invocation on questions '''

for i, question in enumerate(questions):
    print(f"Processing question n. {i+1}")
    
    full_result = graph.invoke({"question": question})
    result = {
        "question": question,
        "answer": full_result.get("answer", []),
        "explanation": full_result.get("explanation", [])
    }
    
    all_results.append(result)
# Save results to json file
with open("all_outputs_k76_llama70b.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=4, ensure_ascii=False)
