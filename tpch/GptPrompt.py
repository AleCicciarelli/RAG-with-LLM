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
from langchain_community.chat_models import ChatOllama

os.environ["LANGSMITH_TRACING"] = "true" 
os.environ["LANGSMITH_API_KEY"] =  "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"
#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 old old token langsmith
# olt token langsmith 
if not os.environ.get("GROQ_API_KEY"):
  os.environ["GROQ_API_KEY"] = "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"

# LLM: Llama3-8b by Groq
#llm = init_chat_model("llama3-8b-8192", model_provider="groq", temperature = 0)
# MISTRAL by Groq
#llm = init_chat_model("mistral-saba-24b", model_provider="groq", temperature = 0)
#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 token hugging-face
llm = ChatOllama(model="llama3:70b", temperature=0)
# Embedding model: Hugging Face
#embedding_model = HuggingFaceEmbeddings(model_name="/home/ciccia/.cache/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2/snapshots/12e86a3c702fc3c50205a8db88f0ec7c0b6b94a0")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

""" Indexing part """

csv_folder = "csv_data_tpch"
faiss_index_folder = "faiss_index"

# Verify if the FAISS files already exist
if os.path.exists(faiss_index_folder):
    # Load the FAISS index folder ( allow_dangerous_deserialization=True just because we create the files and so we can trust them)
    vector_store = FAISS.load_local(faiss_index_folder, embedding_model, allow_dangerous_deserialization=True)
    print("FAISS index successfully loaded")
else:
    batch_size = 200  # Adjust as needed
    documents = []
    all_files = [f for f in os.listdir(csv_folder) if f.endswith(".csv")]

    for file in all_files:
        file_path = os.path.join(csv_folder, file)
        loader = CSVLoader(file_path=file_path)
        docs = loader.load()
        
        for i in range(0, len(docs), batch_size):
            batch_docs = docs[i:i+batch_size]
            if not documents:
                vector_store = FAISS.from_documents(batch_docs, embedding=embedding_model)
            else:
                vector_store.add_documents(batch_docs)

    # Save after full processing
    vector_store.save_local(faiss_index_folder)
    print("FAISS vector store created and saved successfully!")
""" Retrieve and Generate part """
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
        You are an AI agent answering questions based on a CSV dataset.

        You must output a valid JSON array in this format ONLY:

        [
        {{
            "answer": ["<value1>", "<value2>"],
            "why": ["{{{{<table>_<row>}}}}", "{{{{<table>_<row>}}}}"]
        }}
        ]

        Instructions:
        - Use only the data in the context.
        - The `answer` list must contain values that satisfy the question.
        - Each item in `why` must refer to the source and row number: `{{{{table_row}}}}` (from metadata).
        - If multiple rows justify an answer, include multiple references in `why`.

        Example:

        Context:
        - source: customer.csv , row: 14322
        - source: orders.csv, row: 137
        - source: customer.csv, row: 101
        - source: orders.csv, row: 78528

        Question: Which orders (o_orderkey) done by a customer with nationkey = 2 have a total price between 20500 and 20550?

        Answer:
        [
        {{
            "answer": ["546", "314052"],
            "why": ["{{{{customer_14322,orders_137}}}}", "{{{{customer_101,orders_78528}}}}"]
        }}
        ]
    """

    docs_content = "\n\n".join(str(doc.metadata) + "\n" + doc.page_content for doc in state["context"])
    messages = [
    {"role": "system", "content": "You are an expert data extraction agent working with relational CSVs."},
    {"role": "user", "content": prompt_with_explanation + "\n\nContext:\n" + docs_content}]
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
with open("questions.json", "r") as f:
    data = json.load(f)
    questions = list(data.keys())
# Ora 'questions' contiene solo le domande (le chiavi del dizionario)
for q in questions:
    print(q)
    
# Build the graph structure once
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

for k in range(10, 45):  # k da 0 a 20
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

    output_filename = f"outputs_llama70b/gpt_prompt/outputs_k_{k}_llama8b.json"
    # Save the results for the current value of k to a JSON file for later analysis
    with open(output_filename, "w") as output_file:
        json.dump(all_results, output_file, indent=4, ensure_ascii=False)
    print(f"Results for k={k} saved to {output_filename}")
