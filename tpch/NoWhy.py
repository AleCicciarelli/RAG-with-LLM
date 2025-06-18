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
from langchain_community.retrievers import BM25Retriever
from langchain import hub
import json
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List
import time
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOllama

os.environ["LANGSMITH_TRACING"] = "false" 
os.environ["LANGSMITH_API_KEY"] = "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"


# MISTRAL by Groq
#llm = init_chat_model("mistral-saba-24b", model_provider="groq", temperature = 0)
#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 token hugging-face
llm = ChatOllama(model="llama3:70b", temperature=0)
# Embedding model: Hugging Face
#embedding_model = HuggingFaceEmbeddings(model_name="/home/ciccia/.cache/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2/snapshots/12e86a3c702fc3c50205a8db88f0ec7c0b6b94a0")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
#embedding_model = HuggingFaceEmbeddings(
#    model_name="BAAI/bge-small-en-v1.5",
#    model_kwargs={"device": "cuda"},  
#    encode_kwargs={"normalize_embeddings": True}
#)
""" Indexing part """

csv_folder = "csv_data_tpch"
faiss_index_folder = "faiss_index"
output_filename = f"outputs_llama70b/no_why/outputs_llama70b_nowhy.json"

# Verify if the FAISS files already exist
if os.path.exists(faiss_index_folder):
    # Load the FAISS index folder (allow_dangerous_deserialization=True just because we create the files and so we can trust them)
    vector_store = FAISS.load_local(faiss_index_folder, embedding_model, allow_dangerous_deserialization=True)
    print("FAISS index successfully loaded")
else:
    batch_size = 200  # Adjust as needed
    documents = []
    all_files = [f for f in os.listdir(csv_folder) if f.endswith(".csv")]

    # Initialize vector_store before the loop
    vector_store = None

    for file in all_files:
        file_path = os.path.join(csv_folder, file)
        loader = CSVLoader(file_path=file_path)
        docs = loader.load()
        for i in range(0, len(docs), batch_size):
            batch_docs = docs[i:i+batch_size]
            if vector_store is None: # Only create for the first batch
                vector_store = FAISS.from_documents(batch_docs, embedding=embedding_model)
            else:
                vector_store.add_documents(batch_docs)

    # Save after full processing
    vector_store.save_local(faiss_index_folder)
    print("FAISS vector store created and saved successfully!")


""" Retrieve and Generate part """
# Define prompt for question-answering
prompt = PromptTemplate.from_template("""
    Your task is to provide the correct answer(s) to this question: {question}, based ONLY on the given context: {context}.
        IMPORTANT:

        - Return ONLY a valid JSON array, with NO explanations, comments, or extra text.
        - Do NOT include introductory phrases.
        - Format your response exactly as in the example below.

        EXAMPLE:
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

        EXPECTED RESPONSE:
            {{
                "answer": ["<col_a_val> <col_b_val>", "<col_a_val> <col_b_val>"],
            }}
"""
)
# Step 1: Define Explanation Class: composed by file and row

class AnswerItem(BaseModel):
    answer: List[str]

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: List[AnswerItem]
   
parser = JsonOutputParser(pydantic_schema=AnswerItem)    
 
# Define application steps
# Retrieved the most k relevant docs in the vector store, embedding also the question and computing the similarity function
def retrieve(state: State):
    print(f"Retrieving for question: {state['original_question']}")
    retrieved_docs = vector_store.similarity_search(state["current_question"], k = 10)
    return {"context": retrieved_docs}

# Generate the answer invoking the LLM with the context joined with the question
def generate(state: State):

    '''
    print("\n[DEBUG] CONTEXT USED:")
    for doc in state["context"]:
        print(f"- Source: {doc.metadata} \n  Content: {doc.page_content[:300]}...\n")
    '''
    docs_content = "\n\n".join(str(doc.metadata) + "\n" + doc.page_content for doc in state["context"])
    chain = LLMChain(
        llm=llm,
        prompt = prompt 
    )
    response = chain.run({
    "question": state["question"], 
    "context": docs_content
    })
    print(f"\n[DEBUG] LLM RESPONSE:\n{response}\n")
    try:
        parsed = parser.parse(response)
    except Exception as e:
        print(f"Errore nel parsing: {e}")
        parsed = None

    return {
        "answer": parsed if parsed else response.strip()
    }


# Leggi le domande dal file JSON
with open("questions.json", "r") as f:
    data = json.load(f)
    questions = list(data.keys())

# Build the graph structure once
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

all_results = []

for i, question in enumerate(questions):
    print(f" Processing question n. {i+1}")
    full_result = graph.invoke({"question": question})
    result = {
        "question": question,
        "answer": full_result.get("answer", []),
    }
    all_results.append(result)

# Save the results for the current value of k to a JSON file for later analysis
with open(output_filename, "w") as output_file:
    json.dump(all_results, output_file, indent=4, ensure_ascii=False)
print(f"Results saved to {output_filename}")
