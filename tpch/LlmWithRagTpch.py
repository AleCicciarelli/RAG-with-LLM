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
#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 old old token langsmith
# olt token langsmith  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
if not os.environ.get("GROQ_API_KEY"):
  os.environ["GROQ_API_KEY"] = "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"

# LLM: Llama3-8b by Groq
llm = init_chat_model("llama3-70b-8192", model_provider="groq", temperature = 0)
# MISTRAL by Groq
#llm = init_chat_model("mistral-saba-24b", model_provider="groq", temperature = 0)
#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 token hugging-face
#llm = ChatOllama(model="llama3:70b", temperature=0)
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
output_filename = f"outputs_llama70b/cleaned/outputs_llama70b_groq_bm25.json"


documents = []
all_files = [f for f in os.listdir(csv_folder) if f.endswith(".csv")]

for file in all_files:
    file_path = os.path.join(csv_folder, file)
    loader = CSVLoader(file_path=file_path)
    docs = loader.load()
    documents.extend(docs)
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 10 


""" Retrieve and Generate part """
# Define prompt for question-answering
prompt = PromptTemplate.from_template("""
    Question: {question}

    Your task is to:
    Provide the correct answer(s) based only on the context. The answer MUST be only the value required like in the example below, don't add explanation or extra text.
    For each answer, explain WHY it appears using **Witness Sets**: minimal sets of input tuples that justify the result.

    Each Witness Set must be a string like:
        "{{{{<table_name>_<row>}}}}"
    (use `source`(return just the table_name which correspond to the name of the file,WITHOUT extension and path csv_data/TABLENAME.csv
) and `row` metadata from the context).

    If an answer has multiple Witness Sets, list each one in the `"why"` array "{{{{WitnessSet1}}, {{WitnessSet2}}}}". A result is valid if at least one Witness Set supports it.

    Return the output as a **stringified JSON array**, with NO extra text: (avoid the comments, or the additional information), using this format template:

    [
        {{
        "answer": ["<answer_1>","<answer_2>"],
        "why": [
        "{{{{table_row_a, table_row_b}}, {{table_row_c, table_row_d}}}}",  
        "{{{{table_row_e, table_row_f}}}}"   
        ]
        }}
    ]

    Example:

    CONTEXT:
    - source: customer.csv , row: 14322
    (<col_a>:<val_a>,..., c_nationkey : 2, ...)
    - source: orders.csv, row: 137
    (o_orderkey : 546, ..., o_totalprice : 20531.43, ...)
    - source: customer.csv, row: 101
    (<col_a>:<val_c>, ...,<c_nationkey : 2, ...)
    - source: orders.csv, row: 78528
    (o_orderkey : 314052, ..., o_totalprice : 20548.82, ...)

    QUESTION:
        "Which orders (o_orderkey) done by a customer with nationkey = 2 have a total price between 20500 and 20550?"

    EXPECTED ANSWER:

    [  
        {{
            "answer": [
            {{
                "answer": [ "546","314052" ],
                "why": [
                "{{{{customer_14322,orders_137}}}}", 
                "{{{{customer_101,orders_78528}}}}"
                ]
            }}
            
        }}           
    ]
"""
)
# Step 1: Define Explanation Class: composed by file and row

class AnswerItem(BaseModel):
    answer: List[str]
    why: List[str]

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: List[AnswerItem]
   
parser = JsonOutputParser(pydantic_schema=AnswerItem)    
 
# Define application steps
# Retrieved the most k relevant docs in the vector store, embedding also the question and computing the similarity function
def retrieve(state: State):
   
    retrieved_docs = bm25_retriever.get_relevant_documents(state["question"])
    #for doc in retrieved_docs:
    #    print(f"Source: {doc.metadata}\nContent: {doc.page_content}\n")
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
