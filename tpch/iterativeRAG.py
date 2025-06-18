import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langgraph.graph import START, StateGraph, END
from typing_extensions import List, TypedDict, Optional, Dict, Any
from langchain_core.documents import Document
import json
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List
import time
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.chat_models import ChatOllama

os.environ["LANGSMITH_TRACING"] = "false" 
os.environ["LANGSMITH_API_KEY"] = "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"
#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 old old token langsmith
# olt token langsmith  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
if not os.environ.get("GROQ_API_KEY"):
 os.environ["GROQ_API_KEY"] = "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"

# LLM: Llama3-8b by Groq
#llm = init_chat_model("llama3-70b-8192", model_provider="groq", temperature = 0)
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

csv_folder = "tpch/csv_data"
faiss_index_folder = "tpch/faiss_index"
output_filename = f"tpch/outputs_llama70b/iterative/outputs_llama70b_ollama_iterativeK10_faiss.json"
debug_log_filename = f"iterativeRag/debug_log_llama70b_iterative.txt"
os.makedirs(os.path.dirname(debug_log_filename), exist_ok=True)
# Save the results for the current value of k to a JSON file for later analysis
os.makedirs(os.path.dirname(output_filename), exist_ok=True)

""" Indexing part """

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

documents = []
all_files = [f for f in os.listdir(csv_folder) if f.endswith(".csv")]
'''
for file in all_files:
    file_path = os.path.join(csv_folder, file)
    loader = CSVLoader(file_path=file_path)
    docs = loader.load()
    documents.extend(docs)
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 10 
'''
""" Retrieve and Generate part """
# Define prompt for question-answering
#prompt = hub.pull("rlm/rag-prompt")
prompt = PromptTemplate.from_template("""
        Your task is:

        1. Provide the correct answer(s) to this question: {question}, based ONLY on the given context: {context}.

        2. For each answer, explain WHY it appears using **Witness Sets**: minimal sets of input tuples that justify the result.

        - Each Witness Set must be formatted as:
        "{{{{<table_name>_<row>}}}}"
        where:
            - <table_name> is the name of the table used, from the `source` metadata field in the context.
            - <row> is the corresponding `row` metadata value.

        - If an answer has multiple Witness Sets, list each set separately inside the `"why"` array as a string, e.g.:
        "{{{{WitnessSet1}}, {{WitnessSet2}}}}".

        - A result is valid if at least one Witness Set supports it.

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
                "why": [
                    "{{{{<table_1>_<row_idx_1>,<table_2>_<row_idx_3>,<table_3>_<row_idx_1>}},{{<table_1>_<row_idx_2>,<table_2>_<row_idx_4>,<table_3>_<row_idx_1>}}}}", 
                    "{{{{<table_1>_<row_idx_1>,<table_2>_<row_idx_4>,<table_3>_<row_idx_2>}}}}"
                ]
            }}
"""
)
# Step 1: Define Explanation Class: composed by file and row
'''
class WitnessSet(BaseModel):
    tables_rows: List[str] = Field(description="List of table_row strings, e.g., ['customer_14322', 'orders_137']")

class AnswerItem(BaseModel):
    answer: List[str] = Field(description="The final answer(s) to the question.")
    why: List[str] = Field(description="List of string of witness sets justifying the answer. Each witness set is a list of table_row strings.")
'''  
class AnswerItem(BaseModel):
    answer: List[str]
    why: List[str]
# Define state for application
class State(TypedDict):
    original_question: str # To keep track of the initial question
    current_question: str  # The question used for retrieval in the current iteration
    context: List[Document]
    answer: List[AnswerItem]
    iteration_history: List[Dict[str, Any]] # To store previous answers and contexts for iterative refinement

parser = JsonOutputParser(pydantic_schema=AnswerItem)

# Define application steps
# Retrieved the most k relevant docs in the vector store, embedding also the question and computing the similarity function
def retrieve(state: State):
    print(f"Retrieving for question: {state['original_question']}")
    retrieved_docs = vector_store.similarity_search(state["current_question"], k = 10)
    return {"context": retrieved_docs}
    '''
    print(f"Retrieving for question: {state['original_question']}")

        retrieved_docs = bm25_retriever.get_relevant_documents(state["current_question"])
    return {"context": retrieved_docs}
'''
# Generate the answer invoking the LLM with the context joined with the question
def generate(state: State):
    # Construct a detailed prompt for the LLM
    context_str = "\n".join([f"- source: {doc.metadata.get('source').split('/')[-1].replace('.csv', '')} , row: {doc.metadata.get('row')}\n({doc.page_content})" for doc in state["context"]])

    chain = LLMChain(
        llm=llm,
        prompt = prompt 
    )
    response = chain.run({
    "question": state["current_question"], 
    "context": context_str
    })
    # Print the prompt received by the llm with the context and question
    print(f"Prompt sent to LLM:\n{prompt.format(question=state['current_question'], context=context_str)}")
    # Save to debug log
    with open(debug_log_filename, "a", encoding="utf-8") as debug_file:
        debug_file.write(f"\n=== Question {i+1}: {current_state['original_question']} ===\n")
        debug_file.write(f"--- Iteration {iter_num + 1} ---\n")
        debug_file.write(f"\nPrompt Sent to LLM:\n{prompt.format(question=state['current_question'], context=context_str)}\n")
        debug_file.write(f"\nLLM Response:\n{response}\n")
        debug_file.write("="*80 + "\n")
    # Print the response from the LLM
    print(f"LLM response: {response}")
   
    try:
        parsed= parser.parse(response)
          
        print(f"Previous answer generated: {parsed if parsed else response.strip()}")
    except Exception as e:
        print(f"Errore nel parsing: {e}")
        parsed = None

    return {
        "answer": parsed if parsed else response.strip(),
        "current_question": (
            f"{state['original_question']}\n\n"
            f"Previous answer generated: {parsed if parsed else response.strip()}\n"
            f"Based on the original question, the previous answer, find the correct answer(s) to the question."

        )

    }    

# Read questions from the JSON file
with open("tpch/questions.json", "r") as f:
    data = json.load(f)
    questions = list(data.keys())

# Build the graph structure once
workflow = StateGraph(State)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
graph = workflow.compile()

all_final_results = []
# Iterate over each question and invoke the graph to get the answer
for i, question in enumerate(questions):
    print(f"\n=== Running evaluation for question n. {i+1}: {question} ===")
    initial_state = {
        "original_question": question,
        "current_question": question, # Start with the original question
        "k": 10,
        "context": [], # Initial empty context
        "answer": [], # Initial empty answer
        "iteration_history": []
    }

    # Run the graph until it decides to stop (or a max iteration limit)
    max_iterations = 3
    current_state = initial_state
    for iter_num in range(max_iterations):
        print(f"\n--- Iteration {iter_num + 1} for question n. {i+1} ---")
        full_result = graph.invoke(current_state)
        # Set the current question for the next iteration composed by original question and the last answer
        if full_result.get("answer"):
            # If the answer is a list, take the first item for the next question
            if isinstance(full_result["answer"], list) and full_result["answer"]:
                current_state["current_question"] = f"{current_state['original_question']} {full_result['answer']}"
            else:
                current_state["current_question"] = current_state["original_question"]
        else:
            current_state["current_question"] = current_state["original_question"]
        # Update current_state for the next iteration
        current_state.update(full_result)

        # Store the final result of the iterations for this question
        all_final_results.append({
            "question": current_state["original_question"],
            "iteration": iter_num + 1,
            "answer": current_state["answer"], # The last answer generated
            #"full_iteration_history": current_state["iteration_history"] # All intermediate steps
        })


with open(output_filename, "w", encoding='utf-8') as output_file:
    json.dump(all_final_results, output_file, indent=4, ensure_ascii=False)
print(f"Results saved to {output_filename}")
