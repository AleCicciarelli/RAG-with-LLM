import os
import re
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langgraph.graph import START, StateGraph, END
from typing_extensions import List, TypedDict, Optional, Dict, Any, Set
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


# LLM: Llama3-8b by Groq
#llm = init_chat_model("llama3-70b-8192", model_provider="groq", temperature = 0)
# MISTRAL by Groq
#llm = init_chat_model("mistral-saba-24b", model_provider="groq", temperature = 0)
#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 token hugging-face
llm = ChatOllama(model="llama3:70b", temperature=0)
# Embedding model: Hugging Face
#embedding_model = HuggingFaceEmbeddings(model_name="/home/ciccia/.cache/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2/snapshots/12e86a3c702fc3c50205a8db88f0ec7c0b6b94a0")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

csv_folder = "tpch/csv_data"
faiss_index_folder = "tpch/faiss_index"
output_filename = f"tpch/outputs_llama70b/iterative/outputs_llama70b_ollama_iterative_k10.json"
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

""" Retrieve and Generate part """
# Define prompt for question-answering
#prompt = hub.pull("rlm/rag-prompt")
def definePrompt():
    prompt = """
        Your task is to provide the correct answer(s) to this question: QUESTION_HERE, based ONLY on the given context: CONTEXT_HERE.
        For each answer, explain WHY it appears using **Witness Sets**: minimal sets of input tuples that justify the result.
        Format of Witness Sets (as strings):  
        - If there is ONE relevant tuple set: "{{<table_name>_<row>}}"  
        - If there are MULTIPLE: "{{<table_name>_<row>},{<table_name>_<row>},...}}"  
        IMPORTANT:
        Return ONLY the JSON output, with no explanation, no introductory sentence, and no trailing comments.
        If your output is not a valid JSON block in the format described, it will be discarded.
        If the answer is not present in the context, return an empty array.
        
        
        INVALID OUTPUT EXAMPLE (will be discarded):
        The answer is: {"answer": [...], "why": [...]}
        VALID OUTPUT EXAMPLE (will be accepted):
        ```json
        {
            "answer": ["<answer_1>", "<answer_2>", ...],
            "why": ["{{<table_name>_<row>},{<table_name>_<row>}}", "{{<table_name>_<row>}}", ...]
        }
        ```
            
         EXAMPLE 1:
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

        EXPECTED OUTPUT:
        ```json
        {
            "answer": ["546", "314052"],
            "why": [
                "{{customer_14322,orders_137}}",
                "{{customer_101,orders_78528}}"
            ]
            
        }
        ```
        EXAMPLE 2:    
        CONTEXT:
            - source: suppliers.csv, row: 4
            (..s_name: "Supplier#000000005",...,s_phone: "21-151-690-3663")
           

        QUESTION:  
            "What is the phone number of the supplier named 'Supplier#000000005'?"

        EXPECTED OUTPUT:
        ```json
        {
            "answer": ["21-151-690-3663"],
            "why": [
                "{{supplier_4}}"
            ]
        }
        ```
    """
    return prompt

def get_k_to_add(previous_answer: str,) -> int:

    seen_entries: Set[str] = set()
    k = 0
    if isinstance(previous_answer, str):
        previous_answer = [previous_answer["why"]] if previous_answer else []  

    # Regex per catturare tutte le occorrenze tipo table_row
    pattern = re.compile(r'(\w+_\d+)')

    for witness_set in previous_answer:
        matches = pattern.findall(witness_set)

        for entry in matches:
            if entry in seen_entries:
                continue
            k += 1

    print(f"Number of unique entries in previous answer: {k}")    
    return k
class AnswerItem(BaseModel):
    answer: List[str]
    why: List[str]
# Define state for application
class State(TypedDict):
    original_question: str # To keep track of the initial question
    current_question: str  # The question used for retrieval in the current iteration
    context: List[Document]
    answer: AnswerItem
    #iteration_history: List[Dict[str, Any]] # To store previous answers and contexts for iterative refinement

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
    docs_content = "\n".join([f"- source: {doc.metadata.get('source').split('/')[-1].replace('.csv', '')} , row: {doc.metadata.get('row')}\n({doc.page_content})" for doc in state["context"]])

    #docs_content = "\n\n".join(str(doc.metadata) + "\n" + doc.page_content for doc in state["context"])
    raw_prompt = definePrompt()
    final_prompt = raw_prompt.replace("QUESTION_HERE", state["current_question"]).replace("CONTEXT_HERE", docs_content)
    response = llm.invoke(final_prompt)
    output_text = response.content.strip()
    print(f"\n[DEBUG] LLM RESPONSE:\n{output_text}\n")
    # Save to debug log
    with open(debug_log_filename, "a", encoding="utf-8") as debug_file:
        debug_file.write(f"\n=== Question {i+1}: {current_state['original_question']} ===\n")
        debug_file.write(f"--- Iteration {iter_num + 1} ---\n")
        #debug_file.write(f"\nPrompt Sent to LLM:\n{final_prompt.format(question=state['current_question'], context=docs_content)}\n")
        debug_file.write(f"\nLLM Response:\n{response}\n")
        debug_file.write("="*80 + "\n")
    # Print the response from the LLM
    print(f"LLM response: {response}")
   
    try:
        parsed= parser.parse(output_text)
          
        print(f"Previous answer generated: {parsed if parsed else response.strip()}")
    except Exception as e:
        print(f"Errore nel parsing: {e}")
        parsed = None

    return {
        "answer": parsed if parsed else response.strip(),
        "current_question": (
            f"{state['original_question']}\n\n"
            f"Previous answer generated: {json.dumps(parsed, indent = 2) if parsed else response.strip()}\n"
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
}

    # Run the graph until it decides to stop (or a max iteration limit)
    max_iterations = 5
    current_state = initial_state
    for iter_num in range(max_iterations):
        print(f"\n--- Iteration {iter_num + 1} for question n. {i+1} ---")
        k = k + get_k_to_add(current_state["answer"]["why"]) if current_state["answer"] else 10
        current_state["k"] = k
        print(f"Current k value: {k}")
        full_result = graph.invoke(current_state)
        # Set the current question for the next iteration composed by original question and the last answer
        #if full_result.get("answer"):
            # If the answer is a list, take the first item for the next question
            #if isinstance(full_result["answer"], list) and full_result["answer"]:
                #current_state["current_question"] = f"{current_state['original_question']} {full_result['answer']}"
            #else:
                #current_state["current_question"] = current_state["original_question"]
        #else:
            #current_state["current_question"] = current_state["original_question"]
        # Update current_state for the next iteration
        current_state.update(full_result)
        print("after current state update")
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
