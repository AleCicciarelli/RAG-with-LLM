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
from langchain_community.retrievers import BM25Retriever
from langchain import hub
import json
import csv
import re
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

csv_folder = "tpch/csv_data_tpch"
faiss_index_folder = "tpch/faiss_index"
output_filename = f"tpch/outputs_llama70b/iterative/outputs_llama70b_nowhy_FC_3rounds.json"

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

        - Do NOT include introductory phrases, explanations or any dots at the end.
        - If the answer is not present in the context, return an empty array.
        - Return the answer strictly in the following JSON format:

        ```json
        {{
            "answer": ["<answer_1>", "<answer_2>", ...]
        }}
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
        {{
            "answer": ["546", "314052"]
        }}
        ```
        EXAMPLE 2:    
        CONTEXT:
            - source: suppliers.csv, row: 4
            (..s_name: "Supplier#000000005",...,s_phone: "21-151-690-3663")
           

        QUESTION:  
            "What is the phone number of the supplier named 'Supplier#000000005'?"

        EXPECTED OUTPUT:
        ```json
        {{
            "answer": ["21-151-690-3663"]
        }}
        ```

"""
)
# Step 1: Define Explanation Class: composed by file and row

class AnswerItem(BaseModel):
    answer: str

# Define state for application
#class State(TypedDict):
#    question: str
#    context: List[Document]
#    answer: List[AnswerItem]
class State(TypedDict):
    original_question: str # To keep track of the initial question
    current_question: str  # The question used for retrieval in the current iteration
    context: List[Document]
    answer: AnswerItem   
parser = JsonOutputParser(pydantic_schema=AnswerItem)    

# Define application steps
# Retrieved the most k relevant docs in the vector store, embedding also the question and computing the similarity function
'''
def retrieve(state: State):
    print(f"Retrieving for question: {state['question']}")
    retrieved_docs = vector_store.similarity_search(state["question"], k = 10)
    return {"context": retrieved_docs}
'''
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

   
    print("\n[DEBUG] CONTEXT USED:")
    for doc in state["context"]:
        print(f"- Source: {doc.metadata} \n  Content: {doc.page_content[:300]}...\n")
   
    docs_content = "\n\n".join(str(doc.metadata) + "\n" + doc.page_content for doc in state["context"])
    chain = LLMChain(
        llm=llm,
        prompt = prompt 
    )
    response = chain.run({
            "question": state["question"], 
            "context": docs_content
        })

    if not response:
        raise ValueError("Empty response from LLM.")

    output_text = response.strip()
    print(f"\n[DEBUG] RAW LLM RESPONSE:\n{output_text}\n")

    try:
    # Esegui il modello LLM con la catena
        
        # Regex: estrae il primo oggetto JSON, tra ```json ... ``` o solo {}
        json_match = re.search(r"```json\s*([\s\S]*?)\s*```", output_text)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            # Fallback: qualsiasi blocco tra { }
            json_match = re.search(r"\{[\s\S]*?\}", output_text)
            if not json_match:
                raise ValueError("No valid JSON found in LLM response.")
            json_str = json_match.group(0).strip()

        # Parse JSON
        parsed_output = json.loads(json_str)

        if not isinstance(parsed_output, dict) or "answer" not in parsed_output:
            raise ValueError("Invalid JSON format. Missing 'answer'.")

        return {
            "answer": parsed_output["answer"]
        }

    except (json.JSONDecodeError, ValueError) as e:
        print(f"⚠️ Errore nel parsing del JSON: {e}")
        return {
            "answer": []
        }
# Read questions from the JSON file
with open("tpch/questions.json", "r") as f:
    data = json.load(f)
    questions = list(data.keys())
'''
# Build the graph structure once
workflow = StateGraph(State)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
graph = workflow.compile()
'''
with open("tpch/ground_truthTpch.json", "r", encoding="utf-8") as f:
    ground_truth = json.load(f)  
all_final_results = []

# Iterate over each question and invoke the graph to get the answer
for i, question in enumerate(questions):
    print(f"\n=== Running evaluation for question n. {i+1}: {question} ===")
    #initial_state = {
    #"original_question": question,
    #"current_question": question, # Start with the original question
    #"k": 10,
    #"context": [], # Initial empty context
    #"answer": [], # Initial empty answer
#}

    gt = ground_truth[i]
    gt_source_info = gt["why"]
    
    # Step 2: Costruisci contesto perfetto a partire dalle righe vere
    context_docs = get_rows_from_ground_truth(gt_source_info, csv_folder="tpch/csv_data_tpch")
    
    
    state = {
        "original_question": question,
        "current_question": question, # Start with the original question
        "k": 10,    
        "context": context_docs,
        "answer": [],
    }

    # Run the graph until it decides to stop (or a max iteration limit)
    max_iterations = 5
    current_state = state
    for iter_num in range(max_iterations):
        print(f"\n--- Iteration {iter_num + 1} for question n. {i+1} ---")
        #k = k + get_k_to_add(current_state["answer"]["why"]) if current_state["answer"] else 10
        #current_state["k"] = k
        #print(f"Current k value: {k}")
        full_result = generate(state)
        
        #full_result = graph.invoke(current_state)
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
'''
# Leggi le domande dal file JSON
with open("tpch/questions.json", "r") as f:
    data = json.load(f)
    questions = list(data.keys())

# Build the graph structure once
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

all_results = []
with open("tpch/ground_truthTpch.json", "r", encoding="utf-8") as f:
    ground_truth = json.load(f)   
for i, question in enumerate(questions):
    print(f"Processing question n. {i+1}")
    gt = ground_truth[i]
    gt_source_info = gt["why"]
    
    # Step 2: Costruisci contesto perfetto a partire dalle righe vere
    context_docs = get_rows_from_ground_truth(gt_source_info, csv_folder="tpch/csv_data_tpch")
    
    print(f" Processing question n. {i+1}")
#    full_result = graph.invoke({"question": question})
    
    state = {
        "question": question,
        "context": context_docs
    }
    
    full_result = generate(state)
 
    result = {
        "question": question,
        "answer": full_result.get("answer", []),
    }
    all_results.append(result)

# Save the results for the current value of k to a JSON file for later analysis
with open(output_filename, "w") as output_file:
    json.dump(all_results, output_file, indent=2, ensure_ascii=False)
    #for i,result in enumerate(all_results,1):
    #    output_file.write(f"----Results {i}---- \n")
    #    output_file.write(f"Question:{result['question']} \n")
    #    output_file.write(f"Answer:{result['answer']} \n")
    #    output_file.write("\n\n")			
print(f"Results saved to {output_filename}")
'''