import os
import re
import csv
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
from planGenerator import generate_plan

os.environ["LANGSMITH_TRACING"] = "false" 
os.environ["LANGSMITH_API_KEY"] = "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"
# LLM model to use, in this case the provider is Ollama local 
llm = ChatOllama(model="llama3:70b", temperature=0)
# Embedding model: Hugging Face
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
# Define the folder paths for CSV data and FAISS index
csv_folder = "csv_data"
faiss_index_folder = "faiss_index"
output_filename = f"iterativeRAG/outputs_llama70b/iterative_FC_5rounds.json"
debug_log_filename = f"iterativeRag/debug_log_llama70b_iterative.txt"
# Ensure the output and debug log directories exist
os.makedirs(os.path.dirname(debug_log_filename), exist_ok=True)
os.makedirs(os.path.dirname(output_filename), exist_ok=True)

"""Uploading the database schema for the plan generation"""
schema_path = "schemaTPCH.txt"
# Load the schema from the file
with open(schema_path, "r") as f:
    schema = f.read().strip()
# Print the schema to verify it has been loaded correctly
print(f"Schema loaded from {schema_path}:\n{schema}\n")
""" Indexing part """

# Verify if the FAISS files already exist
if os.path.exists(faiss_index_folder):
    # Load the FAISS index folder (allow_dangerous_deserialization=True just because we create the files and so we can trust them)
    vector_store = FAISS.load_local(faiss_index_folder, embedding_model, allow_dangerous_deserialization=True)
    print("FAISS index successfully loaded")
else:
    batch_size = 200 
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

def definePrompt():
    prompt = """
    Your task is to provide the correct answer(s) to this step: STEP_HERE, 
    which is part of answering the overall question: QUESTION_HERE, , based ONLY on the given context: CONTEXT_HERE.
        IMPORTANT:

        - Do NOT include introductory phrases, explanations or any dots at the end.
        - If the answer is not present in the context, return an empty array.
        - Return the answer strictly in the following JSON format:

        ```json
        {
            "answer": ["<answer_1>", "<answer_2>", ...]
        }
        ```
        EXAMPLE 1:
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
        ```json
            {
            "answer": ["Giulia Rossi","Marco Bianchi"],
            }
        ```
        
        EXAMPLE 2:    
        CONTEXT:
            - source: departments.csv, row: 1
            (department_id:2, department_name:Electronics, faculty:Engineering)
           

        QUESTION:  
            "Which faculty does the Electronics department belong to?"

        EXPECTED OUTPUT:
        ```json
        {
            "answer": ["Engineering"]
        }
        ```

"""

    return prompt
# Define the AnswerItem model to parse the output from the LLM
class AnswerItem(BaseModel):
    answer: List[str]
    why: List[str]
# Define state for application
class State(TypedDict):
    original_question: str 
    current_question: str 
    context: List[Document]
    k: int
    step: str
    answer: AnswerItem
# Define the output parser to parse the LLM response into the AnswerItem model
parser = JsonOutputParser(pydantic_schema=AnswerItem)

# Define application steps
# Retrieved the most k relevant docs in the vector store, embedding also the question and computing the similarity(L2 distance) function

def retrieve(state: State):
    print(f"Retrieving for question: {state['original_question']}")
    retrieved_docs = vector_store.similarity_search(state["current_question"], k = 10)
    return {"context": retrieved_docs}
# Generate the answer invoking the LLM with the context joined with the question
def generate(state: State):
    # Construct a detailed prompt for the LLM
   
    docs_content = "\n\n".join(str(doc.metadata) + "\n" + doc.page_content for doc in state["context"])
    raw_prompt = definePrompt()
    final_prompt = raw_prompt.replace("STEP_HERE", state["step"]).replace("QUESTION_HERE", state["current_question"]).replace("CONTEXT_HERE", docs_content)
    response = llm.invoke(final_prompt)
    output_text = response.content.strip()
    print(f"\n[DEBUG] LLM RESPONSE:\n{output_text}\n")

   
    # Print the response from the LLM
    print(f"LLM response: {response}")
   
    try:
        parsed= parser.parse(output_text)
          
        print(f"Previous answer generated: {parsed if parsed else response.content.strip()}")
    except Exception as e:
        print(f"Errore nel parsing: {e}")
        parsed = AnswerItem(answer=[], why=[])

    return {
        "answer": parsed,
        "current_question": (
            f"{state['original_question']}\n\n"
            f"Previous answer generated: {json.dumps(parsed, indent = 2) if parsed else response.content.strip()}\n"
            f"Based on the original question, the previous answer, find the correct answer(s) to the question."

        )

    }    

# Read questions from the JSON file
with open("questions.json", "r") as f:
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
    print(f"\n=== Plan generation===")
    plan = generate_plan(question, schema, llm)
    
    step_results = []
    previous_answer = None
    initial_state = {
    "original_question": question,
    "current_question": question, # Start with the original question
    "k": 10,
    "context": [], # Initial empty context
    "answer": {
            "answer": [],
            "why": []
        },  # Initial empty answer
}
   
    current_state = initial_state
    for step_idx, step in enumerate(plan):
        print(f"\n--- Step {step_idx + 1}/{len(plan)} ---")
        current_state["step"] = step
        full_result = graph.invoke(current_state)
        # Set the current question for the next iteration composed by original question and the last answer
        answer_obj = full_result.get("answer")
        if answer_obj and isinstance(answer_obj, AnswerItem):
            if answer_obj.answer:
                current_state["current_question"] = f"{current_state['original_question']} {' '.join(answer_obj.answer)}"
            else:
                current_state["current_question"] = current_state["original_question"]
        else:
            current_state["current_question"] = current_state["original_question"]
        # Update current_state for the next iteration
        current_state.update(full_result)
        print("after current state update")
        # Store the final result of the iterations for this question
        all_final_results.append({
            "question": current_state["original_question"],
            "iteration": step_idx + 1,
            "answer": current_state["answer"], # The last answer generated
        })


with open(output_filename, "w", encoding='utf-8') as output_file:
    json.dump(all_final_results, output_file, indent=4, ensure_ascii=False)
print(f"Results saved to {output_filename}")
