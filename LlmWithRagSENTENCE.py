import os
import pandas as pd
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

import re


os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "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"

if not os.environ.get("GROQ_API_KEY"):
  os.environ["GROQ_API_KEY"] = "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"

# LLM: Llama3-8b by Groq
llm = init_chat_model("llama3-8b-8192", model_provider="groq", temperature = 0)

folder_path = "./csv_data"
faiss_db_path = "./faiss_db"
descriptions = []

# Template for CSV files
TEMPLATES = {
    "classrooms.csv": lambda row: f'The classroom with id: "{row["classroom_id"]}", is in the building: "{row["building"]}", it has the number: "{row["room_number"]}" and a capacity of: "{row["capacity"]}".',
    "departments.csv": lambda row: f'The department with id: "{row["department_id"]}" is named "{row["department_name"]}" and is part of the faculty "{row["faculty"]}".',
    "exams.csv": lambda row: f'The exam with id: "{row["exam_id"]}" is for course "{row["course_id"]}", scheduled on "{row["date"]}" at "{row["time"]}" in classroom "{row["classroom_id"]}".',
    "thesis.csv": lambda row: f'Thesis with id: "{row["thesis_id"]}" titled "{row["title"]}" was written by student "{row["student_id"]}", supervised by teacher "{row["teacher_id"]}", in the academic year "{row["academic_year"]}", with status "{row["status"]}" and research area "{row["research_area"]}".',
    "enrollments.csv": lambda row: f'Enrollment with id: "{row["enrollment_id"]}" refers to student with student_id: "{row["student_id"]}" enrolled in course with course:id: "{row["course_id"]}" on "{row["enrollment_date"]}".',
    "students.csv": lambda row: f'Student with student_id"{row["id"]}": with name {row["name"]} and surname {row["surname"]}, born on "{row["birth_date"]}", nationality "{row["nationality"]}", gender "{row["gender"]}", enrolled on "{row["enrollment_date"]}", email: {row["email"]}.',
    "grades.csv": lambda row: f'The grade with grade_id: "{row["grade_id"]}", is the grade obtained by the student with student_id "{row["student_id"]}" in the course with course_id: "{row["course_id"]}" and is: "{row["grade"]}".',
    "teachers.csv": lambda row: f'Teacher with teacher_id:"{row["teacher_id"]}": with name {row["name"]} and surname {row["surname"]}, part of department with department_id"{row["department_id"]}", email: {row["email"]}.',
    "teacher_research_areas.csv": lambda row: f'Teacher with teacher_id:"{row["teacher_id"]}" works in the research area "{row["research_area"]}" and is involved in "{row["number_of_project"]}" projects.',
    "courses.csv": lambda row: f'Course with course_id:"{row["course_id"]}" named "{row["course_name"]}" is part of department named "{row["department"]}", taught by teacher named "{row["teacher"]}", worth "{row["credits"]}" credits, offered in semester "{row["semester"]}".'
}
# Embedding model
#embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
embedding_model = HuggingFaceEmbeddings(model_name="/home/ciccia/.cache/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2/snapshots/12e86a3c702fc3c50205a8db88f0ec7c0b6b94a0")
''' Indexing Part '''

if os.path.exists(faiss_db_path):
     # Load the FAISS index folder ( allow_dangerous_deserialization=True just because we create the files and so we can trust them)
    vector_store = FAISS.load_local(faiss_db_path, embedding_model, allow_dangerous_deserialization=True)
    print("FAISS index successfully loaded")
else:
# Sentences from CSV 
    for file in os.listdir(folder_path):
        if file.endswith(".csv") and file in TEMPLATES:
            filepath = os.path.join(folder_path, file)
            df = pd.read_csv(filepath)
            for _, row in df.iterrows():
                try:
                    text = TEMPLATES[file](row)
                    metadata = {"source": file}
                    descriptions.append(Document(page_content=text, metadata=metadata))
                except Exception as e:
                    print(f"Errore nel file {file} su riga: {row} -> {e}")
    # Vector store creation
    vector_store = FAISS.from_documents(descriptions, embedding_model)

    # Save vectore store locally
    vector_store.save_local("faiss_db")
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
    retrieved_docs = vector_store.similarity_search(state["question"], k = 10)
    #for doc in retrieved_docs:
    #    print(f"Source: {doc.metadata}\nContent: {doc.page_content}\n")
    return {"context": retrieved_docs}

# Generate the answer invoking the LLM with the context joined with the question
def generate(state: State):
    prompt_with_explanation = f"""
    Question: {state["question"]}
    Given this question and the context provided, provide the answer including the explanation on how you get the information: 
    - the name of the file
    - the row of the file
    (You can find this two information in the metadata of the document you use for the answer.)
    The answer must respect the following structure, but return it as a string representation of a JSON:
    In the context you find some sentences, explaining the 
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
# Control flow: Compile the application into a graph
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# Process questions from a txt file
with open("question.txt", "r") as f:
    questions = [line.strip() for line in f.readlines() if line.strip()]

all_results = []

''' Loop for LLM invocation on questions'''

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
with open("all_outputs_SENTENCE.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=4, ensure_ascii=False)
    
'''
# Test query
query = "What is the grade obtained by Sophie Durand in Embedded Systems?"
docs = vector_store.similarity_search(query, k=20)

for doc in docs:
    print(doc.page_content + str(doc.metadata))
'''