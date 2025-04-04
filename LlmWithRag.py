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

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "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"


if not os.environ.get("GROQ_API_KEY"):
  os.environ["GROQ_API_KEY"] = "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"

# LLM: Llama3-8b by Groq
llm = init_chat_model("llama3-8b-8192", model_provider="groq")

#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 token hugging-face
# Embedding model: Hugging Face
embedding_model = HuggingFaceEmbeddings(model_name="/home/ciccia/.cache/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2/snapshots/12e86a3c702fc3c50205a8db88f0ec7c0b6b94a0")

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


def save_to_json(result, filename="results.json"):
    """Save results as a properly formatted JSON file"""
    if os.path.exists(filename):
        with open(filename, "r") as f:
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    data = []
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    data.append(result)

    with open(filename, "w") as f:
        json.dump(data, f, indent=4)  # Pretty-print with indentation

# Define prompt for question-answering
prompt = hub.pull("rlm/rag-prompt")

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
# Define application steps
# Retrieved the most k relevant docs in the vector store, embedding also the question and computing the similarity function
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"], k = 10)
    for doc in retrieved_docs:
        print(f"Source: {doc.metadata}\nContent: {doc.page_content}\n")
    return {"context": retrieved_docs}

# Generate the answer invoking the LLM with the context joined with the question
def generate(state: State):
    prompt_with_explanation = f"""
    Question: {state["question"]}
    Given this question and the context provided, provide the answer including the explanation on how you get the information: 
    - the name of the file
    - the row of the file
    (You can find this two information in the metadata of the document you use for the answer.)
    The answer must respect this following structure in a JSON format, without adding more words or sentences before or after:
        {{
        "answer": ["<answer_1>", "<answer_2>", "..."],  
        "explanation": [  
            {{"file": "<file_name>", "row": <row_number>}},  
            {{"file": "<file_name>", "row": <row_number>}}  
            ]  
        }}
    ### IMPORTANT ###
    - The answer must be a list of strings. If there is only one answer, it must still be inside a list.
    - The explanation must be a list of dictionaries, where each dictionary contains the file name and row number from which the answer was extracted.
    - The output must be a valid JSON object with no extra text.

    ### Example ###
        Question: "In which courses is enrolled Giulia Rossi?"
    Expected output:
        {{
        "answer": ["Machine Learning", "Advanced Algorithms"],  
        "explanation": [
            {{"file": "students.csv", "row": 1}},
            {{"file": "enrollments.csv", "row": 1}},
            {{"file": "enrollments.csv", "row": 4}},
            {{"file": "courses.csv", "row": 1}},
            {{"file": "courses.csv", "row": 4}}
        ]
        }}
        
    """
    docs_content = "\n\n".join(str(doc.metadata) + doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": prompt_with_explanation, "context": docs_content})
    
    response = llm.invoke(messages)
    return {"answer": response.content}

# Control flow: compile the application into a single graph object. 
# Connect the retrieval and generation steps into a single sequence.
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# Load question txt
with open("question.txt", "r") as f:
    questions = [line.strip() for line in f.readlines() if line.strip()]
print(questions[2])
for i, question in enumerate(questions):
    print(f"Processing question n. {i+1}")
    result = graph.invoke({"question": question})
    
    save_to_json({
        "question": result["question"],
        "answer": result["answer"]
    })
    
    
    print(result["answer"] + "\n")



