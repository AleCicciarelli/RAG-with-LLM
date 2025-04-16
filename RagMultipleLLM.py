import os
import json
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain import hub
from pydantic import BaseModel
from typing import List, TypedDict
from langchain_core.documents import Document

# API keys
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "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"
if not os.environ.get("GROQ_API_KEY"):
  os.environ["GROQ_API_KEY"] = "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"

# Modelli Groq da testare
groq_models = {
    "llama3-8b-8192": "llama8B",
    "llama3-70b-8192": "llama70B",
    "mistral-7b-32768": "mistral",
    "mixtral-8x7b-32768": "mixtral",
    "gemma-7b-it": "gemma"
}

# Embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Indexing
csv_folder = "csv_data"
faiss_index_folder = "faiss_index"

if os.path.exists(faiss_index_folder):
    vector_store = FAISS.load_local(faiss_index_folder, embedding_model, allow_dangerous_deserialization=True)
    print("✅ FAISS index loaded")
else:
    documents = []
    for file in os.listdir(csv_folder):
        if file.endswith(".csv"):
            file_path = os.path.join(csv_folder, file)
            loader = CSVLoader(file_path=file_path)
            docs = loader.load()
            documents.extend(docs)
    vector_store = FAISS.from_documents(documents=documents, embedding=embedding_model)
    vector_store.save_local(faiss_index_folder)
    print("✅ FAISS index created and saved")

# Prompt template
prompt = hub.pull("rlm/rag-prompt")

# ExplanationItem
class ExplanationItem(BaseModel):
    file: str
    row: int

# State
class State(TypedDict):
    question: str
    context: List[Document]
    answer: List[str]
    explanation: List[ExplanationItem]

# Retrieval
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"], k=10)
    return {"context": retrieved_docs}

# Generation
def generate(llm, state: State):
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
    """

    docs_content = "\n\n".join(str(doc.metadata) + "\n" + doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": prompt_with_explanation, "context": docs_content})
    response = llm.invoke(messages)
    cleaned_response = response.content.strip()

    try:
        parsed_response = json.loads(cleaned_response)
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON error: {e}")
        print(f"Raw output: {cleaned_response}")
        return {"answer": [], "explanation": []}

    return {
        "answer": parsed_response.get("answer", []),
        "explanation": parsed_response.get("explanation", [])
    }

# Carica le domande
with open("question.txt", "r") as f:
    questions = [line.strip() for line in f.readlines() if line.strip()]

# Loop su ciascun modello
for model_id, model_shortname in groq_models.items():
    print(f"\n🚀 Running model: {model_id}")
    llm = init_chat_model(model_id, model_provider="groq", temperature=0)
    all_results = []

    for i, question in enumerate(questions):
        print(f"  → Question {i+1}/{len(questions)}")
        state = {"question": question}
        state.update(retrieve(state))
        result = generate(llm, state)

        all_results.append({
            "question": question,
            "answer": result["answer"],
            "explanation": result["explanation"]
        })

    # Salvataggio risultati
    output_file = f"outputs_{model_shortname}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
    print(f"✅ Saved: {output_file}")
