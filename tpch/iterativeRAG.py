import os
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langgraph.graph import START, StateGraph, END
from typing_extensions import List, TypedDict, Optional, Dict, Any
from langchain_core.documents import Document
from langchain import hub
import json
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List
import time
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.chat_models import ChatOllama

os.environ["LANGSMITH_TRACING"] = "false" 
os.environ["LANGSMITH_API_KEY"] = "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"
#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 old old token langsmith
# olt token langsmith  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
if not os.environ.get("GROQ_API_KEY"):
  os.environ["GROQ_API_KEY"] = "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"

# LLM: Llama3-8b by Groq
llm = init_chat_model("llama3-8b-8192", model_provider="groq", temperature = 0)
# MISTRAL by Groq
#llm = init_chat_model("mistral-saba-24b", model_provider="groq", temperature = 0)
#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 token hugging-face
#llm = ChatOllama(model="llama3:70b", temperature=0)
# Embedding model: Hugging Face
#embedding_model = HuggingFaceEmbeddings(model_name="/home/ciccia/.cache/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2/snapshots/12e86a3c702fc3c50205a8db88f0ec7c0b6b94a0")
#embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cuda"},  
    encode_kwargs={"normalize_embeddings": True}
)

""" Indexing part """
csv_folder = "tpch/csv_data_tpch"
faiss_index_folder = "tpch/faiss_index"

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

# Step 1: Define Explanation Class: composed by file and row
class WitnessSet(BaseModel):
    tables_rows: List[str] = Field(description="List of table_row strings, e.g., ['customer_14322', 'orders_137']")

class AnswerItem(BaseModel):
    answer: List[str] = Field(description="The final answer(s) to the question.")
    why: List[WitnessSet] = Field(description="List of witness sets justifying the answer. Each witness set is a list of table_row strings.")
    next_query_hint: Optional[str] = Field(None, description="An optional hint or refined query for the next iteration, based on the current answer to find related information. This could be a specific entity ID, a condition, or a rephrased question.")
    final_answer: bool = Field(False, description="True if this is the final answer and no more iterations are needed.")


# Define state for application
class State(TypedDict):
    original_question: str # To keep track of the initial question
    current_question: str  # The question used for retrieval in the current iteration
    context: List[Document]
    answer: List[AnswerItem]
    k: int
    iteration_history: List[Dict[str, Any]] # To store previous answers and contexts for iterative refinement

parser = JsonOutputParser(pydantic_schema=AnswerItem)

# Define application steps
# Retrieved the most k relevant docs in the vector store, embedding also the question and computing the similarity function
def retrieve(state: State):
    print(f"Retrieving for question: {state['current_question']}")
    retrieved_docs = vector_store.similarity_search(state["current_question"], k = state["k"])
    return {"context": retrieved_docs}

# Generate the answer invoking the LLM with the context joined with the question
def generate(state: State):
    # Construct a detailed prompt for the LLM
    context_str = "\n".join([f"- source: {doc.metadata.get('source').split('/')[-1].replace('.csv', '')} , row: {doc.metadata.get('row')}\n({doc.page_content})" for doc in state["context"]])

    prompt_with_explanation = f"""
    You are an expert at extracting information from provided CSV data to answer complex questions, especially those requiring information from multiple tables.
    Your task is to:
    1. **Answer the given question** based *only* on the provided CONTEXT. If the context does not contain enough information to fully answer the question, state that you need more information and provide a `next_query_hint` to guide the next retrieval step.
    2. For each answer, explain WHY it appears using **Witness Sets**: minimal sets of input tuples that justify the result.
    3. Determine if the answer is **final** or if more information is needed.

    **CONTEXT:**
    {context_str}

    **QUESTION:**
    {state["current_question"]}

    **IMPORTANT FORMATTING RULES:**
    * Return the output as a **stringified JSON array**, with NO extra text or comments outside the JSON.
    * The structure of the JSON array MUST follow this template:
        ```json
        [
            {{
                "answer": ["<answer_1>", "<answer_2>"],
                "why": [
                    {{"tables_rows": ["<table_name_1_row_a>", "<table_name_2_row_b>"]}},
                    {{"tables_rows": ["<table_name_3_row_c>", "<table_name_4_row_d>"]}}
                ],
                "next_query_hint": "Optional hint for next retrieval, e.g., 'customer_id: 123', or a rephrased question.",
                "final_answer": true/false
            }}
        ]
        ```
    * For `source` in `Witness Sets`, use just the table name (e.g., `customer` for `csv_data/customer.csv`), WITHOUT the `.csv` extension or path. Use the `row` metadata.
    * If no answer can be formed, provide an empty `answer` list, an empty `why` list, and set `final_answer` to `false` and provide a `next_query_hint` if possible.

    **Example of Expected Answer (if the question was "Which orders (o_orderkey) done by a customer with nationkey = 2 have a total price between 20500 and 20550?"):**
    ```json
    [
        {{
            "answer": ["546", "314052"],
            "why": [
                {{"tables_rows": ["customer_14322", "orders_137"]}},
                {{"tables_rows": ["customer_101", "orders_78528"]}}
            ],
            "next_query_hint": null,
            "final_answer": true
        }}
    ]
    ```
    """
    print("\n[DEBUG] PROMPT SENT TO LLM:")
    print(prompt_with_explanation)

   
    # Corrected line: Pass a list of BaseMessages
    messages= [
        SystemMessage(content="You are an expert at extracting information from CSV data."),
        HumanMessage(content=prompt_with_explanation),
    ]
    response = llm.invoke(messages)

    print("\nRAW LLM RESPONSE:")
    print(response.content)

    try:
        parsed = parser.parse(response.content)
        # Ensure 'answer' is a list and 'why' contains WitnessSet objects
        if not isinstance(parsed, list):
            parsed = [parsed] # Wrap if it's a single object
        for item in parsed:
            # Important: Ensure 'item' is an AnswerItem here if it's not already
            # The JsonOutputParser with pydantic_schema=AnswerItem *should* return AnswerItem instances,
            # but sometimes there are nuances. A safe check might be:
            if not isinstance(item, AnswerItem):
                # This might happen if `parser.parse` returns raw dicts from a generic JSON parse
                # and you rely on later manual conversion.
                # For robustness, let's assume `parser.parse` generally works as intended to produce AnswerItem.
                # If not, you'd need `item = AnswerItem(**item)` here.
                pass # If parser.parse already returns AnswerItem, no conversion needed.

            if not isinstance(item.answer, list):
                item.answer = [str(item.answer)]
            if not isinstance(item.why, list):
                item.why = [item.why] # Ensure why is a list
            item.why = [WitnessSet(tables_rows=ws.tables_rows if isinstance(ws, WitnessSet) else (ws if isinstance(ws, list) else [ws])) for ws in item.why] # Ensure inner witness sets are lists, handle if already WitnessSet


    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        print("Attempting to parse as a list of AnswerItem directly...")
        try:
            # Fallback parsing if initial attempt fails
            parsed_dicts = json.loads(response.content)
            # Manually convert to AnswerItem if parsing as dicts was successful
            parsed = [AnswerItem(**item) for item in parsed_dicts] # Ensure these are AnswerItem instances
        except Exception as e_fallback:
            print(f"Fallback parsing also failed: {e_fallback}")
            # Corrected: Create an AnswerItem instance, not a plain dict
            parsed = [AnswerItem(answer=[], why=[], next_query_hint=None, final_answer=False)]
            print("Set parsed to default empty AnswerItem due to persistent parsing errors.")

    # Update the state with the parsed answer and iteration history
    # current_answer_item should be an AnswerItem, not a dict.
    # Since parsed is now guaranteed to contain AnswerItem instances (or be empty),
    # parsed[0] will be an AnswerItem or this condition will be skipped.
    current_answer_item = parsed[0] if parsed else AnswerItem(answer=[], why=[], next_query_hint=None, final_answer=False)

    return {
        "answer": parsed, # Store the parsed answer structure (list of AnswerItem)
        "iteration_history": state.get("iteration_history", []) + [{"question": state["current_question"], "context": state["context"], "answer": current_answer_item.dict() if isinstance(current_answer_item, BaseModel) else current_answer_item}]
    }
# Node to decide whether to continue iterating or stop
def decide_to_continue(state: State):
    if not state["answer"] or not state["answer"][0].final_answer:
        print("\nDECISION: Continue iterative retrieval.")
        return "continue"
    else:
        print("\nDECISION: Final answer reached, stopping.")
        return "end"

# Create a dictionary to store results for each k
results_by_k = {}
# Read questions from the JSON file
with open("tpch/questions.json", "r") as f:
    data = json.load(f)
    questions = list(data.keys())

# Build the graph structure once
workflow = StateGraph(State)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.add_conditional_edges(
    "generate",
    decide_to_continue,
    {"continue": "retrieve", "end": END} # Loop back to retrieve or end
)
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
        full_result = graph.invoke(current_state, config={"recursion_limit": 10})  # Set a recursion limit to avoid infinite loops

        # Update current_state for the next iteration
        current_state = full_result

        # Check if the LLM decided it's a final answer
        if current_state["answer"] and current_state["answer"][0].final_answer:
            print(f"Question {i+1} completed in {iter_num + 1} iterations.")
            break

        # If not final, update current_question for the next retrieval
        if current_state["answer"] and current_state["answer"][0].next_query_hint:
            # Use the hint from the LLM to refine the next question
            current_state["current_question"] = f"{current_state['original_question']} - Further information needed: {current_state['answer'][0].next_query_hint}"
            print(f"Updated current_question for next iteration: {current_state['current_question']}")
        else:
            # If no specific hint, try to rephrase based on the original question and what's been found (or simply repeat)
            # This part could be more sophisticated, e.g., asking for related entities.
            # For simplicity, if no hint, we'll just re-run with the original question and current context.
            print("No specific next_query_hint from LLM, continuing with original question and accumulated context.")
            current_state["current_question"] = current_state["original_question"]

    # Store the final result of the iterations for this question
    all_final_results.append({
        "original_question": current_state["original_question"],
        "final_answer": current_state["answer"], # The last answer generated
        "full_iteration_history": current_state["iteration_history"] # All intermediate steps
    })

output_filename = f"tpch/outputs_groq/outputs_llama8b/iterative/outputs_llama8b_iterative.json"
# Save the results for the current value of k to a JSON file for later analysis
os.makedirs(os.path.dirname(output_filename), exist_ok=True)
with open(output_filename, "w", encoding='utf-8') as output_file:
    json.dump(all_final_results, output_file, indent=4, ensure_ascii=False)
print(f"Results saved to {output_filename}")