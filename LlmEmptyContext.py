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


""" Retrieve and Generate part """

# Define prompt for question-answering
prompt = hub.pull("rlm/rag-prompt")


# Generate the answer invoking the LLM with the context joined with the question
def generate(question):
    prompt_with_explanation = f"""
    Question: {question}
    Given this question, provide the answer including the explanation on where and how you get the information: 
    
    The answer must respect the following structure, but return it as a string representation of a JSON:

    ### IMPORTANT ###

    - The output must be a valid JSON object, WITHOUT extra text before or after the JSON object. If ypu don't know the answer just say I don't know.
    Example:
    {{
        "question": "In which area Laura Bianchi teaches?",
        "answer": [
            "Computer Science"
        ],
        "explanation": {{  "file": "<file_name>",
                "row": <row_number>}},  
            {{  "file": "<file_name>", 
                "row": <row_number>}}  }}
    """
   
    response = llm.invoke(prompt_with_explanation)
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
    


# Process questions from a txt file
with open("question.txt", "r") as f:
    questions = [line.strip() for line in f.readlines() if line.strip()]

all_results = []

''' Loop for LLM invocation on questions'''

for i, question in enumerate(questions):
    print(f"Processing question n. {i+1}")
    
    full_result = generate(question)
    result = {
        "question": question,
        "answer": full_result.get("answer", []),
        "explanation": full_result.get("explanation", [])
    }
    
    all_results.append(result)
# Save results to json file
with open("all_outputs_EMPTY.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=4, ensure_ascii=False)
