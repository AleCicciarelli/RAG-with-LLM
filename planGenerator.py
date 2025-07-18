import os
import re
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import json
from langchain_community.chat_models import ChatOllama
from langchain.chat_models import init_chat_model
os.environ["LANGSMITH_TRACING"] = "false" 
os.environ["LANGSMITH_API_KEY"] = "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"

#if not os.environ.get("GROQ_API_KEY"):
# os.environ["GROQ_API_KEY"] = "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"

llm = ChatOllama(model="llama3:70b", temperature=0)

# LLM: Llama3-8b by Groq
#llm = init_chat_model("llama3-70b-8192", model_provider="groq", temperature = 0)

#schema_path = "schemaTPCH.txt"

# Load the schema from the file
#with open(schema_path, "r") as f:
    #schema = f.read().strip()
# Print the schema to verify it has been loaded correctly
#print(f"Schema loaded from {schema_path}:\n{schema}\n")
def generate_plan(question, schema, llm):
    """
    Generate a plan of atomic steps to answer the question based on the provided schema.

    Args:
        question (str): The question to be answered.
        schema (str): The database schema in a structured format.
        llm (ChatOllama): The language model to use for generating the plan.

    Returns:
        tuple: (plan, max_iterations) where:
            - plan is a list of atomic steps (strings)
            - max_iterations is the number of steps, or 1 if fallback
    """
    print("📌 generate_plan called")

    prompt = PromptTemplate.from_template(
        """Based on the schema below and this question: {question}, generate a plan of natural-language, atomic steps to answer the question. 
        The plan should be a list of minimum steps, each step should be a single action that can be executed to answer the question. 
        
        SCHEMA:
        {schema}

        The plan should be in the following format without any additional text or explanation including the square brackets:

        [
            "Step 1: <action description>",
            "Step 2: <action description>",
            ...
        ]
        """
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    try:
        response = chain.run({"question": question, "schema": schema}).strip()
        print(f"📨 Raw LLM response:\n{response}")

        # Estrarre la lista JSON tra le prime parentesi quadre
        start = response.find("[")
        end = response.rfind("]")

        if start == -1 or end == -1 or start >= end:
            raise ValueError("No valid bracketed list found in the LLM response.")

        extracted = response[start:end+1]
        extracted = re.sub(r",\s*\]", "]", extracted)  # Rimuove virgole finali

        plan = json.loads(extracted)
        if not isinstance(plan, list):
            raise ValueError("Extracted plan is not a list.")

        # Pulisce ogni step
        plan = [step.strip() for step in plan if isinstance(step, str) and step.strip()]
        if not plan:
            raise ValueError("Plan is empty after filtering.")

        max_iterations = len(plan)
        print(f"✅ Parsed {max_iterations} steps.")

    except Exception as e:
        print(f"⚠️ Failed to generate plan: {e}")
        plan = []
        max_iterations = 1

    print(f"[INFO] Returning plan with {len(plan)} steps and max_iterations = {max_iterations}")
    return plan, max_iterations
