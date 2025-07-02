import os
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import json
from langchain_community.chat_models import ChatOllama

os.environ["LANGSMITH_TRACING"] = "false" 
os.environ["LANGSMITH_API_KEY"] = "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"

#if not os.environ.get("GROQ_API_KEY"):
# os.environ["GROQ_API_KEY"] = "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"

llm = ChatOllama(model="llama3:70b", temperature=0)

# LLM: Llama3-8b by Groq
#llm = init_chat_model("llama3-70b-8192", model_provider="groq", temperature = 0)

def generate_plan(question, schema, llm):
    """
    Generate a plan of atomic steps to answer the question based on the provided schema.
    
    Args:
        question (str): The question to be answered.
        schema (str): The database schema in a structured format.
        llm(ChatOllama): The language model to use for generating the plan.
    Returns:
        plan: A list of atomic steps to answer the question.
    """
    prompt = PromptTemplate.from_template(
        """Based on the schema below and this question: {question}, generate a plan of natural-language, atomic steps to answer the question. 
        The plan should be a list of minimum steps, each step should be a single action that can be executed to answer the question. 
        
       SCHEMA :
       {schema}
        The plan should be in the following format without any additional text or explanation:

        [
            "Step 1: <action description>",
            "Step 2: <action description>",
            ...
        ]

        """
    )

    chain = LLMChain(
        llm=llm,
        prompt=prompt
    )

    response = chain.run({
        "question": question,
        "schema": schema
    })

    return response

prompt = PromptTemplate.from_template(
    """Based on the schema below and this question: {question}, generate a plan of natural-language, atomic steps to answer the question. 
    The plan should be a list of minimum steps, each step should be a single action that can be executed to answer the question. 
    
   SCHEMA :
   {schema}
    The plan should be in the following format without any additional text or explanation:

    [
        "Step 1: <action description>",
        "Step 2: <action description>",
        ...
    ]

    """
)

chain = LLMChain(
    llm=llm,
    prompt = prompt 
)


response = chain.run({
    "question": "Which suppliers (s_name) belong to the region with r_name EUROPE and have a balance lower than 0?",
    "schema": """
 
    1. CUSTOMER(
        C_CUSTKEY INTEGER PRIMARY KEY,
        C_NAME VARCHAR,
        C_ADDRESS VARCHAR,
        C_NATIONKEY INTEGER REFERENCES NATION(N_NATIONKEY),
        C_PHONE VARCHAR,
        C_ACCTBAL DECIMAL,
        C_MKTSEGMENT VARCHAR,
    )
    
    2. ORDERS(
        O_ORDERKEY INTEGER PRIMARY KEY,
        O_CUSTKEY INTEGER REFERENCES CUSTOMER(C_CUSTKEY),
        O_ORDERSTATUS CHAR,
        O_TOTALPRICE DECIMAL,
        O_ORDERDATE DATE,
        O_ORDERPRIORITY VARCHAR,
        O_CLERK VARCHAR,
        O_SHIPPRIORITY INTEGER,
    )
    
    3. LINEITEM(
        L_ORDERKEY INTEGER REFERENCES ORDERS(O_ORDERKEY),
        L_PARTKEY INTEGER REFERENCES PART(P_PARTKEY),
        L_SUPPKEY INTEGER REFERENCES SUPPLIER(S_SUPPKEY),
        L_LINENUMBER INTEGER,
        L_QUANTITY DECIMAL,
        L_EXTENDEDPRICE DECIMAL,
        L_DISCOUNT DECIMAL,
        L_TAX DECIMAL,
        L_RETURNFLAG CHAR,
        L_LINESTATUS CHAR,
        L_SHIPDATE DATE,
        L_COMMITDATE DATE,
        L_RECEIPTDATE DATE,
        L_SHIPINSTRUCT VARCHAR,
        L_SHIPMODE VARCHAR,
        PRIMARY KEY (L_ORDERKEY, L_LINENUMBER)
    )
    
    4. PART(
        P_PARTKEY INTEGER PRIMARY KEY,
        P_NAME VARCHAR,
        P_MFGR VARCHAR,
        P_BRAND VARCHAR,
        P_TYPE VARCHAR,
        P_SIZE INTEGER,
        P_CONTAINER VARCHAR,
        P_RETAILPRICE DECIMAL,
    )
    
    5. SUPPLIER(
        S_SUPPKEY INTEGER PRIMARY KEY,
        S_NAME VARCHAR,
        S_ADDRESS VARCHAR,
        S_NATIONKEY INTEGER REFERENCES NATION(N_NATIONKEY),
        S_PHONE VARCHAR,
        S_ACCTBAL DECIMAL,
    )
    
    6. PARTSUPP(
        PS_PARTKEY INTEGER REFERENCES PART(P_PARTKEY),
        PS_SUPPKEY INTEGER REFERENCES SUPPLIER(S_SUPPKEY),
        PS_AVAILQTY INTEGER,
        PS_SUPPLYCOST DECIMAL,
        PRIMARY KEY (PS_PARTKEY, PS_SUPPKEY)
    )
    
    7. NATION(
        N_NATIONKEY INTEGER PRIMARY KEY,
        N_NAME VARCHAR,
        N_REGIONKEY INTEGER REFERENCES REGION(R_REGIONKEY),
    )
    
    8. REGION(
        R_REGIONKEY INTEGER PRIMARY KEY,
        R_NAME VARCHAR,
    )
    """
})

print(response)
try:
    response = response.strip()
    print(f"Raw response: {response}")

    # Cerca le parentesi quadre per isolare la lista JSON
    start = response.find("[")
    end = response.rfind("]")

    if start != -1 and end != -1:
        extracted = response[start:end+1]
        print(f"📦 Extracted list: {extracted}")
        plan = json.loads(extracted)
        print(f"✅ Parsed plan with {len(plan)} steps")
    else:
        raise ValueError("No JSON list brackets found in the response.")

except Exception as e:
    print(f"⚠️ Error parsing plan: {e}")
    plan = []
max_iterations = len(plan) if isinstance(plan, list) else 1
if not isinstance(plan, list):
    print("⚠️ The response is not a valid list. Using max_iterations = 1.")
else:
    plan = [step.strip() for step in plan if isinstance(step, str) and step.strip()]
    if not plan:
        print("⚠️ The plan is empty after filtering. Using max_iterations = 1.")
        max_iterations = 1
    else:
        max_iterations = len(plan)
print(f"[INFO] Generated plan with {len(plan)} steps. Setting max_iterations = {max_iterations}")
