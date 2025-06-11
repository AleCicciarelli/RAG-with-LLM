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
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
os.environ["LANGSMITH_TRACING"] = "false" 
os.environ["LANGSMITH_API_KEY"] = "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"

if not os.environ.get("GROQ_API_KEY"):
 os.environ["GROQ_API_KEY"] = "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"

# LLM: Llama3-8b by Groq
llm = init_chat_model("llama3-70b-8192", model_provider="groq", temperature = 0)


prompt = PromptTemplate.from_template(
    """Based on the schema below and this question: {question}, generate a plan of natural-language, atomic steps to answer the question. 
    The plan should be a list of minimum steps, each step should be a single action that can be executed to answer the question. 
    
    SCHEMA:
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
    The plan should be in the following format:
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
    "question": "What is the order date of the order with order key 323?"
})

print(response)