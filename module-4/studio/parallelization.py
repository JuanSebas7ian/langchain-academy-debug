import operator
from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from langchain_community.document_loaders import WikipediaLoader
from langchain_tavily import TavilySearch  # updated 1.0

# from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END

# llm = ChatOpenAI(model="gpt-4o", temperature=0) 

from dotenv import load_dotenv
load_dotenv('/home/juansebas7ian/langchain-academy/.env')

from langchain_aws import ChatBedrockConverse

# 1. CONFIGURACIÓN PARA DEEPSEEK-R1 (Razonamiento Complejo)
# Ideal para agentes que necesitan planificar pasos lógicos.
llm_deepseek_r1 = ChatBedrockConverse(
    model="us.deepseek.r1-v1:0",
    region_name="us-east-1",
    temperature=0.6,
    max_tokens=8192,
    top_p=0.95,
)

# 2. CONFIGURACIÓN PARA DEEPSEEK-V3
llm_deepseek_v3 = ChatBedrockConverse(
    model="us.deepseek.v3-v1:0",
    region_name="us-east-1",
    temperature=0.7,
    max_tokens=4096,
)

# 3. CONFIGURACIÓN PARA LLAMA 4 SCOUT
llm_scout = ChatBedrockConverse(
    model="us.meta.llama4-scout-17b-instruct-v1:0",
    region_name="us-east-1",
    temperature=0.5,
    max_tokens=2048,
    top_p=0.9,
)

# 4. CONFIGURACIÓN PARA LLAMA 4 MAVERICK
llm_maverick = ChatBedrockConverse(
    model="us.meta.llama4-maverick-17b-instruct-v1:0",
    region_name="us-east-1",
    temperature=0.5,
    max_tokens=2048,
    top_p=0.9,
)

# 5. CONFIGURACIÓN PARA AMAZON NOVA LITE
llm_nova_lite = ChatBedrockConverse(
    model="amazon.nova-lite-v1:0",
    region_name="us-east-1",
    temperature=0.5,
    max_tokens=2048,
    top_p=0.9,
)

# 6. CONFIGURACIÓN PARA AMAZON NOVA MICRO
llm_nova_micro = ChatBedrockConverse(
    model="amazon.nova-micro-v1:0",
    region_name="us-east-1",
    temperature=0.5,
    max_tokens=2048,
    top_p=0.9,
)

# 7. CONFIGURACIÓN PARA AMAZON NOVA PRO
llm_nova_pro = ChatBedrockConverse(
    model="amazon.nova-pro-v1:0",
    region_name="us-east-1",
    temperature=0.5,
    max_tokens=2048,
    top_p=0.9,
)

# Seleccionar el LLM activo
llm = llm_scout

class State(TypedDict):
    question: str
    answer: str
    context: Annotated[list, operator.add]

def search_web(state):
    
    """ Retrieve docs from web search """

    # Search
    tavily_search = TavilySearch(max_results=3)
    data = tavily_search.invoke({"query": state['question']})
    search_docs = data.get("results", data)

     # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]} 

def search_wikipedia(state):
    
    """ Retrieve docs from wikipedia """

    # Search
    search_docs = WikipediaLoader(query=state['question'], 
                                  load_max_docs=2).load()

     # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]} 

def generate_answer(state):
    
    """ Node to answer a question """

    # Get state
    context = state["context"]
    question = state["question"]

    # Template
    answer_template = """Answer the question {question} using this context: {context}"""
    answer_instructions = answer_template.format(question=question, 
                                                       context=context)    
    
    # Answer
    answer = llm.invoke([SystemMessage(content=answer_instructions)]+[HumanMessage(content=f"Answer the question.")])
      
    # Append it to state
    return {"answer": answer}

# Add nodes
builder = StateGraph(State)

# Initialize each node with node_secret 
builder.add_node("search_web",search_web)
builder.add_node("search_wikipedia", search_wikipedia)
builder.add_node("generate_answer", generate_answer)

# Flow
builder.add_edge(START, "search_wikipedia")
builder.add_edge(START, "search_web")
builder.add_edge("search_wikipedia", "generate_answer")
builder.add_edge("search_web", "generate_answer")
builder.add_edge("generate_answer", END)
graph = builder.compile()
