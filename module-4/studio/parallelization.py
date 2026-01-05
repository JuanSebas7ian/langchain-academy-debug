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

from langchain_aws import ChatBedrock

# 1. CONFIGURACIÓN PARA DEEPSEEK-R1 (Razonamiento Complejo)
# Ideal para agentes que necesitan planificar pasos lógicos.
# llm = ChatBedrock(
#     model_id="us.deepseek.r1-v1:0",  # ID oficial validado
#     region_name="us-east-1",
#     model_kwargs={
#         "temperature": 0.6, # DeepSeek recomienda 0.6 para razonamiento
#         "max_tokens": 8192,  # Recomendado para no degradar calidad del CoT
#         "top_p": 0.95,
#     }
# )


# llm = ChatBedrock(
#     model_id="us.deepseek.v3-v1:0", # Prueba este primero
#     region_name="us-east-1",        # O us-west-2
#     model_kwargs={
#         "temperature": 0.7,
#         "max_tokens": 4096
#     }
# )

llm = ChatBedrock(
    model_id="us.meta.llama4-scout-17b-instruct-v1:0",  # Nota el prefijo "us."
    # model_id="cohere.command-r-plus-v1:0",
    region_name="us-east-1",
    model_kwargs={
        "temperature": 0.5,
        "max_tokens": 2048,
        "top_p": 0.9,
    }
)


# llm = ChatBedrock(
#     model_id="us.meta.llama4-maverick-17b-instruct-v1:0",  # Nota el prefijo "us."
#     region_name="us-east-1",
#     model_kwargs={
#         "temperature": 0.5,
#         "max_tokens": 2048,
#         "top_p": 0.9,
#     }
# )

# llm = ChatBedrock(
#     model_id="amazon.nova-lite-v1:0",  # Nota el prefijo "us."
#     region_name="us-east-1",
#     model_kwargs={
#         "temperature": 0.5,
#         "max_tokens": 2048,
#         "top_p": 0.9,
#     }
# )

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
