# from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

# Tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

# LLM with bound tool
# llm = ChatOpenAI(model="gpt-4o")

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
llm_with_tools = llm.bind_tools([multiply])

# Node
def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode([multiply]))
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", END)

# Compile graph
graph = builder.compile()