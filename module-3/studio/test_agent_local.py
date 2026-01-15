
import sys
import os

# Add the current directory to path so we can import agent
sys.path.append(os.getcwd())

from agent import graph
from langchain_core.messages import HumanMessage

print("--- Starting Local Graph Test ---")
try:
    input_message = HumanMessage(content="Multiply 2 and 3")
    print(f"Invoking graph with: {input_message}")
    
    # Test simple invoke
    result = graph.invoke({"messages": [input_message]})
    print("Graph invocation successful!")
    print("Result:", result)
    
except Exception as e:
    print("--- Local Graph Test FAILED ---")
    print(e)
    import traceback
    traceback.print_exc()

