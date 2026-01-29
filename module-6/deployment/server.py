"""
Simple FastAPI server to expose the LangGraph graph as an API.
This is a lightweight alternative to langgraph up for Docker deployments.
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
import json

# Import the graph from task_maistro
from task_maistro import graph
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver

# Initialize stores
store = InMemoryStore()
memory = MemorySaver()

# Recompile graph with checkpointer and store
from task_maistro import builder
compiled_graph = builder.compile(checkpointer=memory, store=store)

app = FastAPI(title="Task Maistro API", version="1.0.0")


class Message(BaseModel):
    role: str
    content: str


class InvokeRequest(BaseModel):
    messages: List[Message]
    thread_id: str
    user_id: str = "default_user"
    todo_category: str = "personal"
    task_maistro_role: str = "You are a helpful chatbot."


class InvokeResponse(BaseModel):
    messages: List[Dict[str, Any]]


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/invoke", response_model=InvokeResponse)
async def invoke_graph(request: InvokeRequest):
    """Invoke the graph with a list of messages."""
    try:
        # Convert messages to LangChain format
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
        
        lc_messages = []
        for msg in request.messages:
            if msg.role == "user" or msg.role == "human":
                lc_messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant" or msg.role == "ai":
                lc_messages.append(AIMessage(content=msg.content))
            elif msg.role == "system":
                lc_messages.append(SystemMessage(content=msg.content))
            else:
                lc_messages.append(HumanMessage(content=msg.content))
        
        config = {
            "configurable": {
                "thread_id": request.thread_id,
                "user_id": request.user_id,
                "todo_category": request.todo_category,
                "task_maistro_role": request.task_maistro_role,
            }
        }
        
        result = compiled_graph.invoke({"messages": lc_messages}, config=config)
        
        # Convert response messages to dict format
        response_messages = []
        for msg in result.get("messages", []):
            response_messages.append({
                "role": msg.type if hasattr(msg, 'type') else "unknown",
                "content": msg.content if hasattr(msg, 'content') else str(msg),
            })
        
        return InvokeResponse(messages=response_messages)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stream")
async def stream_graph(request: InvokeRequest):
    """Stream the graph response."""
    try:
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
        
        lc_messages = []
        for msg in request.messages:
            if msg.role == "user" or msg.role == "human":
                lc_messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant" or msg.role == "ai":
                lc_messages.append(AIMessage(content=msg.content))
            else:
                lc_messages.append(HumanMessage(content=msg.content))
        
        config = {
            "configurable": {
                "thread_id": request.thread_id,
                "user_id": request.user_id,
                "todo_category": request.todo_category,
                "task_maistro_role": request.task_maistro_role,
            }
        }
        
        async def generate():
            for chunk in compiled_graph.stream({"messages": lc_messages}, config=config, stream_mode="values"):
                yield json.dumps({"messages": [{"content": str(chunk)}]}) + "\n"
        
        return StreamingResponse(generate(), media_type="application/x-ndjson")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
