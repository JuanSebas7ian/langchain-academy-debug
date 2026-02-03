"""
Servidor FastAPI sencillo para exponer el grafo de LangGraph como una API.
Esta es una alternativa ligera a 'langgraph up' para despliegues con Docker.
"""
# Importa los módulos necesarios de FastAPI para el servidor web y manejo de errores
from fastapi import FastAPI, HTTPException
# Importa StreamingResponse para manejar respuestas en tiempo real (streaming)
from fastapi.responses import StreamingResponse
# Importa BaseModel de Pydantic para definir el esquema de los datos de entrada/salida
from pydantic import BaseModel
# Importa tipos de Python para anotaciones de tipo claras
from typing import Optional, List, Dict, Any
# Importa uvicorn para ejecutar la aplicación web ASGI
import uvicorn
# Importa json para manipular strings en formato JSON
import json

# Importa el grafo definido anteriormente en el archivo task_maistro
from task_maistro import graph
# Importa el almacenamiento en memoria para nodos que requieren guardar datos (como el perfil del usuario)
from langgraph.store.memory import InMemoryStore
# Importa el checkpointer en memoria para guardar el historial de la conversación (threads)
from langgraph.checkpoint.memory import MemorySaver

# Inicializa el almacenamiento de nodos (InMemoryStore)
store = InMemoryStore()
# Inicializa el guardado de historial de hilos (MemorySaver)
memory = MemorySaver()

# Importa el constructor del grafo para recompilarlo con persistencia
from task_maistro import builder
# Compila el grafo inyectándole el checkpointer (memoria corta) y el store (memoria larga persistente)
compiled_graph = builder.compile(checkpointer=memory, store=store)

# Crea la instancia principal de la aplicación FastAPI con un título y versión
app = FastAPI(title="Task Maistro API", version="1.0.0")


# DEFINE LOS MODELOS DE DATOS (SCHEMAS)
# Modelo de un mensaje individual
class Message(BaseModel):
    role: str # Rol del emisor: 'user', 'assistant' o 'system'
    content: str # Contenido de texto del mensaje


# Modelo de la petición de entrada al endpoint /invoke
class InvokeRequest(BaseModel):
    messages: List[Message] # Lista de mensajes previos y actuales
    thread_id: str # Identificador único de la conversación
    user_id: str = "default_user" # ID del usuario (por defecto 'default_user')
    todo_category: str = "personal" # Categoría de tareas (por defecto 'personal')
    task_maistro_role: str = "You are a helpful chatbot." # Instrucciones del sistema


# Modelo de la respuesta de salida del endpoint /invoke
class InvokeResponse(BaseModel):
    messages: List[Dict[str, Any]] # Lista de mensajes procesados por el agente


# DEFINE LOS ENDPOINTS (RUTAS)
# Endpoint de diagnóstico para verificar que el servidor está vivo
@app.get("/health")
async def health_check():
    """Endpoint de salud del servicio."""
    return {"status": "healthy"} # Retorna un status 200 OK


# Endpoint para ejecutar el grafo de forma síncrona (espera a que termine)
@app.post("/invoke", response_model=InvokeResponse)
async def invoke_graph(request: InvokeRequest):
    """Invoca el grafo con una lista de mensajes."""
    try:
        # Importa las clases de mensajes oficiales de LangChain
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
        
        # Convierte los mensajes recibidos formato JSON a objetos de LangChain
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
        
        # Configuración del estado configurable (usado por los nodos del grafo)
        config = {
            "configurable": {
                "thread_id": request.thread_id,
                "user_id": request.user_id,
                "todo_category": request.todo_category,
                "task_maistro_role": request.task_maistro_role,
            }
        }
        
        # Ejecuta el grafo pasando los mensajes y la configuración
        result = compiled_graph.invoke({"messages": lc_messages}, config=config)
        
        # Convierte los objetos de mensaje de LangChain de vuelta a diccionarios JSON
        response_messages = []
        for msg in result.get("messages", []):
            response_messages.append({
                "role": msg.type if hasattr(msg, 'type') else "unknown",
                "content": msg.content if hasattr(msg, 'content') else str(msg),
            })
        
        # Retorna la lista de mensajes generada por el agente
        return InvokeResponse(messages=response_messages)
    
    except Exception as e:
        # En caso de error, retorna un error 500 con el detalle de la excepción
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint para ejecutar el grafo en modo streaming (recibir tokens poco a poco)
@app.post("/stream")
async def stream_graph(request: InvokeRequest):
    """Retorna la respuesta del grafo en modo streaming."""
    try:
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
        
        # Convierte los mensajes recibidos formato JSON a objetos de LangChain
        lc_messages = []
        for msg in request.messages:
            if msg.role == "user" or msg.role == "human":
                lc_messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant" or msg.role == "ai":
                lc_messages.append(AIMessage(content=msg.content))
            else:
                lc_messages.append(HumanMessage(content=msg.content))
        
        # Configuración del estado configurable (usado por los nodos del grafo)
        config = {
            "configurable": {
                "thread_id": request.thread_id,
                "user_id": request.user_id,
                "todo_category": request.todo_category,
                "task_maistro_role": request.task_maistro_role,
            }
        }
        
        # Función generadora interna para el streaming
        async def generate():
            # Itera sobre los chunks generados por el grafo en tiempo real
            for chunk in compiled_graph.stream({"messages": lc_messages}, config=config, stream_mode="values"):
                # Retorna cada chunk como un string JSON seguido de un salto de línea
                yield json.dumps({"messages": [{"content": str(chunk)}]}) + "\n"
        
        # Retorna una respuesta de streaming con el generador definido
        return StreamingResponse(generate(), media_type="application/x-ndjson")
    
    except Exception as e:
        # Manejo de excepciones para el stream
        raise HTTPException(status_code=500, detail=str(e))


# PUNTO DE ENTRADA DEL SCRIPT
if __name__ == "__main__":
    # Inicia el servidor uvicorn escuchando en todas las interfaces (0.0.0.0) y en el puerto 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
