import uuid # Genera identificadores únicos universales (UUID)
from datetime import datetime # Maneja fechas y horas

from pydantic import BaseModel, Field # Validación de datos y esquemas con Pydantic

from trustcall import create_extractor # Librería especializada en extraer información estructurada

from typing import Literal, Optional, TypedDict # Tipos de datos para anotaciones de tipo claras

from langchain_core.runnables import RunnableConfig # Configuración para ejecuciones de LangChain
from langchain_core.messages import merge_message_runs # Combina mensajes consecutivos del mismo rol
from langchain_core.messages import SystemMessage, HumanMessage # Clases para mensajes de sistema y humanos

# from langchain_openai import ChatOpenAI # (Comentado) Integración con OpenAI
from langchain_aws import ChatBedrockConverse # Integración para modelos de AWS Bedrock
from dotenv import load_dotenv # Carga variables desde archivos .env

load_dotenv() # Carga las credenciales del entorno

from langgraph.checkpoint.memory import MemorySaver # Checkpointer en memoria para persistencia de hilos
from langgraph.graph import StateGraph, MessagesState, START, END # Elementos básicos para construir grafos
from langgraph.store.base import BaseStore # Clase base para almacenamiento persistente
from langgraph.store.memory import InMemoryStore # Implementación en memoria del store

import configuration # Importa la configuración personalizada del proyecto

## Utilities (Utilidades)

# Clase Spy para inspeccionar las llamadas a herramientas (tool calls) hechas por Trustcall
class Spy:
    def __init__(self):
        self.called_tools = [] # Lista para almacenar las herramientas invocadas

    def __call__(self, run):
        # Función que se ejecuta al terminar un paso del grafo
        q = [run]
        while q:
            r = q.pop()
            if r.child_runs:
                q.extend(r.child_runs) # Explora ejecuciones hijas
            if r.run_type == "chat_model":
                # Si es un modelo de chat, guarda sus llamadas a herramientas
                self.called_tools.append(
                    r.outputs["generations"][0][0]["message"]["kwargs"]["tool_calls"]
                )

# Función para extraer información legible de las llamadas a herramientas (parches o memorias nuevas)
def extract_tool_info(tool_calls, schema_name="Memory"):
    """Extrae información de llamadas a herramientas tanto para actualizaciones como creaciones."""
    # Lista para acumular los cambios detectados
    changes = []
    
    for call_group in tool_calls:
        for call in call_group:
            if call['name'] == 'PatchDoc':
                # Si la herramienta es PatchDoc (actualización incremental)
                if call['args']['patches']:
                    changes.append({
                        'type': 'update',
                        'doc_id': call['args']['json_doc_id'],
                        'planned_edits': call['args']['planned_edits'],
                        'value': call['args']['patches'][0]['value']
                    })
                else:
                    # Caso donde el modelo decidió que no hacían falta cambios
                    changes.append({
                        'type': 'no_update',
                        'doc_id': call['args']['json_doc_id'],
                        'planned_edits': call['args']['planned_edits']
                    })
            elif call['name'] == schema_name:
                # Si es una creación de un nuevo documento (Profile o ToDo)
                changes.append({
                    'type': 'new',
                    'value': call['args']
                })

    # Formatea los resultados en un solo string amigable para el usuario
    result_parts = []
    for change in changes:
        if change['type'] == 'update':
            result_parts.append(
                f"Document {change['doc_id']} updated:\n"
                f"Plan: {change['planned_edits']}\n"
                f"Added content: {change['value']}"
            )
        elif change['type'] == 'no_update':
            result_parts.append(
                f"Document {change['doc_id']} unchanged:\n"
                f"{change['planned_edits']}"
            )
        else:
            result_parts.append(
                f"New {schema_name} created:\n"
                f"Content: {change['value']}"
            )
    
    return "\n\n".join(result_parts) # Une todo con saltos de línea

## Schema definitions (Definición de Esquemas)

# Esquema para el perfil del usuario
class Profile(BaseModel):
    """Perfil del usuario con el que se está conversando"""
    name: Optional[str] = Field(description="The user's name", default=None) # Nombre del usuario
    location: Optional[str] = Field(description="The user's location", default=None) # Ubicación
    job: Optional[str] = Field(description="The user's job", default=None) # Profesión
    connections: list[str] = Field(
        description="Personal connection of the user, such as family members, friends, or coworkers",
        # Conexiones personales (familia, amigos)
        default_factory=list
    )
    interests: list[str] = Field(
        description="Interests that the user has", # Intereses y hobbies
        default_factory=list
    )

# Esquema para una tarea por hacer (ToDo)
class ToDo(BaseModel):
    task: str = Field(description="The task to be completed.") # Descripción de la tarea
    time_to_complete: Optional[int] = Field(description="Estimated time to complete the task (minutes).") # Tiempo estimado
    deadline: Optional[datetime] = Field(
        description="When the task needs to be completed by (if applicable)", # Fecha límite
        default=None
    )
    solutions: list[str] = Field(
        description="List of specific, actionable solutions (e.g., specific ideas, service providers, or concrete options relevant to completing the task)",
        # Sugerencias o soluciones concretas para la tarea
        min_items=1,
        default_factory=list
    )
    status: Literal["not started", "in progress", "done", "archived"] = Field(
        description="Current status of the task", # Estado actual de la tarea
        default="not started"
    )

## Initialize the model and tools (Inicialización del Modelo y Herramientas)

# Definición del tipo para decidir qué tipo de memoria actualizar
class UpdateMemory(TypedDict):
    """ Decisión sobre qué tipo de memoria actualizar """
    update_type: Literal['user', 'todo', 'instructions'] # Puede ser perfil, tareas o instrucciones

# Inicialización del LLM (AWS Bedrock Nova Lite)
llm_nova_lite = ChatBedrockConverse(
    model="us.amazon.nova-2-lite-v1:0", # ID del modelo Nova 2 Lite
    region_name="us-east-1", # Región de AWS
    temperature=0.5, # Creatividad balanceada
    max_tokens=2048, # Límite de tokens de salida
    top_p=0.9, # Muestreo núcleo (nucleous sampling)
)

# Establece el modelo activo
model = llm_nova_lite

# Crea el extractor de Trustcall para el perfil del usuario (usado en el nodo update_profile)
profile_extractor = create_extractor(
    model,
    tools=[Profile], # Usa el esquema Profile
    tool_choice="Profile", # Forza al modelo a usar este esquema
)

## Prompts (Mensajes del Sistema)

# Mensaje para el agente principal (task_mAIstro) que decide la ruta a seguir
MODEL_SYSTEM_MESSAGE = """{task_maistro_role} 

You have a long term memory which keeps track of three things:
1. The user's profile (general information about them) 
2. The user's ToDo list
3. General instructions for updating the ToDo list

Here is the current User Profile (may be empty if no information has been collected yet):
<user_profile>
{user_profile}
</user_profile>

Here is the current ToDo List (may be empty if no tasks have been added yet):
<todo>
{todo}
</todo>

Here are the current user-specified preferences for updating the ToDo list (may be empty if no preferences have been specified yet):
<instructions>
{instructions}
</instructions>

Here are your instructions for reasoning about the user's messages:

1. Reason carefully about the user's messages as presented below. 

2. Decide whether any of the your long-term memory should be updated:
- If personal information was provided about the user, update the user's profile by calling UpdateMemory tool with type `user`
- If tasks are mentioned, update the ToDo list by calling UpdateMemory tool with type `todo`
- If the user has specified preferences for how to update the ToDo list, update the instructions by calling UpdateMemory tool with type `instructions`

3. Tell the user that you have updated your memory, if appropriate:
- Do not tell the user you have updated the user's profile
- Tell the user them when you update the todo list
- Do not tell the user that you have updated instructions

4. Err on the side of updating the todo list. No need to ask for explicit permission.

5. Respond naturally to user user after a tool call was made to save memories, or if no tool call was made."""

# Instrucciones para el extractor Trustcall
TRUSTCALL_INSTRUCTION = """Reflect on following interaction. 

Use the provided tools to retain any necessary memories about the user. 

Use parallel tool calling to handle updates and insertions simultaneously.

System Time: {time}"""

# Instrucciones para actualizar las preferencias del ToDo
CREATE_INSTRUCTIONS = """Reflect on the following interaction.

Based on this interaction, update your instructions for how to update ToDo list items. Use any feedback from the user to update how they like to have items added, etc.

Your current instructions are:

<current_instructions>
{current_instructions}
</current_instructions>"""

## Node definitions (Definición de Nodos)

# Nodo principal: conversa con el usuario y decide si hay que guardar información
def task_mAIstro(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Carga memorias del store y las usa para personalizar la respuesta del chatbot."""
    
    # Obtiene identificadores de la configuración (threads, usuarios, etc.)
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id # ID del usuario
    todo_category = configurable.todo_category # Categoría de tareas
    task_maistro_role = configurable.task_maistro_role # Rol personalizado

    # Busca el perfil del usuario en el store permanente
    namespace = ("profile", todo_category, user_id)
    memories = store.search(namespace)
    user_profile = memories[0].value if memories else None

    # Busca la lista de tareas en el store permanente
    namespace = ("todo", todo_category, user_id)
    memories = store.search(namespace)
    todo = "\n".join(f"{mem.value}" for mem in memories) # Concatena las tareas

    # Busca instrucciones personalizadas
    namespace = ("instructions", todo_category, user_id)
    memories = store.search(namespace)
    instructions = memories[0].value if memories else ""
    
    # Prepara el mensaje del sistema con la información recuperada
    system_msg = MODEL_SYSTEM_MESSAGE.format(
        task_maistro_role=task_maistro_role, 
        user_profile=user_profile, 
        todo=todo, 
        instructions=instructions
    )

    # El modelo decide qué hacer; se le asocia la herramienta UpdateMemory para decidir la ruta
    response = model.bind_tools([UpdateMemory]).invoke(
        [SystemMessage(content=system_msg)] + state["messages"]
    )

    return {"messages": [response]} # Retorna el mensaje del modelo

# Nodo para actualizar el perfil del usuario
def update_profile(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Reflexiona sobre la historia y actualiza la colección de memorias del perfil."""
    
    # Configuración de usuario
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    todo_category = configurable.todo_category

    # Namespace (ruta) para el perfil
    namespace = ("profile", todo_category, user_id)

    # Recupera lo que ya sabemos del perfil para dar contexto al extractor
    existing_items = store.search(namespace)
    tool_name = "Profile"
    existing_memories = ([(existing_item.key, tool_name, existing_item.value)
                          for existing_item in existing_items]
                          if existing_items else None)

    # Prepara los mensajes para el extractor con la hora actual
    TRUSTCALL_INSTRUCTION_FORMATTED = TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
    updated_messages = list(merge_message_runs(
        messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)] + state["messages"][:-1]
    ))

    # Usa el Spy para capturar qué herramientas llama el extractor
    spy = Spy()
    profile_extractor_with_spy = profile_extractor.with_listeners(on_end=spy)

    # Lanza el extractor de Trustcall
    result = profile_extractor_with_spy.invoke({
        "messages": updated_messages, 
        "existing": existing_memories
    })

    # Guarda los resultados (nuevos o parches) en el store persistente
    for r, rmeta in zip(result["responses"], result["response_metadata"]):
        store.put(namespace,
                  rmeta.get("json_doc_id", str(uuid.uuid4())),
                  r.model_dump(mode="json"),
            )
    
    # Recupera ID de la llamada a la herramienta original para responderla correctamente
    tool_calls = state['messages'][-1].tool_calls
    # Extrae descripción legible de lo que cambió para que el agente sepa
    profile_update_msg = extract_tool_info(spy.called_tools, tool_name)
    return {"messages": [{"role": "tool", "content": profile_update_msg, "tool_call_id": tool_calls[0]['id']}]}

# Nodo para actualizar la lista de tareas (ToDos)
def update_todos(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Reflexiona sobre la historia y actualiza la colección de tareas."""
    
    # Configuración de usuario y categoría
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    todo_category = configurable.todo_category

    # Ruta para las tareas
    namespace = ("todo", todo_category, user_id)

    # Contexto de tareas existentes
    existing_items = store.search(namespace)
    tool_name = "ToDo"
    existing_memories = ([(existing_item.key, tool_name, existing_item.value)
                          for existing_item in existing_items]
                          if existing_items else None)

    # Preparación de mensajes para Trustcall
    TRUSTCALL_INSTRUCTION_FORMATTED = TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
    updated_messages = list(merge_message_runs(
        messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)] + state["messages"][:-1]
    ))

    # Spy para ver las llamadas internas de Trustcall
    spy = Spy()
    
    # Crea un extractor específico para las ToDos (permite inserciones nuevas)
    todo_extractor = create_extractor(
        model,
        tools=[ToDo],
        tool_choice=tool_name,
        enable_inserts=True
    ).with_listeners(on_end=spy)

    # Ejecuta el extractor
    result = todo_extractor.invoke({
        "messages": updated_messages, 
        "existing": existing_memories
    })

    # Persiste los cambios en el store (MongoDB/Memoria)
    for r, rmeta in zip(result["responses"], result["response_metadata"]):
        store.put(namespace,
                  rmeta.get("json_doc_id", str(uuid.uuid4())),
                  r.model_dump(mode="json"),
            )
        
    # Obtiene ID de la herramienta llamada en task_mAIstro
    tool_calls = state['messages'][-1].tool_calls

    # Genera el resumen legible de lo que se actualizó o creó
    todo_update_msg = extract_tool_info(spy.called_tools, tool_name)
    return {"messages": [{"role": "tool", "content": todo_update_msg, "tool_call_id": tool_calls[0]['id']}]}

# Nodo para actualizar las instrucciones de preferencia del usuario
def update_instructions(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Reflexiona sobre la historia y actualiza las instrucciones generales."""
    
    # Carga configuración
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    todo_category = configurable.todo_category
    
    # Namespace para instrucciones
    namespace = ("instructions", todo_category, user_id)

    # Obtiene instrucciones actuales si existen
    existing_memory = store.get(namespace, "user_instructions")
        
    # Pide al modelo generar nuevas instrucciones basadas en el historial
    system_msg = CREATE_INSTRUCTIONS.format(
        current_instructions=existing_memory.value if existing_memory else None
    )
    new_memory = model.invoke(
        [SystemMessage(content=system_msg)] + state['messages'][:-1] + 
        [HumanMessage(content="Please update the instructions based on the conversation")]
    )

    # Sobreescribe las instrucciones anteriores con las nuevas
    key = "user_instructions"
    store.put(namespace, key, {"memory": new_memory.content})
    
    # Responde a la llamada técnica de la herramienta
    tool_calls = state['messages'][-1].tool_calls
    return {"messages": [{"role": "tool", "content": "updated instructions", "tool_call_id": tool_calls[0]['id']}]}

# Función de enrutamiento (Conditional Edge): decide a qué nodo ir basándose en la salida de task_mAIstro
def route_message(state: MessagesState, config: RunnableConfig) -> Literal[END, "update_todos", "update_instructions", "update_profile"]:
    """Decide si el flujo termina o si debe ir a actualizar algún tipo de memoria."""
    message = state['messages'][-1] # Último mensaje del modelo
    
    # Si no hubo llamadas a herramientas, el flujo termina
    if len(message.tool_calls) == 0:
        return END
    else:
        # Analiza el tipo de actualización solicitado en la herramienta
        tool_call = message.tool_calls[0]
        if tool_call['args']['update_type'] == "user":
            return "update_profile" # Va al nodo de perfil
        elif tool_call['args']['update_type'] == "todo":
            return "update_todos" # Va al nodo de tareas
        elif tool_call['args']['update_type'] == "instructions":
            return "update_instructions" # Va al nodo de instrucciones
        else:
            raise ValueError("Tipo de actualización desconocido") # Error si el tipo no es válido

# Creación del Grafo de Estado
builder = StateGraph(MessagesState, config_schema=configuration.Configuration)

# Agrega los nodos definidos al constructor del grafo
builder.add_node(task_mAIstro) # Nodo de lógica principal
builder.add_node(update_todos) # Nodo de gestión de tareas
builder.add_node(update_profile) # Nodo de gestión de perfil
builder.add_node(update_instructions) # Nodo de gestión de instrucciones

# Define las conexiones (aristas) entre nodos
builder.add_edge(START, "task_mAIstro") # Inicia siempre en task_mAIstro
# Define la salida condicional de task_mAIstro usando la función de router
builder.add_conditional_edges("task_mAIstro", route_message)
# Después de actualizar cualquier memoria, vuelve a task_mAIstro para responder al usuario
builder.add_edge("update_todos", "task_mAIstro")
builder.add_edge("update_profile", "task_mAIstro")
builder.add_edge("update_instructions", "task_mAIstro")

# Compilación del Grafo final listo para ejecutarse
graph = builder.compile()