import os # Módulo para interactuar con el sistema operativo y variables de entorno
from dataclasses import dataclass, field, fields # Utilidades para crear clases de datos (dataclasses)
from typing import Any, Optional # Tipos para anotaciones compatibles con cualquier valor u opcionales

from langchain_core.runnables import RunnableConfig # Clase que maneja la configuración de ejecución en LangChain
from typing_extensions import Annotated # Para añadir metadatos a las anotaciones de tipo

@dataclass(kw_only=True) # Crea una clase de datos donde los argumentos deben pasarse por nombre (keyword-only)
class Configuration:
    """Define los campos configurables para el asistente del chatbot."""
    user_id: str = "default-user" # Identificador único del usuario (por defecto 'default-user')
    todo_category: str = "general" # Categoría de la lista de tareas (por defecto 'general')
    # Rol predefinido del sistema para el asistente
    task_maistro_role: str = "You are a helpful task management assistant. You help you create, organize, and manage the user's ToDo list."

    @classmethod # Método de clase para instanciar la configuración desde una fuente externa
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Crea una instancia de Configuration a partir de un objeto RunnableConfig de LangChain."""
        # Extrae el diccionario 'configurable' si existe dentro de la configuración
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        # Diccionario para acumular los valores finales de configuración
        values: dict[str, Any] = {
            # Busca el valor: 1. En variables de entorno (en MAYÚSCULAS) 2. En el diccionario configurable
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls) # Itera sobre los campos definidos en la clase
            if f.init # Solo considera campos que se pueden inicializar
        }
        # Retorna la instancia de la clase filtrando solo los valores que no son nulos
        return cls(**{k: v for k, v in values.items() if v})