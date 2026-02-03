# Usamos la imagen oficial de LangGraph como base para tener los endpoints de hilos y Studio
FROM langchain/langgraph-api:3.11

# Directorio donde el servidor espera encontrar el código
WORKDIR /deps/project_code

# Copiamos el archivo de requerimientos primero para cachear la instalación
COPY requirements.txt .

# Instalamos las dependencias adicionales del proyecto (trustcall, bedrock, etc.)
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el resto del código
COPY . .

# El servidor de la imagen base ya expone el puerto 8000 y define el CMD
