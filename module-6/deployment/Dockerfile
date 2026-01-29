FROM python:3.11-slim

WORKDIR /app

# Install system dependencies if needed (e.g. for building packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port the API runs on
EXPOSE 8000

# Command to run the LangGraph server using the CLI
CMD ["langgraph", "up", "--host", "0.0.0.0", "--port", "8000"]
