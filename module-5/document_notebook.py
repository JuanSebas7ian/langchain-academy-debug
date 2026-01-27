import json

notebook_path = '/home/juansebas7ian/langchain-academy/module-5/memory_agent.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Helper function to inject comments
def inject_documentation(source_code, documentation_map):
    lines = source_code.splitlines()
    new_lines = []
    
    # Simple state machine to find insertion points
    for line in lines:
        for key, doc in documentation_map.items():
            if key in line and "# DOCUMENTATION INJECTED" not in "".join(new_lines):
                 # Add the documentation block before the key line
                 new_lines.append(f"\n# =============================================================================\n")
                 new_lines.append(f"# {doc.upper()}\n")
                 new_lines.append(f"# =============================================================================\n")
        new_lines.append(line)
    
    return "\n".join(new_lines)


# Define documentation blocks
doc_profile = "SCHEMA DEFINITION: USER PROFILE\n# Defines the structure for storing general user information.\n# This Pydantic model ensures the LLM extracts data in this exact format."
doc_todo = "SCHEMA DEFINITION: TODO ITEM\n# Defines the structure for a single actionable task.\n# The 'status' field is restricted to specific Literal values to ensure consistency."
doc_extractor = "EXTRACTOR SETUP: PROFILE\n# Wraps the LLM with Trustcall's logic.\n# It forces the model to look for 'Profile' updates in the conversation."
doc_system_msg = "SYSTEM PROMPT: THE BRAIN\n# This is the main personality of the agent.\n# It injects the CURRENT memory state into the context so the model knows what it already knows."
doc_trustcall_instr = "SYSTEM PROMPT: EXTRACTOR INSTRUCTIONS\n# Specific instructions for the specialized extractor model.\n# Tells it to 'patch' existing JSON documents rather than overwriting them."
doc_task_maistro = "NODE A: THE DECISION MAKER (task_mAIstro)\n# This node is the 'brain'.\n# It reads all memories, looks at the new message, and DECIDES if memory needs updating.\n# It does NOT update memory itself; it just calls the 'UpdateMemory' tool."
doc_update_profile = "NODE B: THE WORKER (update_profile)\n# This node executes the profile update.\n# 1. Loads current profile.\n# 2. Uses Trustcall to calculate the 'patch' (diff).\n# 3. Saves the new version to the database."
doc_update_todos = "NODE C: THE WORKER (update_todos)\n# Similar to update_profile, but for the ToDo list.\n# Uses 'enable_inserts=True' to allow adding NEW items to the list."
doc_update_instructions = "NODE D: META-MEMORY (update_instructions)\n# Updates the user's PREFERENCES on how to handle tasks.\n# This is 'memory about memory' logic."
doc_route = "ROUTING LOGIC: THE TRAFFIC COP\n# Reads the output of 'task_mAIstro'.\n# If the brain decided to update memory (tool call), this routes execution to the correct worker."
doc_graph = "GRAPH CONSTRUCTION\n# Assembles the nodes into a workflow.\n# Configures the storage (MemoryStore) for long-term persistence."

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        full_source = "".join(cell['source'])
        
        # Inject documentation based on unique identifiers in the code
        if "class Profile(BaseModel):" in full_source:
             if "# SCHEMA DEFINITION: USER PROFILE" not in full_source:
                # We rebuild the source manually to place things nicely
                cell['source'] = [
                    "# =============================================================================\n",
                    "# 1. SCHEMA DEFINITION: USER PROFILE\n",
                    "# Defines the structure for storing general user information.\n",
                    "# This Pydantic model ensures the LLM extracts data in this exact format.\n",
                    "# =============================================================================\n",
                    full_source
                ]
        
        if "class ToDo(BaseModel):" in full_source:
             if "# SCHEMA DEFINITION: TODO ITEM" not in full_source:
                 # Find the index of the class def to insert before it
                 lines = full_source.splitlines(keepends=True)
                 new_source = []
                 for line in lines:
                     if "class ToDo(BaseModel):" in line:
                         new_source.append("\n# =============================================================================\n")
                         new_source.append("# 2. SCHEMA DEFINITION: TODO ITEM\n")
                         new_source.append("# Defines the structure for a single actionable task.\n")
                         new_source.append("# The 'status' field is restricted to specific Literal values for consistency.\n")
                         new_source.append("# =============================================================================\n")
                     new_source.append(line)
                 cell['source'] = new_source

        if "profile_extractor = create_extractor(" in full_source:
             if "# EXTRACTOR SETUP" not in full_source:
                 lines = full_source.splitlines(keepends=True)
                 new_source = []
                 for line in lines:
                     if "profile_extractor = create_extractor(" in line:
                         new_source.append("\n# =============================================================================\n")
                         new_source.append("# 3. EXTRACTOR SETUP: PROFILE\n")
                         new_source.append("# Wraps the LLM with Trustcall's logic.\n")
                         new_source.append("# It forces the model to look for 'Profile' updates in the conversation.\n")
                         new_source.append("# =============================================================================\n")
                     new_source.append(line)
                 cell['source'] = new_source

        if "MODEL_SYSTEM_MESSAGE = " in full_source:
              if "# SYSTEM PROMPT" not in full_source:
                 lines = full_source.splitlines(keepends=True)
                 new_source = []
                 for line in lines:
                     if "MODEL_SYSTEM_MESSAGE = " in line:
                         new_source.append("# =============================================================================\n")
                         new_source.append("# 4. SYSTEM PROMPT: THE BRAIN\n")
                         new_source.append("# This is the main personality of the agent.\n")
                         new_source.append("# It injects the CURRENT memory state into the context so the model knows what it already knows.\n")
                         new_source.append("# =============================================================================\n")
                     new_source.append(line)
                 cell['source'] = new_source

        if "def task_mAIstro(" in full_source:
             if "# NODE A" not in full_source:
                 lines = full_source.splitlines(keepends=True)
                 new_source = []
                 for line in lines:
                     if "def task_mAIstro(" in line:
                         new_source.append("\n# =============================================================================\n")
                         new_source.append("# 5. NODE A: THE DECISION MAKER (task_mAIstro)\n")
                         new_source.append("# This node is the 'brain'.\n")
                         new_source.append("# It reads all memories, looks at the new message, and DECIDES if memory needs updating.\n")
                         new_source.append("# It does NOT update memory itself; it just calls the 'UpdateMemory' tool.\n")
                         new_source.append("# =============================================================================\n")
                     new_source.append(line)
                 cell['source'] = new_source

        if "def update_profile(" in full_source:
             if "# NODE B" not in full_source:
                 lines = full_source.splitlines(keepends=True)
                 new_source = []
                 for line in lines:
                     if "def update_profile(" in line:
                         new_source.append("\n# =============================================================================\n")
                         new_source.append("# 6. NODE B: THE WORKER (update_profile)\n")
                         new_source.append("# This node executes the profile update.\n")
                         new_source.append("# 1. Loads current profile.\n")
                         new_source.append("# 2. Uses Trustcall to calculate the 'patch' (diff) between chat and existing data.\n")
                         new_source.append("# 3. Saves the new version to the database.\n")
                         new_source.append("# =============================================================================\n")
                     new_source.append(line)
                 cell['source'] = new_source
        
        if "def update_todos(" in full_source:
             if "# NODE C" not in full_source:
                 lines = full_source.splitlines(keepends=True)
                 new_source = []
                 for line in lines:
                     if "def update_todos(" in line:
                         new_source.append("\n# =============================================================================\n")
                         new_source.append("# 7. NODE C: THE WORKER (update_todos)\n")
                         new_source.append("# Similar to update_profile, but for the ToDo list.\n")
                         new_source.append("# Uses 'enable_inserts=True' to allow adding NEW items to the list.\n")
                         new_source.append("# =============================================================================\n")
                     new_source.append(line)
                 cell['source'] = new_source

        if "def route_message(" in full_source:
             if "# ROUTING LOGIC" not in full_source:
                 lines = full_source.splitlines(keepends=True)
                 new_source = []
                 for line in lines:
                     if "def route_message(" in line:
                         new_source.append("\n# =============================================================================\n")
                         new_source.append("# 8. ROUTING LOGIC: THE TRAFFIC COP\n")
                         new_source.append("# Reads the output of 'task_mAIstro'.\n")
                         new_source.append("# If the brain decided to update memory (tool call), this routes execution to the correct worker.\n")
                         new_source.append("# =============================================================================\n")
                     new_source.append(line)
                 cell['source'] = new_source

        if "builder = StateGraph(" in full_source:
             if "# GRAPH CONSTRUCTION" not in full_source:
                 lines = full_source.splitlines(keepends=True)
                 new_source = []
                 for line in lines:
                     if "builder = StateGraph(" in line:
                         new_source.append("\n# =============================================================================\n")
                         new_source.append("# 9. GRAPH CONSTRUCTION\n")
                         new_source.append("# Assembles the nodes into a workflow.\n")
                         new_source.append("# Configures the storage (MemoryStore) for long-term persistence.\n")
                         new_source.append("# =============================================================================\n")
                     new_source.append(line)
                 cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
