import json

notebook_path = '/home/juansebas7ian/langchain-academy/module-5/memory_agent.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        full_source = "".join(cell['source'])
        
        # FIX: ChatBedrockConverse does not support parallel_tool_calls=False in bind_tools
        # We need to remove it from the task_mAIstro function
        if "parallel_tool_calls=False" in full_source:
             new_source = full_source.replace(", parallel_tool_calls=False", "")
             # Handle case where it might be the only argument or last argument with different spacing
             new_source = new_source.replace("parallel_tool_calls=False", "") 
             cell['source'] = new_source.splitlines(keepends=True)

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
