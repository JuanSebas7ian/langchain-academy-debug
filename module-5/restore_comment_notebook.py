import json

notebook_path = '/home/juansebas7ian/langchain-academy/module-5/memory_agent.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        full_source = "".join(cell['source'])
        
        # RESTORE AS COMMENT: We previously removed it, now we put it back but commented out
        # We look for the bind_tools call without the argument
        if "model.bind_tools([UpdateMemory])" in full_source:
             new_source = full_source.replace(
                 "model.bind_tools([UpdateMemory])", 
                 "model.bind_tools([UpdateMemory] #, parallel_tool_calls=False # Bedrock Converse does not support this parameter\n    )"
             )
             cell['source'] = new_source.splitlines(keepends=True)
        elif "model.bind_tools([UpdateMemory]).invoke" in full_source:
             new_source = full_source.replace(
                 "model.bind_tools([UpdateMemory]).invoke", 
                 "model.bind_tools([UpdateMemory] #, parallel_tool_calls=False\n    ).invoke"
             )
             cell['source'] = new_source.splitlines(keepends=True)


with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
