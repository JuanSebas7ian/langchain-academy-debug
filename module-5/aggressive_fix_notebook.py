import json

notebook_path = '/home/juansebas7ian/langchain-academy/module-5/memory_agent.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        full_source = "".join(cell['source'])
        
        # We need to find the specific bind_tools call and clean it up completely
        if "bind_tools" in full_source and "UpdateMemory" in full_source:
             lines = full_source.splitlines()
             new_lines = []
             for line in lines:
                 if "bind_tools" in line:
                     # This handles both cases:
                     # parallel_tool_calls=False (no spaces)
                     # parallel_tool_calls = False (spaces)
                     
                     # First, check if it's there
                     if "parallel_tool_calls" in line:
                         # Replace it with an empty string or just the closing bracket if needed
                         # Case 1:  response = model.bind_tools([UpdateMemory], parallel_tool_calls=False).invoke...
                         new_line = line.replace(", parallel_tool_calls=False", "")
                         new_line = new_line.replace(", parallel_tool_calls = False", "")
                         
                         # Case 2: Just strict replacement 
                         new_line = new_line.replace("parallel_tool_calls=False", "") 
                         new_line = new_line.replace("parallel_tool_calls = False", "")
                         
                         # Add a comment to be transparent
                         if "# Bedrock fix" not in new_line:
                              new_line = new_line.rstrip() + " # parallel_tool_calls=False removed for Bedrock compat"
                         
                         new_lines.append(new_line)
                     else:
                         new_lines.append(line)
                 else:
                     new_lines.append(line)
             
             cell['source'] = [l + "\n" for l in new_lines]


with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
