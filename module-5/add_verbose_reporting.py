import json

notebook_path = '/home/juansebas7ian/langchain-academy/module-5/memory_agent.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        full_source = "".join(cell['source'])
        
        # 1. ADD 'spy = Spy()' to update_profile so we can track the changes
        if "result = profile_extractor.invoke" in full_source and "spy = Spy()" not in full_source:
             lines = full_source.splitlines()
             new_lines = []
             for line in lines:
                 if "result = profile_extractor.invoke" in line:
                     new_lines.append("# Initialize Spy for monitoring (Added for Verbose Reporting)")
                     new_lines.append("    spy = Spy()")
                     new_lines.append("    # Attach Spy listener to extractor")
                     new_lines.append("    profile_extractor_with_spy = profile_extractor.with_listeners(on_end=spy)")
                     new_lines.append(line.replace("profile_extractor.invoke", "profile_extractor_with_spy.invoke"))
                 else:
                     new_lines.append(line)
             full_source = "\n".join(new_lines)
        
        # 2. MODIFY RETURN MESSAGE in update_profile to include the extracted info
        if 'return {"messages": [{"role": "tool", "content": "updated profile", "tool_call_id":tool_calls[0][\'id\']}]}' in full_source:
             new_return = (
                 '    # Extract detailed changes using the spy\\n'
                 '    profile_update_msg = extract_tool_info(spy.called_tools, tool_name)\\n'
                 '    return {"messages": [{"role": "tool", "content": profile_update_msg, "tool_call_id":tool_calls[0][\'id\']}]}'
             )
             full_source = full_source.replace(
                 'return {"messages": [{"role": "tool", "content": "updated profile", "tool_call_id":tool_calls[0][\'id\']}]}', 
                 new_return
             )
             cell['source'] = full_source.splitlines(keepends=True)

        # 3. VERIFY update_todos already has spy, just ensure it uses extract_tool_info
        # (It seems it already does from the user's previous code, but let's double check)
        # If 'todo_update_msg = extract_tool_info' is already there, we are good.

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
