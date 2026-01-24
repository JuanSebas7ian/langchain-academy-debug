import json
import re

notebook_path = '/home/juansebas7ian/langchain-academy/module-5/memory_agent_llama.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# The "Nuclear" and robust monkeypatch
monkeypatch_code = """
from langchain_aws import ChatBedrockConverse
import langchain_aws.chat_models.bedrock_converse as bc

# FORCE RESET: Always restore the library's original method first to clear any old patches/recursion
ChatBedrockConverse.bind_tools = bc.ChatBedrockConverse.bind_tools

# CAPTURE ORIGINAL: This is now guaranteed to be the clean library version
_lib_original_bind_tools = ChatBedrockConverse.bind_tools

def _patched_bind_tools(self, tools, tool_choice=None, **kwargs):
    # Determine model name safely using the internal method
    model_name = self._get_base_model()
    if "llama" in model_name.lower():
        if tool_choice not in [None, "auto"]:
            # print(f"⚠️ Intercepted tool_choice='{tool_choice}', forcing 'auto' for Llama compatibility")
            tool_choice = "auto"
            
    return _lib_original_bind_tools(self, tools, tool_choice=tool_choice, **kwargs)

ChatBedrockConverse.bind_tools = _patched_bind_tools
"""

found_cell = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        full_source = "".join(cell['source'])
        
        # Look for the setup cell
        if 'from langchain_aws import ChatBedrockConverse' in full_source:
             # Remove ALL previous monkeypatch attempts (regex covers different versions)
             full_source = re.sub(r'# MONKEYPATCH:.*?ChatBedrockConverse\.bind_tools = _patched_bind_tools', '', full_source, flags=re.DOTALL)
             full_source = re.sub(r'if not hasattr\(ChatBedrockConverse, "_original_bind_tools"\):.*?ChatBedrockConverse\.bind_tools = _patched_bind_tools', '', full_source, flags=re.DOTALL)
             
             # Add the nuclear patch
             full_source = monkeypatch_code.strip() + "\n\n" + full_source.strip()
             found_cell = True

        new_lines = full_source.splitlines(keepends=True)
        cell['source'] = new_lines

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
